#!/usr/bin/env python3
"""
Qwen3-0.6B 分层隐状态特征抽取器
 - 支持超长 Markdown 报告分块
 - 对每层做 mean pooling 得到文档级表征
 - 可选择仅前向到指定中间层（加速推理）

注意：暂不集成 Venn-Abers 校准。
"""

import os
import math
from typing import List, Tuple, Optional, Dict
import numpy as np

# 需在导入 transformers 前禁用 TF/Flax 以避免可选依赖触发及 numpy 冲突
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LayerwiseFeatureExtractor:
    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        enable_hidden_states: bool = True,
    ) -> None:
        # 上面已在模块层设置镜像与禁用TF/Flax，这里留作冗余确保生效
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.dtype = dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            output_hidden_states=enable_hidden_states,
        )
        self.model.to(self.device)
        self.model.eval()

        # 统计层数与隐层维度
        with torch.no_grad():
            tmp = self.tokenizer("test", return_tensors="pt").to(self.model.device)
            out = self.model(**tmp)
            hidden_states = out.hidden_states  # tuple: embedding + N层
            self.num_layers = len(hidden_states) - 1
            self.hidden_size = hidden_states[-1].shape[-1]

    def _forward_until_layer(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, until_layer: Optional[int]) -> List[torch.Tensor]:
        """
        前向到指定层，返回 1..L 的隐藏层列表（不含 embedding 层）。
        until_layer: 若为 None 或 > num_layers，则使用全部层。
        """
        if until_layer is None or until_layer > self.num_layers:
            until_layer = self.num_layers

        # 直接利用模型完整前向，再切片取前 until_layer 层，简单稳妥
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hs = out.hidden_states  # len = 1 + num_layers
        return list(hs[1: until_layer + 1])

    @staticmethod
    def _mask_mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        对 [B, T, H] 的隐藏层按 mask 做均值池化 -> [B, H]
        """
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def encode_document_layers(
        self,
        text: str,
        max_tokens: int = 1024,
        stride: int = 256,
        until_layer: Optional[int] = None,
    ) -> torch.Tensor:
        """
        返回文档级分层向量: [L, H]
        - 将长文本按 window/stride 分块
        - 每块对每层做 mean pooling，再对块做均值
        - until_layer: 仅取前 L 层
        """
        enc = self.tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = enc["input_ids"][0]
        attention_full = torch.ones_like(input_ids)

        window = max_tokens
        step = max(1, window - stride)

        chunk_layer_vecs: List[torch.Tensor] = []  # [num_chunks, L, H]

        for start in range(0, input_ids.size(0), step):
            end = min(start + window, input_ids.size(0))
            chunk_ids = input_ids[start:end].unsqueeze(0).to(self.model.device)
            chunk_attn = attention_full[start:end].unsqueeze(0).to(self.model.device)

            layer_hiddens = self._forward_until_layer(chunk_ids, chunk_attn, until_layer)
            # 每层做 mask-mean -> [B, H] 再 squeeze -> [H]
            pooled_layers = [self._mask_mean_pool(h, chunk_attn).squeeze(0) for h in layer_hiddens]
            chunk_layer_vecs.append(torch.stack(pooled_layers, dim=0))  # [L, H]

            if end == input_ids.size(0):
                break

        doc_layers = torch.stack(chunk_layer_vecs, dim=0).mean(dim=0)  # [L, H]
        # 转为 float32 以便后续 numpy 处理
        return doc_layers.detach().to(torch.float32).cpu()

    # ==========================
    # Token-level attribution
    # ==========================
    def _collect_layer_token_hiddens(
        self,
        text: str,
        layer_index: int,
        max_tokens: int = 1024,
        stride: int = 256,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Collect token-level hidden states at a specific layer for the entire document.

        Returns:
          - token_hiddens: [T, H] float32 on CPU
          - token_ids: list of token ids length T

        Notes:
          - Uses non-overlapping windows to avoid duplicated tokens.
          - For very long docs (> max_tokens * n_windows), windows are concatenated.
        """
        assert layer_index >= 1, "layer_index is 1-based"
        enc = self.tokenizer(text, return_tensors="pt", truncation=False)
        full_ids: torch.Tensor = enc["input_ids"][0]

        window = max_tokens
        # Use non-overlap to keep one representation per token
        step = max_tokens

        collected: List[torch.Tensor] = []
        collected_ids: List[int] = []

        for start in range(0, full_ids.size(0), step):
            end = min(start + window, full_ids.size(0))
            chunk_ids = full_ids[start:end].unsqueeze(0).to(self.model.device)
            attn = torch.ones_like(chunk_ids)
            with torch.no_grad():
                out = self.model(input_ids=chunk_ids, attention_mask=attn)
                hs = out.hidden_states  # len = 1 + L
                lyr = hs[layer_index]  # [B, T, H]
            collected.append(lyr.squeeze(0).to(torch.float32).cpu())
            collected_ids.extend(chunk_ids.squeeze(0).tolist())
            if end == full_ids.size(0):
                break

        token_hiddens = torch.cat(collected, dim=0) if collected else torch.empty(0, self.hidden_size)
        return token_hiddens, collected_ids

    @staticmethod
    def _tokens_to_text_spans(tokenizer: AutoTokenizer, token_ids: List[int]) -> List[str]:
        """
        Convert token ids to displayable spans with reasonable spacing handling for SentencePiece/BPE.
        """
        toks = tokenizer.convert_ids_to_tokens(token_ids)
        spans: List[str] = []
        for t in toks:
            if t is None:
                spans.append("")
                continue
            # Common SentencePiece convention: '▁' denotes a leading space
            if t.startswith("▁"):
                spans.append(t.replace("▁", " "))
            else:
                spans.append(t)
        return spans

    def explain_token_attribution(
        self,
        text: str,
        layer_index: int,
        linear_weight: np.ndarray,
        bias: float = 0.0,
        method: str = "grad_x_input",
        max_tokens: int = 1024,
        stride: int = 256,
        normalize: bool = True,
        shap_samples: int = 50,
    ) -> Dict[str, object]:
        """
        Compute token-level attributions for a linear head: z = w·mean(H_t) + b.

        - method='grad': gradient magnitude ||∂z/∂H_t||_2 = ||w||_2 / T (uniform across tokens, not informative)
        - method='grad_x_input' (recommended): (w·H_t)/T as token contribution  
        - method='shap': SHAP values using token masking (most accurate but slower)

        Returns dict with:
          - 'tokens': list[str]
          - 'scores': list[float]
          - 'logit': float (document logit using mean pooling)
        """
        assert method in {"grad", "grad_x_input", "shap"}
        token_hiddens, token_ids = self._collect_layer_token_hiddens(
            text=text, layer_index=layer_index, max_tokens=max_tokens, stride=stride
        )
        if token_hiddens.numel() == 0:
            return {"tokens": [], "scores": [], "logit": float(bias)}

        T, H = token_hiddens.shape
        w = np.asarray(linear_weight, dtype=np.float32).reshape(-1)
        assert w.shape[0] == H, f"weight dim {w.shape[0]} != hidden_size {H}"

        # mean pooled document vector
        doc_vec = token_hiddens.mean(dim=0).numpy()  # [H]
        logit = float(np.dot(w, doc_vec) + bias)

        if method == "grad":
            # ||w|| / T, uniform scores; keep informational but not very useful visually
            grad_norm = float(np.linalg.norm(w, ord=2)) / float(T)
            scores = np.full((T,), grad_norm, dtype=np.float32)
        elif method == "grad_x_input":
            # Grad×Input -> (w·H_t)/T
            scores = (token_hiddens.numpy() @ w) / float(T)  # [T]
        else:
            # SHAP attribution using token masking
            scores = self._compute_shap_attributions(
                token_hiddens, w, bias, shap_samples
            )

        # Optional normalization to [-1, 1]
        if normalize and scores.size > 0:
            max_abs = float(np.max(np.abs(scores)))
            if max_abs > 1e-8:
                scores = scores / max_abs

        tokens = self._tokens_to_text_spans(self.tokenizer, token_ids)
        return {"tokens": tokens, "scores": scores.tolist(), "logit": logit}

    def _compute_shap_attributions(
        self, 
        token_hiddens: torch.Tensor, 
        weight: np.ndarray, 
        bias: float,
        n_samples: int = 50
    ) -> np.ndarray:
        """
        计算基于token masking的SHAP值
        """
        import shap
        
        T, H = token_hiddens.shape
        hiddens_np = token_hiddens.numpy()
        
        def model_func(masked_hiddens):
            """模型函数：输入masked token hiddens，输出logit"""
            # masked_hiddens: [n_samples, T, H]
            if len(masked_hiddens.shape) == 2:
                masked_hiddens = masked_hiddens.reshape(1, T, H)
            
            logits = []
            for sample in masked_hiddens:
                # 对每个样本计算mean pooling后的logit
                mean_vec = np.mean(sample, axis=0)  # [H]
                logit = np.dot(weight, mean_vec) + bias
                logits.append(logit)
            
            return np.array(logits)
        
        # 创建SHAP explainer
        # 使用零向量作为baseline
        baseline = np.zeros((1, T, H))
        explainer = shap.Explainer(model_func, baseline)
        
        # 计算SHAP值
        input_data = hiddens_np.reshape(1, T, H)
        shap_values = explainer(input_data, max_evals=n_samples)
        
        # 聚合每个token的SHAP值（对H维度求和）
        token_shap = np.sum(shap_values.values[0], axis=1)  # [T]
        
        return token_shap

    @staticmethod
    def render_attribution_html(tokens: List[str], scores: List[float]) -> str:
        """
        Render tokens with background colors based on scores in [-1,1].
        Positive -> red, Negative -> blue, magnitude -> opacity.
        """
        html_parts: List[str] = [
            "<div style=\"font-family:monospace; line-height:1.6; white-space:pre-wrap;\">"
        ]
        for tok, s in zip(tokens, scores):
            a = min(1.0, max(0.0, abs(float(s))))
            if s >= 0:
                color = f"rgba(255,0,0,{a:.2f})"
            else:
                color = f"rgba(0,102,255,{a:.2f})"
            safe_tok = tok.replace("<", "&lt;").replace(">", "&gt;")
            html_parts.append(f"<span style=\"background:{color};\">{safe_tok}</span>")
        html_parts.append("</div>")
        return "".join(html_parts)



def encode_corpus(
    extractor: LayerwiseFeatureExtractor,
    documents: List[str],
    max_tokens: int = 1024,
    stride: int = 256,
    until_layer: Optional[int] = None,
) -> List[torch.Tensor]:
    """对一组文档进行编码，返回每个文档的 [L, H] 张量列表。"""
    features = []
    for text in documents:
        feats = extractor.encode_document_layers(
            text=text,
            max_tokens=max_tokens,
            stride=stride,
            until_layer=until_layer,
        )
        features.append(feats)
    return features


__all__ = ["LayerwiseFeatureExtractor", "encode_corpus"]


