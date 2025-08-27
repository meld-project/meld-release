
# MELD 实验复现操作手册

## 第一步：环境安装

### 1. 安装 uv（如果还没安装）

```bash
# 使用 curl 安装（推荐）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或者使用 pip 安装
pip install uv

# 或者使用 homebrew（macOS）
brew install uv
```

### 2. 同步依赖

```bash
# 同步依赖
uv sync
```

### 3. 验证环境

```bash
# 验证核心依赖
uv run python -c "
import torch, transformers, sklearn, pandas, numpy
print('✅ 环境配置成功')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"
```

## 第二步：模型准备

### 1. 下载 Qwen 模型

```bash
# 下载 Qwen3-0.6B 模型
uv run python scripts/download_models.py
```

### 2. 验证模型路径

```bash
# 检查模型是否下载成功
ls -la models/
```

## 第三步：数据准备

### 1. 准备输入数据

将 CAPE JSON 数据放入指定目录：
```bash
# 恶意样本（按家族分组）
mkdir -p input/cape_behavior_benign_train

# 良性样本（平铺）
mkdir -p input/cape_behavior_benign_train

# 从数据集仓库下载数据集
# 必要文件：malicious_dataset_manifest.csv
# 必要文件：cape_reports.7z.* -> cape_reports
uv run python scripts/copy_correct_cape_samples.py --manifest malicious_dataset_manifest.csv  --source-malicious cape_reports --source-benign cape_reports --target-malicious input/cape_behavior_malicious_train --target-benign input/cape_behavior_benign_train
```

### 2. 数据转换

```bash
# 转换恶意样本到 Markdown
uv run python scripts/batch_convert_cape_data.py \
  --input input/cape_behavior_malicious_train \
  --output data/processed/cape_reports_malicious_md \
  --workers 4

# 转换良性样本到 Markdown
uv run python scripts/batch_convert_cape_data.py \
  --input input/cape_behavior_benign_train \
  --output data/processed/cape_reports_benign_md \
  --workers 4
```

### 3. 数据预处理

```bash
# 生成数据集索引
uv run python scripts/preprocess_data.py \
  --input data/processed \
  --output data/processed

# 验证数据预处理结果
ls -la data/processed/
cat data/processed/dataset_stats.json
```

## 第四步：系统验证

**目的**：验证MELD核心功能是否正常，为后续正式实验做准备

### 使用验证脚本（推荐）

```bash
# 运行MELD系统验证脚本
./scripts/verify_meld_system.sh
```

这个脚本会自动完成以下验证步骤：
1. **环境验证** - 检查Python版本和所有依赖包
2. **模型路径检查** - 验证Qwen模型是否可用
3. **数据完整性检查** - 检查输入数据和处理后的数据
4. **MELD功能验证** - 测试特征提取和分类功能
5. **生成验证报告** - 创建详细的验证报告

### 查看验证结果

```bash
# 查看验证报告
cat results/meld_system_verification_report.md

# 查看详细的JSON结果
cat results/meld_verification.json

# 检查所有结果文件
ls -la results/
```


## 第五步：实验执行

### 1. Time-OOD 实验

```bash
# 运行时间分布外检测实验
uv run python experiments/time_ood/run_time_ood.py \
  --data_dir data/processed \
  --model_dir models/qwen3-0.6b \
  --output results/time_ood_results.json
```

### 2. Family-OOD 实验

#### 单个家族实验

```bash
# 运行家族分布外检测实验（以 AgentTesla 为例）
uv run python experiments/family_ood/run_family_ood.py \
  --target_family AgentTesla \
  --data_dir data/processed \
  --model_dir models/qwen3-0.6b

# 快速测试：使用小样本量（推荐用于初次验证）
uv run python experiments/family_ood/run_family_ood.py \
  --target_family AgentTesla \
  --max_samples 100

# 精细控制样本数量
uv run python experiments/family_ood/run_family_ood.py \
  --target_family AsyncRAT \
  --max_train_samples 500 \
  --max_val_samples 50 \
  --max_test_samples 200

# 跳过报告生成
uv run python experiments/family_ood/run_family_ood.py \
  --target_family Formbook \
  --max_samples 100 \
  --no-report
```

#### 批量实验（推荐）

```bash
# 运行所有 Top-8 家族的批量实验（自动生成汇总报告）
uv run python experiments/family_ood/run_family_ood.py \
  --batch \
  --data_dir data/processed \
  --model_dir models/qwen3-0.6b

# 批量快速测试（带样本限制，推荐用于资源有限环境）
uv run python experiments/family_ood/run_family_ood.py \
  --batch \
  --max_samples 200

# 批量实验跳过报告生成
uv run python experiments/family_ood/run_family_ood.py \
  --batch \
  --max_samples 100 \
  --no-report

# 自定义批量汇总报告路径
uv run python experiments/family_ood/run_family_ood.py \
  --batch \
  --max_samples 300 \
  --batch_report results/custom_family_summary.md
```

**参数说明：**

**样本限制参数：**
- `--max_samples N`: 设置所有样本限制为N（便捷参数）
- `--max_train_samples N`: 最大训练样本数
- `--max_val_samples N`: 最大验证样本数  
- `--max_test_samples N`: 最大测试样本数
- 不指定限制参数时使用所有可用样本

**报告生成参数：**
- `--report FILE`: 指定单个实验报告文件路径（自动生成如不指定）
- `--no-report`: 跳过报告生成
- `--batch_report FILE`: 指定批量实验汇总报告路径

**批量实验参数：**
- `--batch`: 运行所有8个家族的批量实验
- 批量模式会自动生成每个家族的个人报告 + 一个汇总对比报告

**实验输出：**
- **JSON结果**: `results/family_ood/meld_family_ood_{family}.json`
- **个人报告**: `results/family_ood/family_ood_{family}_report.md`
- **汇总报告**: `results/family_ood/family_ood_summary_report.md`


## 第六步：结果分析

### 1. Time-OOD 实验结果分析

```bash
# 查看 Time-OOD JSON 结果
cat results/time_ood_results.json

# 查看 Time-OOD Markdown 报告（自动生成）
cat results/time_ood/time_ood_report.md

# 查看详细的 JSON 结果文件
ls results/time_ood/
```

**Time-OOD 结果解读：**
- **Macro F1-Score**：整体分类性能
- **AUROC**：ROC曲线下面积，衡量分类器区分能力
- **AUPR**：PR曲线下面积，适用于不平衡数据
- **最优阈值**：用于二分类的决策阈值

### 2. Family-OOD 实验结果分析

```bash
# 查看所有家族的实验结果
ls results/family_ood/

# 查看汇总报告（推荐）
cat results/family_ood/family_ood_summary_report.md

# 查看单个家族的详细报告
cat results/family_ood/family_ood_agenttesla_report.md
cat results/family_ood/family_ood_asyncrat_report.md
# ... 其他家族

# 查看原始 JSON 结果
cat results/family_ood/meld_family_ood_agenttesla.json
```

**Family-OOD 结果解读：**
- **每个家族作为 OOD**：测试模型对未见过家族的泛化能力
- **性能对比**：不同家族的检测难度差异
- **汇总统计**：所有家族的平均性能


## 第七步：高级实验

### 1. 层级分析实验

```bash
# 测试不同层的特征效果
python scripts/test_mistral_all_layers.py

# 无随机化的层级测试
python scripts/test_mistral_all_layers_norandom.py
```

### 2. 批量层级评估

```bash
# 运行 Layer 15 实验
./scripts/run_rq2_layer15.sh

# 运行 Layer 28 实验
./scripts/run_rq2_layer28.sh
```

### 3. OOD 评估实验

```bash
# 安装
uv pip install -e .

# 运行 OOD 评估
uv run python src/meld/ood_eval.py \
  --index_csv data/processed/dataset_with_family_time.csv \
  --model_dir models/qwen3-0.6b \
  --mode family \
  --test_family AgentTesla \
  --out results/ood_evaluation.json
```

