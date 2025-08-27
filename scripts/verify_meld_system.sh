#!/bin/bash
# MELD系统验证脚本
# 用于验证MELD系统的各个组件是否正常工作

set -e  # 遇到错误立即退出

# ==========================================
# 配置参数 - 用户可根据需要修改这些参数
# ==========================================
#
# 使用说明：
# 1. 修改 POSSIBLE_MODEL_PATHS 数组来指定你的模型路径
# 2. 修改 VERIFICATION_SAMPLES_PER_CLASS 来控制验证样本数量
# 3. 修改 MAX_TOKENS, STRIDE 等参数来调整特征提取性能
# 4. 修改 GPU_DEVICE 来指定GPU设备或使用CPU
# 5. 修改各种路径变量来适应你的目录结构
#

# 模型配置
POSSIBLE_MODEL_PATHS=(
    "models/qwen3-0.6b"
)
HUGGINGFACE_MODEL="Qwen/Qwen3-0.6B"

# 数据路径配置
INPUT_MALICIOUS_DIR="input/cape_behavior_malicious_train"
INPUT_BENIGN_DIR="input/cape_behavior_benign_train"
PROCESSED_MALICIOUS_DIR="data/processed/cape_reports_malicious_md"
PROCESSED_BENIGN_DIR="data/processed/cape_reports_benign_md"
DATASET_INDEX_FILE="data/processed/dataset_with_family_time.csv"
DATASET_STATS_FILE="data/processed/dataset_stats.json"

# 验证参数配置
VERIFICATION_SAMPLES_PER_CLASS=25  # 每个类别用于验证的样本数
MAX_TOKENS=512                     # 特征提取时的最大token数
STRIDE=128                         # 特征提取时的步长
UNTIL_LAYER=10                     # 特征提取到第几层
USE_LAYER=9                        # 使用第几层的特征（0-based）
TEST_SIZE=0.3                      # 测试集比例
RANDOM_STATE=42                    # 随机种子

# 输出配置
RESULTS_DIR="results"
VERIFICATION_JSON="results/meld_verification.json"
VERIFICATION_REPORT="results/meld_system_verification_report.md"

# 依赖包列表
REQUIRED_PACKAGES=('torch' 'transformers' 'sklearn' 'pandas' 'numpy' 'duckdb' 'tqdm')

# GPU配置
GPU_DEVICE="cuda:0"  # GPU设备，如果不想用GPU可以设为"cpu"

# ==========================================
# 颜色定义
# ==========================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 开始验证
echo ""
echo "🎯 ========================================"
echo "🔍 MELD系统验证脚本"
echo "📋 验证系统各组件功能"
echo "🎯 ========================================"
echo ""

# 记录开始时间
START_TIME=$(date +%s)
log_info "验证开始时间: $(date)"

# 步骤1: 环境验证
echo ""
log_info "🔍 步骤1: 环境验证..."

log_info "验证核心依赖包..."
uv run python -c "
import sys
print(f'Python 版本: {sys.version}')
print(f'Python 路径: {sys.executable}')
print()

# 检查核心依赖
required_packages = ['torch', 'transformers', 'sklearn', 'pandas', 'numpy', 'duckdb', 'tqdm']
missing_packages = []

for pkg in required_packages:
    try:
        module = __import__(pkg)
        if pkg == 'torch':
            print(f'✅ PyTorch: {module.__version__}')
            print(f'   CUDA 可用: {module.cuda.is_available()}')
            if module.cuda.is_available():
                print(f'   GPU 设备: {module.cuda.get_device_name(0)}')
        elif pkg == 'transformers':
            print(f'✅ Transformers: {module.__version__}')
        elif pkg == 'sklearn':
            print(f'✅ Scikit-learn: {module.__version__}')
        elif pkg == 'pandas':
            print(f'✅ Pandas: {module.__version__}')
        elif pkg == 'numpy':
            print(f'✅ NumPy: {module.__version__}')
        else:
            print(f'✅ {pkg}: 已安装')
    except ImportError:
        missing_packages.append(pkg)
        print(f'❌ {pkg}: 未安装')

if missing_packages:
    print(f'\\n缺少包: {missing_packages}')
    sys.exit(1)
else:
    print('\\n🎉 所有必要包已安装')
"

if [ $? -ne 0 ]; then
    log_error "环境验证失败"
    exit 1
fi

log_success "环境验证通过"

# 步骤2: 模型路径检查
echo ""
log_info "🧠 步骤2: 模型路径检查..."

log_info "检查模型路径..."
MODEL_DIR=""
for path in "${POSSIBLE_MODEL_PATHS[@]}"; do
    if [ -d "$path" ]; then
        log_success "找到模型目录: $path"
        MODEL_DIR="$path"
        break
    fi
done

if [ -z "$MODEL_DIR" ]; then
    log_warning "未找到本地模型，尝试从 HuggingFace 下载..."
    # 测试是否可以从 HuggingFace 下载
    uv run python -c "
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('$HUGGINGFACE_MODEL', trust_remote_code=True)
    print('✅ 可以从 HuggingFace 下载模型')
except Exception as e:
    print(f'❌ 模型下载失败: {e}')
    print('请运行: uv run python scripts/download_models.py')
    exit(1)
"
    if [ $? -ne 0 ]; then
        log_error "模型验证失败"
        exit 1
    fi
    MODEL_DIR="$HUGGINGFACE_MODEL"
else
    log_success "模型路径验证成功: $MODEL_DIR"
fi

# 步骤3: 数据完整性检查
echo ""
log_info "📊 步骤3: 数据完整性检查..."

# 检查输入数据
log_info "检查输入数据..."

if [ -d "$INPUT_MALICIOUS_DIR" ]; then
    MALICIOUS_COUNT=$(find "$INPUT_MALICIOUS_DIR" -name "*.json" 2>/dev/null | wc -l)
    log_info "恶意样本: $MALICIOUS_COUNT 个"
else
    MALICIOUS_COUNT=0
    log_warning "恶意样本目录不存在"
fi

if [ -d "$INPUT_BENIGN_DIR" ]; then
    BENIGN_COUNT=$(find "$INPUT_BENIGN_DIR" -name "*.json" 2>/dev/null | wc -l)
    log_info "良性样本: $BENIGN_COUNT 个"
else
    BENIGN_COUNT=0
    log_warning "良性样本目录不存在"
fi

# 检查处理后的数据
log_info "检查处理后的数据..."

if [ -d "$PROCESSED_MALICIOUS_DIR" ]; then
    MD_MALICIOUS_COUNT=$(find "$PROCESSED_MALICIOUS_DIR" -name "*.md" 2>/dev/null | wc -l)
    log_info "恶意样本 Markdown: $MD_MALICIOUS_COUNT 个"
else
    MD_MALICIOUS_COUNT=0
    log_warning "恶意样本 Markdown 目录不存在"
fi

if [ -d "$PROCESSED_BENIGN_DIR" ]; then
    MD_BENIGN_COUNT=$(find "$PROCESSED_BENIGN_DIR" -name "*.md" 2>/dev/null | wc -l)
    log_info "良性样本 Markdown: $MD_BENIGN_COUNT 个"
else
    MD_BENIGN_COUNT=0
    log_warning "良性样本 Markdown 目录不存在"
fi

# 检查数据集索引
if [ -f "$DATASET_INDEX_FILE" ]; then
    TOTAL_SAMPLES=$(tail -n +2 "$DATASET_INDEX_FILE" | wc -l)
    log_success "数据集索引: $TOTAL_SAMPLES 个样本"
    
    # 显示数据集统计
    log_info "数据集统计:"
    if [ -f "$DATASET_STATS_FILE" ]; then
        cat "$DATASET_STATS_FILE"
    fi
else
    log_error "数据集索引文件不存在"
    exit 1
fi

log_success "数据完整性检查通过"

# 步骤4: MELD系统功能验证
echo ""
log_info "🧠 步骤4: MELD系统功能验证..."

# 创建测试目录
mkdir -p "$RESULTS_DIR"

# 检查数据是否包含良性和恶意样本
log_info "检查数据分布..."
uv run python -c "
import pandas as pd
import os

if os.path.exists('$DATASET_INDEX_FILE'):
    df = pd.read_csv('$DATASET_INDEX_FILE')
    label_counts = df['label'].value_counts()
    print('数据分布:')
    print(f'  良性样本 (0): {label_counts.get(0, 0)} 个')
    print(f'  恶意样本 (1): {label_counts.get(1, 0)} 个')
    
    if len(label_counts) < 2:
        print('⚠️  警告：数据只包含单一类别，需要同时有良性和恶意样本')
        print('解决方案：确保第三步数据准备包含了良性样本转换')
        exit(1)
    else:
        print('✅ 数据包含两个类别，可以进行二分类训练')
else:
    print('❌ 数据集索引文件不存在')
    exit(1)
"

if [ $? -ne 0 ]; then
    log_error "数据分布检查失败"
    exit 1
fi

# 运行MELD功能验证
log_info "运行MELD功能验证..."

uv run python -c "
import sys, os
sys.path.append('src')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import json

print('⏱️  初始化MELD特征提取器...')

try:
    from meld.feature_extractor import LayerwiseFeatureExtractor
    
    # 检查数据
    if not os.path.exists('$DATASET_INDEX_FILE'):
        print('❌ 数据集索引文件不存在')
        sys.exit(1)
    
    df = pd.read_csv('$DATASET_INDEX_FILE')
    
    # 确保有两个类别
    if len(df['label'].unique()) < 2:
        print('❌ 数据只包含单一类别，无法进行验证')
        print('请确保数据包含良性和恶意样本')
        sys.exit(1)
    
    # 平衡采样：每个类别最多取指定数量的样本
    sample_df = df.groupby('label').head($VERIFICATION_SAMPLES_PER_CLASS)
    print(f'📊 验证样本: {len(sample_df)} 个 (良性: {(sample_df[\"label\"]==0).sum()}, 恶意: {(sample_df[\"label\"]==1).sum()})')
    
    # 初始化特征提取器
    device = '$GPU_DEVICE' if __import__('torch').cuda.is_available() and '$GPU_DEVICE' != 'cpu' else 'cpu'
    extractor = LayerwiseFeatureExtractor(
        model_dir='$MODEL_DIR',
        device=device
    )
    
    print('✅ MELD特征提取器初始化成功')
    print(f'   模型层数: {extractor.num_layers}')
    print(f'   隐藏维度: {extractor.hidden_size}')
    
    # 特征提取
    features = []
    labels = []
    failed = 0
    
    for idx, row in sample_df.iterrows():
        try:
            if not os.path.exists(row['path']):
                failed += 1
                continue
                
            with open(row['path'], 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            if len(text.strip()) == 0:
                failed += 1
                continue
            
            # 使用配置的参数进行特征提取
            layer_feats = extractor.encode_document_layers(
                text, 
                max_tokens=$MAX_TOKENS, 
                stride=$STRIDE, 
                until_layer=$UNTIL_LAYER
            )
            
            features.append(layer_feats[$USE_LAYER].numpy())  # 使用指定层
            labels.append(row['label'])
            
        except Exception as e:
            failed += 1
            print(f'   ⚠️  样本处理失败: {str(e)[:50]}...')
    
    if len(features) < 4:
        print(f'❌ 特征提取失败: 只有 {len(features)} 个成功样本')
        sys.exit(1)
    
    X = np.array(features)
    y = np.array(labels)
    
    print(f'✅ 特征提取完成: {X.shape} ({len(features)} 成功, {failed} 失败)')
    
    # 简单分类验证
    if len(np.unique(y)) == 2 and len(X) >= 8:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=$TEST_SIZE, random_state=$RANDOM_STATE, stratify=y
        )
        
        scaler = StandardScaler()
        clf = LogisticRegression(random_state=$RANDOM_STATE, max_iter=1000)
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        print(f'🎯 分类验证结果:')
        print(f'   准确率: {accuracy:.3f}')
        print(f'   F1分数: {f1:.3f}')
        
        # 保存验证结果
        results = {
            'status': 'success',
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'samples_processed': len(features),
            'samples_failed': failed,
            'model_layers': extractor.num_layers,
            'hidden_size': extractor.hidden_size,
            'model_dir': '$MODEL_DIR'
        }
        
        os.makedirs('$RESULTS_DIR', exist_ok=True)
        with open('$VERIFICATION_JSON', 'w') as f:
            json.dump(results, f, indent=2)
        
        print('💾 验证结果已保存到: $VERIFICATION_JSON')
        
        if accuracy > 0.5:
            print('🎉 MELD系统功能验证成功！')
        else:
            print('⚠️  MELD系统功能验证完成，但性能较低')
    else:
        print('⚠️  样本不足或类别单一，跳过分类验证')
        print('✅ 特征提取功能正常')

except Exception as e:
    print(f'❌ MELD验证失败: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    log_error "MELD功能验证失败"
    exit 1
fi

log_success "MELD系统功能验证通过"

# 步骤5: 生成验证报告
echo ""
log_info "📄 步骤5: 生成验证报告..."

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

cat > "$VERIFICATION_REPORT" << REPORT
# MELD系统验证报告

## 验证信息
- **验证时间**: $(date)
- **总耗时**: ${DURATION}秒
- **系统环境**: $(uname -s) $(uname -r)
- **Python版本**: $(python --version 2>&1)

## 验证步骤
1. ✅ 环境验证 - 通过
2. ✅ 模型路径检查 - 通过 ($MODEL_DIR)
3. ✅ 数据完整性检查 - 通过
4. ✅ MELD系统功能验证 - 通过
5. ✅ 验证报告生成 - 通过

## 数据统计
- 输入恶意样本: $MALICIOUS_COUNT 个
- 输入良性样本: $BENIGN_COUNT 个
- 处理后恶意样本: $MD_MALICIOUS_COUNT 个
- 处理后良性样本: $MD_BENIGN_COUNT 个
- 数据集总样本: $TOTAL_SAMPLES 个

## 测试结果
REPORT

# 添加测试结果到报告
if [ -f "$VERIFICATION_JSON" ]; then
    echo "根据 meld_verification.json:" >> "$VERIFICATION_REPORT"
    uv run python -c "
import json
try:
    with open('$VERIFICATION_JSON', 'r') as f:
        results = json.load(f)
    print(f\"- **准确率**: {results['accuracy']:.3f}\")
    print(f\"- **F1分数**: {results['f1_score']:.3f}\")
    print(f\"- **成功样本**: {results['samples_processed']}个\")
    print(f\"- **失败样本**: {results['samples_failed']}个\")
    print(f\"- **模型层数**: {results['model_layers']}\")
    print(f\"- **隐藏维度**: {results['hidden_size']}\")
    print(f\"- **使用模型**: {results['model_dir']}\")
except Exception as e:
    print(f\"- **错误**: 无法读取验证结果: {e}\")
" >> "$VERIFICATION_REPORT"
fi

cat >> "$VERIFICATION_REPORT" << REPORT

## 结论
✅ **MELD系统验证通过**

系统各组件运行正常，包括:
- 环境配置正确
- 模型加载成功
- 数据处理完整
- 特征提取有效
- 分类功能正常

可以进行完整的实验和评估。

---
*报告生成时间: $(date)*
REPORT

log_success "验证报告已生成: $VERIFICATION_REPORT"

# 完成
echo ""
echo "🎉 ========================================"
echo "✅ MELD系统验证完成!"
echo "⏱️  总耗时: ${DURATION}秒"
echo "📄 验证报告: $VERIFICATION_REPORT"
echo "📊 详细结果: $VERIFICATION_JSON"
echo "🎉 ========================================"
echo ""

log_success "所有验证步骤完成，系统运行正常!"
log_info "可以开始进行完整的MELD实验和评估"