# MELD Experiment Reproduction Guide

## Step 1: Environment Setup

### 1. Install uv (if not already installed)

```bash
# Install using curl (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or install using pip
pip install uv

# Or install using homebrew (macOS)
brew install uv
```

### 2. Sync dependencies

```bash
# Sync dependencies
uv sync
```

### 3. Verify environment

```bash
# Verify core dependencies
uv run python -c "
import torch, transformers, sklearn, pandas, numpy
print('✅ Environment setup successful')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"
```

## Step 2: Model Preparation

### 1. Download Qwen model

```bash
# Download Qwen3-0.6B model
uv run python scripts/download_models.py
```

### 2. Verify model path

```bash
# Check if model downloaded successfully
ls -la models/
```

## Step 3: Data Preparation

### 1. Prepare input data

Place CAPE JSON data in specified directories:
```bash
# Malicious samples (grouped by family)
mkdir -p input/cape_behavior_malicious_train

# Benign samples (flat structure)
mkdir -p input/cape_behavior_benign_train

# Download dataset from dataset repository
# Required files: malicious_dataset_manifest.csv
# Required files: cape_reports.7z.* -> cape_reports
uv run python scripts/copy_correct_cape_samples.py --manifest malicious_dataset_manifest.csv  --source-malicious cape_reports --source-benign cape_reports --target-malicious input/cape_behavior_malicious_train --target-benign input/cape_behavior_benign_train
```

### 2. Data conversion

```bash
# Convert malicious samples to Markdown
uv run python scripts/batch_convert_cape_data.py \
  --input input/cape_behavior_malicious_train \
  --output data/processed/cape_reports_malicious_md \
  --workers 4

# Convert benign samples to Markdown
uv run python scripts/batch_convert_cape_data.py \
  --input input/cape_behavior_benign_train \
  --output data/processed/cape_reports_benign_md \
  --workers 4
```

### 3. Data preprocessing

```bash
# Generate dataset index
uv run python scripts/preprocess_data.py \
  --input data/processed \
  --output data/processed

# Verify data preprocessing results
ls -la data/processed/
cat data/processed/dataset_stats.json
```

## Step 4: System Verification

**Purpose**: Verify that MELD core functionality works properly before running formal experiments

### Use verification script (recommended)

```bash
# Run MELD system verification script
./scripts/verify_meld_system.sh
```

This script will automatically complete the following verification steps:
1. **Environment verification** - Check Python version and all dependency packages
2. **Model path check** - Verify Qwen model availability
3. **Data integrity check** - Check input data and processed data
4. **MELD functionality verification** - Test feature extraction and classification functions
5. **Generate verification report** - Create detailed verification report

### View verification results

```bash
# View verification report
cat results/meld_system_verification_report.md

# View detailed JSON results
cat results/meld_verification.json

# Check all result files
ls -la results/
```


## Step 5: Experiment Execution

### 1. Time-OOD Experiment

```bash
# Run temporal out-of-distribution detection experiment
uv run python experiments/time_ood/run_time_ood.py \
  --data_dir data/processed \
  --model_dir models/qwen3-0.6b \
  --output results/time_ood_results.json
```

### 2. Family-OOD Experiment

#### Single family experiment

```bash
# Run family out-of-distribution detection experiment (using AgentTesla as example)
uv run python experiments/family_ood/run_family_ood.py \
  --target_family AgentTesla \
  --data_dir data/processed \
  --model_dir models/qwen3-0.6b

# Quick test: use small sample size (recommended for initial verification)
uv run python experiments/family_ood/run_family_ood.py \
  --target_family AgentTesla \
  --max_samples 100

# Fine-grained sample control
uv run python experiments/family_ood/run_family_ood.py \
  --target_family AsyncRAT \
  --max_train_samples 500 \
  --max_val_samples 50 \
  --max_test_samples 200

# Skip report generation
uv run python experiments/family_ood/run_family_ood.py \
  --target_family Formbook \
  --max_samples 100 \
  --no-report
```

#### Batch experiments (recommended)

```bash
# Run batch experiments for all Top-8 families (auto-generate summary report)
uv run python experiments/family_ood/run_family_ood.py \
  --batch \
  --data_dir data/processed \
  --model_dir models/qwen3-0.6b

# Batch quick test (with sample limits, recommended for resource-limited environments)
uv run python experiments/family_ood/run_family_ood.py \
  --batch \
  --max_samples 200

# Batch experiments skip report generation
uv run python experiments/family_ood/run_family_ood.py \
  --batch \
  --max_samples 100 \
  --no-report

# Custom batch summary report path
uv run python experiments/family_ood/run_family_ood.py \
  --batch \
  --max_samples 300 \
  --batch_report results/custom_family_summary.md
```

**Parameter descriptions:**

**Sample limit parameters:**
- `--max_samples N`: Set all sample limits to N (convenience parameter)
- `--max_train_samples N`: Maximum training samples
- `--max_val_samples N`: Maximum validation samples  
- `--max_test_samples N`: Maximum test samples
- Use all available samples when no limit parameters are specified

**Report generation parameters:**
- `--report FILE`: Specify single experiment report file path (auto-generated if not specified)
- `--no-report`: Skip report generation
- `--batch_report FILE`: Specify batch experiment summary report path

**Batch experiment parameters:**
- `--batch`: Run batch experiments for all 8 families
- Batch mode automatically generates individual reports for each family + one summary comparison report

**Experiment outputs:**
- **JSON results**: `results/family_ood/meld_family_ood_{family}.json`
- **Individual reports**: `results/family_ood/family_ood_{family}_report.md`
- **Summary report**: `results/family_ood/family_ood_summary_report.md`


## Step 6: Results Analysis

### 1. Time-OOD experiment results analysis

```bash
# View Time-OOD JSON results
cat results/time_ood_results.json

# View Time-OOD Markdown report (auto-generated)
cat results/time_ood/time_ood_report.md

# View detailed JSON result files
ls results/time_ood/
```

**Time-OOD results interpretation:**
- **Macro F1-Score**: Overall classification performance
- **AUROC**: Area under ROC curve, measures classifier discrimination ability
- **AUPR**: Area under PR curve, suitable for imbalanced data
- **Optimal threshold**: Decision threshold for binary classification

### 2. Family-OOD experiment results analysis

```bash
# View all family experiment results
ls results/family_ood/

# View summary report (recommended)
cat results/family_ood/family_ood_summary_report.md

# View individual family detailed reports
cat results/family_ood/family_ood_agenttesla_report.md
cat results/family_ood/family_ood_asyncrat_report.md
# ... other families

# View raw JSON results
cat results/family_ood/meld_family_ood_agenttesla.json
```

**Family-OOD results interpretation:**
- **Each family as OOD**: Test model's generalization ability to unseen families
- **Performance comparison**: Detection difficulty differences across families
- **Summary statistics**: Average performance across all families


## Step 7: Advanced Experiments

### 1. Layer analysis experiments

```bash
# Test feature effects of different layers
python scripts/test_mistral_all_layers.py

# Layer testing without randomization
python scripts/test_mistral_all_layers_norandom.py
```

### 2. Batch layer evaluation

```bash
# Run Layer 15 experiment
./scripts/run_rq2_layer15.sh

# Run Layer 28 experiment
./scripts/run_rq2_layer28.sh
```

### 3. OOD evaluation experiments

```bash
# Install
uv pip install -e .

# Run OOD evaluation
uv run python src/meld/ood_eval.py \
  --index_csv data/processed/dataset_with_family_time.csv \
  --model_dir models/qwen3-0.6b \
  --mode family \
  --test_family AgentTesla \
  --out results/ood_evaluation.json
```

