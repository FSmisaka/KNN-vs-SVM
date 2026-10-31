# 环境

```bash
conda create --name ml python=3.11
conda activate ml
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
```

# 实验

## exp1 线性可分

```bash
python experiments/exp1_linear.py
```

## exp2 非线性（圆内 / 圆外）

```bash
python experiments/exp2_circle.py
```

## exp3 高维（仅 1 维有效）

```bash
python experiments/exp3_highdim.py
```

## exp4 标签噪声

```bash
python experiments/exp4_label_noise.py
```

## 实验结果

结果统一存放在 `results/` 下，按文件类型分两个子文件夹：

```text
results/
├── csv/        # 每个实验的评价指标（grid search 选出超参数后的测试集指标）
│   ├── exp1_linear.csv
│   ├── exp2_circle.csv
│   ├── exp3_highdim.csv
│   └── exp4_label_noise.csv
└── figures/    # 每个实验两张图（KNN / SVM 各一张）：测试集真实标签 vs 模型预测（PNG）
    ├── exp1_linear_knn.png / exp1_linear_svm.png
    ├── exp2_circle_knn.png / exp2_circle_svm.png
    ├── exp3_highdim_knn.png / exp3_highdim_svm.png
    └── exp4_label_noise_knn.png / exp4_label_noise_svm.png
```

图片说明：每张图左子图为测试集按真实标签染色（+1 蓝 / −1 红）并叠加真实（Bayes）边界虚线；右子图为同一批测试点按模型预测染色，背景为二维网格预测着色的决策区域，加粗线为模型决策边界（SVM 即超平面 w·x + b = 0），误分类点以 × 标注；图例统一置于图下方。高维实验（exp3）展示「维度 1 × 维度 2」切片，其余维度固定为测试集均值。

例外（exp4）：噪声只施加于训练标签，测试标签本身是干净的。为直观展示噪声，左子图染色改用对测试标签施加同比例（η = 30%）翻转后的「观测标签」，因此约有 30% 的点会跨越真实边界；误分类 × 与 CSV 指标仍相对干净的真实标签计算。