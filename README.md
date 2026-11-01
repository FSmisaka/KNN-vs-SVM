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

## exp4 结构性标签噪声

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

图片说明：每张图左子图为测试集按真实标签染色（+1 蓝 / −1 红）并叠加真实（Bayes）边界；右子图为同一批测试点按模型预测染色，误分类点以 × 标注；图例统一置于图下方。

- exp1 / exp2（二维）：右子图背景为二维网格预测着色的决策区域，加粗线为决策边界（SVM 即超平面 w·x + b = 0）。
- exp3（3D）：三轴为维度 1–3，展示框三条轴的相对长度为 1/2 : 1 : 1（维度 1 取值范围远小于噪声维度 N(0,10)，适度压缩其轴长以兼顾可读性；比例尺不等于数据真实比例）；SVM 超平面切片以半透明平面绘制（未展示维度固定为测试集均值），真实边界 x₁ = 0 为虚线线框平面；KNN 无全局边界，仅展示染色散点。
- exp4（二维，结构性标签噪声）：仅训练集被注入 η 比例的错标点——+1 类错标点沿真实边界的投影 t ~ N(+3, 0.5²)、−1 类 t ~ N(−3, 0.5²)（同标准差、均值显著不同），深度 s ~ U(1.0, 1.6)，带相反的观测标签；验证/测试标签干净。右子图以空心三角展示这些错标点（按观测标签着色）。预期观察：系统性错标簇对线性 SVM 的超平面产生一致「扭矩」，使其偏转、精度随 η 下降；KNN 仅在簇附近局部受影响；同时用 acc–C 曲线检验「C 越大扭转越强」的直觉。
