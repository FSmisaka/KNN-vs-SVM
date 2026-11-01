# 环境

```bash
conda create --name ml python=3.11
conda activate ml
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib scikit-learn  # scikit-learn 仅用于实现正确性对照
```

# 数据

把 HEPMASS_500k.csv 和 webspam_30k.svm 放置到 /data 文件夹下

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

## exp5 KNN 算法细节展示 - 邻居选择

```bash
python experiments/exp5_knn_detail.py
```

## exp6 SVM 算法细节展示 - 间隔

```bash
python experiments/exp6_svm_detail.py
```

## exp7 真实数据集：超参数搜索与最终测试（任务 c）

```bash
python experiments/exp7_real.py            # 两个数据集都跑
python experiments/exp7_real.py webspam    # 只跑单个数据集
python experiments/exp7_replot.py          # 仅按最新 CSV 重绘 exp7 的图（不重跑搜索）
```

- **样本量**：webspam（30k）直接使用**全量**；HEPMASS（500k）按类别分层、无放回抽取 **100,000** 条（seed=42，全量的 1/5，类别比例与全量一致）。
- **切分**：train/val/test = 6:2:2（webspam 18000/6000/6000，HEPMASS 60000/20000/20000；`train_val_test_split`，seed=42，与合成实验同一切分逻辑）。
- **预处理**：z-score 标准化的均值/标准差**仅在训练集上估计**；训练集零方差列丢弃（webspam 129→127 维；HEPMASS 27 维全保留）。同一变换作用于 val/test。
- **超参数搜索**：grid search 每个超参的训练/验证 accuracy 与 F1；超参仅由验证集 accuracy 选择（并列则取 F1 更高者）。选定参数后在训练集重训、测试集做一次最终评估。
- **网格**：KNN k ∈ {1,3,5,7,9,11,15,21,31,51} × weights ∈ {uniform, distance}；SVM C ∈ 10⁻²…10⁶。

最终测试结果（`csv/exp7_real_test.csv`）：

| dataset | model | 选出超参 | test acc | test F1 |
|---|---|---|---|---|
| webspam（全量 30k） | KNN | k=3, distance | **0.9645** | **0.9711** |
| webspam（全量 30k） | SVM | C=10⁶ | 0.9307 | 0.9429 |
| hepmass（100k 子集） | KNN | k=51, distance | 0.9036 | 0.9052 |
| hepmass（100k 子集） | SVM | C=100 | **0.9071** | **0.9084** |

## 实现正确性对照（sklearn）

```bash
python experiments/verify_sklearn.py
```

对合成实验 1-4 的全部设置与实验 7 的最终超参数，用 sklearn 的 KNeighborsClassifier 与线性核 SVC 在相同数据、切分与标准化上重跑，逐行比较测试指标，结果写入 `results/csv/verify_sklearn.csv`（含双方训练目标值）。sklearn 只用于本对照，不参与任何实验结论的计算。

## 实验结果

结果统一存放在 `results/` 下，按文件类型分两个子文件夹：

```text
results/
├── csv/        # 每个实验的评价指标（grid search 选出超参数后的测试集指标）
│   ├── exp1_linear.csv
│   ├── exp2_circle.csv
│   ├── exp3_highdim.csv
│   ├── exp4_label_noise.csv
│   ├── exp5_knn_detail.csv   # k 扫描指标（train/val/test accuracy）
│   ├── exp6_svm_detail.csv   # C 扫描指标（间隔宽、支持向量数、accuracy）
│   ├── exp7_real_search.csv  # 真实数据网格搜索：每个网格点的 train/val accuracy 与 F1 + 耗时
│   └── exp7_real_test.csv    # 真实数据最终测试指标（超参由验证集选出）
│   └── verify_sklearn.csv    # 与 sklearn 的对照结果（每行含双方指标、差异与训练目标值）
└── figures/    # 每个实验一张合并图（按印刷尺寸生成，300 dpi）：[真实标签 | KNN 预测 | SVM 预测]
    ├── exp1_linear.png / exp2_circle.png / exp3_highdim.png / exp4_label_noise.png
    ├── exp5_knn_neighbors.png / exp5_knn_ksweep.png
    ├── exp6_svm_margin.png / exp6_svm_c_sweep.png
    └── exp7_{webspam,hepmass}.png
```
