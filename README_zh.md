# 论文五幅图及绘图源文件

在 figures 文件夹中可以直接打开每幅图的 PDF 或 PNG。figures_and_tables.pdf 将五幅图和两张表依次排在单独页面，并附论文题注。

## 重新绘图

保留解压后的目录结构，在包含 code、figures、results 的目录中运行以下命令。本次复现使用 Python 3.12.2。

    python -m pip install -r code/requirements.txt
    python code/reproduce.py

第一次安装绘图库时使用第一条；已有对应版本时直接使用第二条。运行后，模型和绘图程序会更新 figures 中的图像以及 results 中的数据。修改输出前可另存原图。

| 论文图号 | 内容 | code/reproduce.py 中的函数 | 数据 |
|---|---|---|---|
| 1 | 一个季度内的银行决策、支出、定价和状态更新 | make_mechanism | 机制图由程序直接绘制 |
| 2 | 信贷重配在两种供给响应下的季度路径 | make_reallocation | reallocation_weak.csv 与 reallocation_strong.csv |
| 3 | 卖方再支出比例与供给响应决定的稳态反转阈值 | make_threshold | threshold_grid.csv |
| 4 | 资本约束不同的银行在降息后的反应 | make_bank | rate_cut_finite.csv、rate_cut_ample.csv、rate_baseline.csv |
| 5 | 两种期初贷款组合如何改变降息的价格效应 | make_legacy | 8 个 inheritance_*.csv；每组均有降息和基准情景 |

code/model.py 提供银行贷款选择、市场价格、状态递推和稳态计算；code/parameters.json 保存参数。所有数值均由模型生成。单幅图不需要 LaTeX。若要编辑题注，可修改 artwork 中对应的 .tex，再用 XeLaTeX 编译 artwork.tex 两次。
