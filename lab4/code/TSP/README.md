# 遗传算法解决TSP问题

这个程序使用遗传算法求解旅行商问题（TSP）。

## 使用说明

1. 从 [National Traveling Salesman Problems (Waterloo)](https://www.math.uwaterloo.ca/tsp/world/countries.html) 下载TSP问题数据集
2. 将下载的 `.tsp`文件放在 `tsp_data`目录下（如果目录不存在会自动创建）
3. 运行程序：`python tsp.py`
4. 按照提示选择要使用的数据集
5. 程序会自动以不同参数运行多次实验，并输出结果

## 结果说明

程序会生成以下输出：

1. 终端显示每次运行的详细信息和统计数据
2. 为每个参数组合生成图表，显示收敛曲线和最优路径
3. 将所有实验结果保存到 `dataset_name_results.csv`文件中

## 参数说明

可以调整以下参数来优化遗传算法性能：

- `pop_size`: 种群大小
- `num_generations`: 迭代代数
- `mutation_rate`: 变异概率
- `elitism_ratio`: 精英比例

## 代码结构

- `TSPSolver` 类: 包含遗传算法的核心功能
- `read_tsp_file()`: 从TSP数据文件读取城市坐标
- `run_experiment()`: 使用不同参数多次运行实验收集统计数据
