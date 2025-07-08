"""由于本人只看到了PPT上面的要求，没有注意超算习堂的代码格式要求，并且剩余时间不足以重新编写，本人添加了函数注释，希望助教谅解，下次一定注意格式要求。"""
import numpy as np  # 用于数值计算和矩阵操作
import random  # 生成随机数和随机选择
import matplotlib.pyplot as plt  # 绘制图表和可视化
from math import sqrt  # 计算平方根用于距离计算
from time import time  # 计算算法运行时间
import os  # 操作系统相关功能，如文件路径处理
import pandas as pd  # 数据处理和分析，用于保存结果
import re  # 正则表达式，用于解析TSP文件

class TSPSolver:
    def __init__(self, cities):
        """初始化TSP求解器，设置城市坐标和距离矩阵"""
        self.cities = cities
        self.dist_matrix = self._create_distance_matrix()
        self.num_cities = len(cities)
        
    def _create_distance_matrix(self):
        """计算城市之间的距离矩阵"""
        n = len(self.cities)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dx = self.cities[i][0] - self.cities[j][0]
                dy = self.cities[i][1] - self.cities[j][1]
                dist = sqrt(dx*dx + dy*dy)
                dist_matrix[i][j] = dist
                dist_matrix[j][i] = dist
        return dist_matrix
    
    def _initialize_population(self, pop_size):
        """创建初始随机种群"""
        return [random.sample(range(self.num_cities), self.num_cities) 
                for _ in range(pop_size)]
    
    def _calculate_fitness(self, individual):
        """计算个体适应度（距离的倒数）"""
        total = 0
        for i in range(len(individual)):
            total += self.dist_matrix[individual[i]][individual[(i+1)%len(individual)]]
        return 1/total
    
    def _roulette_wheel_selection(self, population, fitness_values, num_parents):
        """基于轮盘赌法选择父代个体"""
        total_fitness = sum(fitness_values)
        probs = [f/total_fitness for f in fitness_values]
        return random.choices(population, weights=probs, k=num_parents)
    
    def _order_crossover(self, parent1, parent2):
        """使用顺序交叉法（OX）生成子代"""
        size = self.num_cities
        start, end = sorted(random.sample(range(size), 2))
        
        child = [None]*size
        child[start:end] = parent1[start:end]
        
        ptr = end
        for city in parent2:
            if city not in child:
                if ptr >= size:
                    ptr = 0
                while ptr < size and child[ptr] is not None:
                    ptr += 1
                if ptr >= size:
                    ptr = 0
                child[ptr] = city
                ptr += 1
        
        return child
    
    def _swap_mutation(self, individual, mutation_rate):
        """随机交换两个城市位置进行变异"""
        if random.random() < mutation_rate:
            i, j = random.sample(range(self.num_cities), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual
    
    def solve(self, params):
        """使用遗传算法求解TSP问题"""
        pop_size = params.get('pop_size', 100)  # 种群大小
        num_generations = params.get('num_generations', 500)  # 迭代次数
        mutation_rate = params.get('mutation_rate', 0.01)  # 变异率
        elitism_ratio = params.get('elitism_ratio', 0.1)  # 精英保留比例
        
        population = self._initialize_population(pop_size)  # 初始化种群
        num_elites = int(pop_size * elitism_ratio)  # 计算精英数量
        
        history = {'best': [], 'avg': [], 'time': time()}  # 初始化历史记录
        
        for generation in range(num_generations):  # 主循环
            fitness_values = [self._calculate_fitness(ind) for ind in population] # 计算适应度
            # 记录最佳和平均距离(转换为实际距离)
            best_fitness = max(fitness_values)
            avg_fitness = sum(fitness_values) / pop_size
            history['best'].append(1/best_fitness)
            history['avg'].append(1/avg_fitness)
            # 选择精英个体
            elite_indices = np.argsort(fitness_values)[-num_elites:]
            elites = [population[i] for i in elite_indices]
            # 选择父代(轮盘赌)
            parents = self._roulette_wheel_selection(
                population, fitness_values, pop_size - num_elites)
            # 生成子代
            children = []
            for i in range(0, len(parents), 2):  # 每次处理两个父代
                if i+1 < len(parents):  # 确保有足够的父代
                    # 顺序交叉生成两个子代
                    child1 = self._order_crossover(parents[i], parents[i+1])
                    child2 = self._order_crossover(parents[i+1], parents[i])
                    # 对子代进行变异并加入子代列表
                    children.append(self._swap_mutation(child1, mutation_rate))
                    children.append(self._swap_mutation(child2, mutation_rate))
            population = elites + children
        # 计算总运行时间
        history['time'] = time() - history['time']
        # 找到最佳个体
        best_idx = np.argmax([self._calculate_fitness(ind) for ind in population])
        return population[best_idx], history  # 返回最佳路径和历史记录
    
    def visualize(self, route, history, title=None, result_dir=None):
        """可视化TSP路径和收敛历史"""
        plt.figure(figsize=(15, 5))  # 创建大图
        
        # 第一个子图: 收敛记录
        plt.subplot(1, 2, 1)
        plt.plot(history['best'], label='Best Distance')  # 最佳距离曲线
        plt.plot(history['avg'], label='Average Distance')  # 平均距离曲线
        plt.xlabel('Generation')  # x轴标签
        plt.ylabel('Distance')  # y轴标签
        plt.title('Convergence History')  # 标题
        plt.legend()  # 显示图例
        
        # 第二个子图: TSP路径
        plt.subplot(1, 2, 2)
        # 获取坐标
        x = [self.cities[i][0] for i in route] + [self.cities[route[0]][0]]
        y = [self.cities[i][1] for i in route] + [self.cities[route[0]][1]]
        plt.plot(x, y, 'o-')  # 绘制路径
        plt.title(f'TSP Route (Distance: {history["best"][-1]:.2f})')
        plt.tight_layout()  # 调整布局
        if title and result_dir:  
            plt.savefig(os.path.join(result_dir, f"{title}.png"))  # 保存图片
        plt.show()  # 显示图片
    
    def calculate_route_distance(self, route):
        """计算给定路线的总距离"""
        total_distance = 0
        for i in range(len(route)):
            total_distance += self.dist_matrix[route[i]][route[(i+1)%len(route)]]
        return total_distance
    
    def print_results(self, route, history):
        """打印TSP求解结果信息"""
        distance = self.calculate_route_distance(route)
        print("\n==== TSP 求解结果 ====")
        print(f"城市数量: {self.num_cities}")
        print(f"最优路线: {route}")
        print(f"路线距离: {distance:.2f}")
        print(f"求解时间: {history['time']:.2f} 秒")
        print(f"迭代次数: {len(history['best'])}")
        print("====================\n")
        return distance

def read_tsp_file(file_path):
    """从TSP数据文件中读取城市坐标"""
    cities = []
    reading_coords = False
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                reading_coords = True
                continue
            elif line == "EOF":
                break
            
            if reading_coords:
                parts = re.split(r'\s+', line)
                if len(parts) >= 3:
                    # 忽略城市ID，只获取x和y坐标
                    cities.append((float(parts[1]), float(parts[2])))
    
    return cities

def run_experiment(cities_data, params_list, dataset_name, result_dir=None, num_runs=5):
    """使用不同的参数运行多次实验并收集统计结果"""
    results = []
    
    for params in params_list:
        param_results = {
            'pop_size': params['pop_size'],
            'num_generations': params['num_generations'],
            'mutation_rate': params['mutation_rate'],
            'elitism_ratio': params['elitism_ratio'],
            'distances': [],
            'times': [],
        }
        
        print(f"\n开始参数组合: pop_size={params['pop_size']}, generations={params['num_generations']}, "
              f"mutation_rate={params['mutation_rate']}, elitism_ratio={params['elitism_ratio']}")
        
        for run in range(num_runs):
            print(f"运行 {run+1}/{num_runs}...")
            random.seed(run)  # 设置不同的随机种子
            np.random.seed(run)
            
            solver = TSPSolver(cities_data)
            best_route, history = solver.solve(params)
            
            distance = solver.calculate_route_distance(best_route)
            param_results['distances'].append(distance)
            param_results['times'].append(history['time'])
            
            # 输出当前实验运行的结果
            print(f"运行 {run+1} 完成 - 距离: {distance:.2f}, 时间: {history['time']:.2f}秒")
            
            # 只为第一次运行可视化并保存图像
            if run == 0:
                title = f"{dataset_name}_pop{params['pop_size']}_gen{params['num_generations']}_mr{params['mutation_rate']}"
                solver.print_results(best_route, history)
                solver.visualize(best_route, history, title, result_dir)
        
        # 计算多次运行的统计数据
        param_results['min_distance'] = min(param_results['distances'])  # 最小距离
        param_results['max_distance'] = max(param_results['distances'])  # 最大距离
        param_results['avg_distance'] = sum(param_results['distances']) / len(param_results['distances'])  # 平均距离
        param_results['std_distance'] = np.std(param_results['distances']) if len(param_results['distances']) > 1 else 0  # 标准差
        param_results['avg_time'] = sum(param_results['times']) / len(param_results['times'])  # 平均执行时间
        # 将果添加到总结果列表中
        results.append(param_results)
        # 打印
        print(f"结果:")
        print(f"  最小距离: {param_results['min_distance']:.2f}")
        print(f"  最大距离: {param_results['max_distance']:.2f}")
        print(f"  平均距离: {param_results['avg_distance']:.2f} ± {param_results['std_distance']:.2f}")
        print(f"  平均时间: {param_results['avg_time']:.2f}秒")
    
    # 将所有参数组合的结果准备保存到CSV文件
    df_data = []
    for r in results:
        df_data.append({
            'pop_size': r['pop_size'],
            'num_generations': r['num_generations'],
            'mutation_rate': r['mutation_rate'],
            'elitism_ratio': r['elitism_ratio'],
            'min_distance': r['min_distance'],
            'max_distance': r['max_distance'],
            'avg_distance': r['avg_distance'],
            'std_distance': r['std_distance'],
            'avg_time': r['avg_time']
        })
    
    df = pd.DataFrame(df_data)
    results_file = os.path.join(result_dir, f"{dataset_name}_results.csv")
    df.to_csv(results_file, index=False)
    return results

# 使用示例
if __name__ == "__main__":
    """主程序入口，处理数据集并运行TSP实验"""
    # 获取脚本当前目录，确保能找到tsp_data文件夹
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 设置存放TSP数据文件的目录，与tsp.py并列
    data_dir = os.path.join(script_dir, "tsp_data")
    print(f"查找数据文件夹: {data_dir}")
    
    if not os.path.exists(data_dir):
        # 如果目录不存在，创建它并提示用户
        os.makedirs(data_dir)
        print(f"请将TSP数据文件放入 {data_dir} 目录")
    
    # 打印提示信息，准备让用户选择数据集
    print("可用的数据集:")
    # 获取目录中所有以.tsp结尾的文件作为可用数据集
    available_datasets = [f for f in os.listdir(data_dir) if f.endswith('.tsp')]
    
    if not available_datasets:
        # 如果没有找到任何数据集文件，提示用户并使用内置的示例数据
        print("没有找到数据集，请将.tsp文件放入tsp_data目录")
        # 使用硬编码的城市坐标作为默认示例数据
        # 每个元组代表一个城市的(x, y)坐标
        cities = [(1150, 1760), (630, 1660), (40, 2090), (750, 1100), 
                  (750, 2030), (1030, 2070), (1650, 650), (1490, 1630),
                  (790, 2260), (710, 1310), (840, 550), (1170, 2300),
                  (970, 1340), (510, 700), (750, 900), (1280, 1200),
                  (230, 590), (460, 860), (1040, 950), (590, 1390),
                  (830, 1770), (490, 500), (1840, 1240), (1260, 1500),
                  (1280, 790), (490, 2130), (1460, 1420), (1260, 1910),
                  (360, 1980)]
        
        # 创建结果目录
        result_dir = os.path.join(script_dir, "result")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            print(f"创建结果目录: {result_dir}")
        
        # 创建TSP求解器实例
        solver = TSPSolver(cities)
        # 使用预定义参数求解TSP问题
        best_route, history = solver.solve({
            'pop_size': 100,         # 种群大小为100
            'num_generations': 500,  # 进化500代
            'mutation_rate': 0.02,   # 变异率为2%
            'elitism_ratio': 0.1     # 精英保留比例为10%
        })
        solver.print_results(best_route, history)
        solver.visualize(best_route, history, result_dir=result_dir)
    else:
        # 检查每个文件是否可读
        for dataset in available_datasets:
            full_path = os.path.join(data_dir, dataset)
            if os.access(full_path, os.R_OK):
                print(f"文件 {dataset} 可读")
            else:
                print(f"警告: 文件 {dataset} 无法读取")

        # 定义不同的参数组合
        params_list = [
            {'pop_size': 50, 'num_generations': 300, 'mutation_rate': 0.01, 'elitism_ratio': 0.1},
            {'pop_size': 100, 'num_generations': 500, 'mutation_rate': 0.02, 'elitism_ratio': 0.1},
            {'pop_size': 150, 'num_generations': 800, 'mutation_rate': 0.05, 'elitism_ratio': 0.2},
        ]
        
        # 处理每个数据集
        for dataset in available_datasets:
            print(f"=============================================")
            print(f"处理数据集: {dataset}")
            print(f"=============================================")
            
            dataset_path = os.path.join(data_dir, dataset)
            dataset_name = os.path.splitext(dataset)[0]
            
            try:
                # 尝试读取文件并添加错误处理
                print(f"尝试读取文件: {dataset_path}")
                cities = read_tsp_file(dataset_path)
                if not cities:
                    print(f"警告: 未能从文件 {dataset} 读取任何城市数据")
                    continue
                print(f"成功读取了 {len(cities)} 个城市坐标")
                
                # 创建结果目录
                result_dir = os.path.join(script_dir, "result")
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                    print(f"创建结果目录: {result_dir}")
                
                # 使用修复后的函数
                run_experiment(cities, params_list, dataset_name, result_dir=result_dir, num_runs=3)
            except Exception as e:
                print(f"处理数据集 {dataset} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()