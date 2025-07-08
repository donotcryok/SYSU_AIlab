import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures

# 设置matplotlib绘图中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False   

# 读取数据函数
def load_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            data.append([float(x) for x in row])  #浮点数
    return np.array(data)  # 转换为NumPy数组并返回

# 异常值检测与处理
def detect_and_remove_outliers(X, y, method='iqr', threshold=1.5):
    # 将特征和目标合并为一个数组进行异常值检测，确保同时移除X和y中对应的异常点
    data = np.hstack((X, y.reshape(-1, 1)))
    n_samples, n_features = data.shape
    outliers_mask = np.zeros(n_samples, dtype=bool)  # 初始化异常值掩码为全False
    if method == 'iqr':
        # 使用IQR方法(箱线图法)检测异常值:异常值定义为小于Q1-threshold*IQR或大于Q3+threshold*IQR的值
        for i in range(n_features):
            q1 = np.percentile(data[:, i], 25)  
            q3 = np.percentile(data[:, i], 75)  
            iqr = q3 - q1 
            lower_bound = q1 - threshold * iqr 
            upper_bound = q3 + threshold * iqr 
            column_outliers = (data[:, i] < lower_bound) | (data[:, i] > upper_bound)  
            outliers_mask = outliers_mask | column_outliers  
    elif method == 'zscore':
        # 使用Z-score方法检测异常值: Z-score = (x - μ)/σ，异常值定义为|Z-score| > threshold的值
        for i in range(n_features):
            z_scores = np.abs(stats.zscore(data[:, i]))  
            column_outliers = z_scores > threshold 
            outliers_mask = outliers_mask | column_outliers 
    # 返回清洗后的数据和异常值掩码
    return X[~outliers_mask], y[~outliers_mask], outliers_mask

# 数据标准化
def normalize_data(X):
    mean = np.mean(X, axis=0)  # 计算每列(每个特征)的均值
    std = np.std(X, axis=0)    # 计算每列的标准差
    return (X - mean) / std, mean, std  # 返回标准化后的数据及均值、标准差(用于后续逆变换)

def transform_geo_features(X, y=None, cluster_centers=None, n_clusters=5):
    # 提取地理坐标(经度、纬度)
    geo_coords = X[:, :2].copy()
    
    # 1. 使用K-means进行区域聚类
    if cluster_centers is None:
        # 训练阶段: 拟合模型
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(geo_coords)
        cluster_centers = kmeans.cluster_centers_
    else:
        # 预测阶段: 使用已有中心点
        kmeans = KMeans(n_clusters=len(cluster_centers), random_state=42)
        kmeans.cluster_centers_ = cluster_centers
    
    # 2. 计算到各聚类中心的距离作为特征
    dist_to_clusters = np.zeros((len(X), n_clusters))
    for i in range(n_clusters):
        center = cluster_centers[i]
        # 计算欧几里得距离
        dist_to_clusters[:, i] = np.sqrt(np.sum((geo_coords - center) ** 2, axis=1))
    
    # 提取房龄和收入特征
    other_features = X[:, 2:].copy()
    
    # 将距离特征与房龄、收入特征合并
    X_transformed = np.hstack((other_features, dist_to_clusters))
    
    return X_transformed, cluster_centers

# 多层感知机模型 - 实现混合激活函数策略
class MLP:
    def __init__(self, layer_sizes, learning_rate=0.001, epochs=1000):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_layers = len(layer_sizes)
        self.weights = []  # 存储每层的权重
        self.biases = []   # 存储每层的偏置
        self.loss_history = []  # 记录训练过程中的损失
        
        # 初始化权重和偏置
        for i in range(1, self.num_layers):
            # 使用 He 初始化方法初始化权重，适合 ReLU 激活函数
            w = np.random.randn(self.layer_sizes[i-1], self.layer_sizes[i]) * np.sqrt(2.0 / self.layer_sizes[i-1])
            b = np.zeros((1, self.layer_sizes[i]))
            self.weights.append(w)
            self.biases.append(b)
    
    def tanh(self, x):
        """双曲正切激活函数"""
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        """双曲正切激活函数的导数"""
        return 1.0 - np.tanh(x)**2
    
    def relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """ReLU激活函数的导数"""
        return np.where(x > 0, 1.0, 0.0)
    
    def leaky_relu(self, x, alpha=0.01):
        """Leaky ReLU激活函数"""
        return np.maximum(alpha * x, x)
    
    def leaky_relu_derivative(self, x, alpha=0.01):
        """Leaky ReLU激活函数的导数"""
        return np.where(x > 0, 1.0, alpha)
    
    def linear(self, x):
        """线性激活函数，直接返回输入"""
        return x
    
    def linear_derivative(self, x):
        """线性激活函数的导数"""
        return np.ones_like(x)
    
    def forward(self, X):
        activations = [X]  # 第一个元素是网络的输入
        layer_inputs = []  # 存储每层的输入(激活函数前的值)
        
        # 前向传播
        for i in range(self.num_layers - 1):
            # 计算当前层的输入
            layer_input = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            layer_inputs.append(layer_input)
            
            # 应用激活函数
            if i == 0:  # 第一隐藏层使用tanh激活函数
                activation = self.tanh(layer_input)
            elif i == self.num_layers - 2:  # 输出层使用线性激活函数
                activation = self.linear(layer_input)
            elif i == 1:  # 第二隐藏层使用ReLU激活函数
                activation = self.relu(layer_input)
            elif i == 2 and self.num_layers > 4:  # 第三隐藏层使用Leaky ReLU
                activation = self.leaky_relu(layer_input)
            elif i == 3 and self.num_layers > 5:  # 第四隐藏层(新增)使用tanh激活函数
                activation = self.tanh(layer_input)
            else:  # 其他隐藏层使用ReLU激活函数
                activation = self.relu(layer_input)
            activations.append(activation)
            
        return activations, layer_inputs
    
    def backward(self, X, y, activations, layer_inputs):
        n_samples = X.shape[0]
        
        # 计算输出层误差 (y_pred - y_true)
        output_error = activations[-1] - y.reshape(-1, 1)
        
        # 初始化当前误差为输出层误差
        delta = output_error
        
        # 从输出层向前反向传播
        for i in range(self.num_layers - 2, -1, -1):
            # 计算当前层权重的梯度
            dw = np.dot(activations[i].T, delta) / n_samples
            db = np.sum(delta, axis=0, keepdims=True) / n_samples
            
            # 更新当前层的权重和偏置
            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db
            
            # 如果不是第一层，则计算前一层的误差
            if i > 0:
                # 确定当前层使用的激活函数的导数
                if i == 1:  # 第一隐藏层使用tanh
                    derivative = self.tanh_derivative(layer_inputs[i-1])
                elif i == 2 and self.num_layers > 4:  # 第三隐藏层使用Leaky ReLU
                    derivative = self.leaky_relu_derivative(layer_inputs[i-1])
                elif i == 3 and self.num_layers > 5:  # 第四隐藏层(新增)使用tanh
                    derivative = self.tanh_derivative(layer_inputs[i-1])
                else:  # 其他隐藏层使用ReLU
                    derivative = self.relu_derivative(layer_inputs[i-1])
                
                # 计算前一层的误差
                delta = np.dot(delta, self.weights[i].T) * derivative

    def fit(self, X, y):
        for epoch in range(self.epochs):
            # 前向传播
            activations, layer_inputs = self.forward(X)
            
            # 计算损失(均方误差)
            predictions = activations[-1]
            loss = np.mean((predictions - y.reshape(-1, 1)) ** 2)
            self.loss_history.append(loss)
            
            # 反向传播
            self.backward(X, y, activations, layer_inputs)
    
    def predict(self, X):
        activations, _ = self.forward(X)
        return activations[-1].flatten()  # 返回一维数组形式的输出

print("加载数据...")
data = load_data('MLP_data.csv')  # 从CSV文件加载数据集

# 划分特征和目标
X = data[:, :4]  # 特征：longitude, latitude, housing_age, homeowner_income
y = data[:, 4]   # 目标变量：house_price

print("检测并处理异常值...")
original_data_size = X.shape[0]  # 原始样本数量

# 应用IQR方法处理异常值 - threshold=1.5是箱线图异常值检测的标准阈值
X_clean, y_clean, outliers_mask = detect_and_remove_outliers(X, y, method='iqr', threshold=1.5)
cleaned_data_size = X_clean.shape[0]  # 处理后的样本数量
outliers_removed = original_data_size - cleaned_data_size  # 移除的异常样本数量
print(f"原始数据样本数: {original_data_size}")
print(f"处理后数据样本数: {cleaned_data_size}")
print(f"被移除的异常样本数: {outliers_removed} ({outliers_removed/original_data_size*100:.2f}%)")

# 清理前后对比
plt.figure(figsize=(20, 15))
plt.suptitle('数据清理前后对比', fontsize=20)

feature_names = ['经度', '纬度', '房龄', '收入']
for i in range(4):
    # 原始数据散点图
    plt.subplot(2, 4, i+1) 
    plt.scatter(X[:, i], y, alpha=0.5, c='blue', label='所有数据点') 
    plt.scatter(X[outliers_mask, i], y[outliers_mask], alpha=0.7, c='red', label='异常数据点')  # 突出显示异常点
    plt.title(f'清理前: {feature_names[i]}与房价关系', fontsize=12)
    plt.xlabel(feature_names[i])
    plt.ylabel('房价')
    plt.grid(True, linestyle='--', alpha=0.7) 
    plt.legend()
    
    # 清理后的数据散点图
    plt.subplot(2, 4, i+5) 
    plt.scatter(X_clean[:, i], y_clean, alpha=0.5, c='green', label='保留的数据点') 
    plt.title(f'清理后: {feature_names[i]}与房价关系', fontsize=12)
    plt.xlabel(feature_names[i])
    plt.ylabel('房价')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()  # 添加图例

plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.savefig('data_cleaning_comparison.png')
plt.show()

# 转换地理特征 - 处理经度和纬度的非线性关系
X_transformed, cluster_centers = transform_geo_features(X_clean, y_clean, n_clusters=5)


# 可视化聚类结果
plt.figure(figsize=(10, 8))
plt.scatter(X_clean[:, 0], X_clean[:, 1], c=y_clean, cmap='viridis', alpha=0.6)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=100)

# 在聚类中心附近标注区域编号
for i, center in enumerate(cluster_centers):
    plt.annotate(f'区域{i+1}', xy=(center[0], center[1]), 
                 xytext=(center[0]+0.02, center[1]+0.02),
                 color='red', fontsize=12, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.colorbar(label='房价')
plt.title('基于地理位置的聚类分析')
plt.xlabel('经度')
plt.ylabel('纬度')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('geo_clustering.png')
plt.show()

# 标准化数据 - 使得每个特征均值为0，标准差为1
print("数据标准化...")
X_norm, X_mean, X_std = normalize_data(X_transformed)

# 标准化目标变量y
y_norm, y_mean, y_std = normalize_data(y_clean.reshape(-1, 1))
y_norm = y_norm.flatten()  # 将二维数组转回一维

# 划分(80% 训练, 20% 测试)
np.random.seed(42)  # 设置随机种子，确保结果可复现
indices = np.random.permutation(len(X_norm))  # 生成随机排列的索引
train_size = int(0.8 * len(X_norm)) 
train_indices = indices[:train_size] 
test_indices = indices[train_size:] 
X_train, y_train = X_norm[train_indices], y_norm[train_indices]
X_test, y_test = X_norm[test_indices], y_norm[test_indices]

# 定义两层隐藏层的网络架构和参数
learning_rate = 0.1   # 学习率
epochs = 5000        # 迭代次数
# 两层隐藏层的网络
two_layer_architecture = [X_train.shape[1], 20, 10, 1]  # 输入层-20神经元隐藏层-10神经元隐藏层-输出层

# 训练两层隐藏层MLP模型
print(f"训练多层感知机模型 (架构: {'-'.join(map(str, two_layer_architecture))})")
arch_name = f"双隐层模型 ({'-'.join(map(str, two_layer_architecture))})"

print(f"  训练模型参数 - 学习率: {learning_rate}, 迭代次数: {epochs}")

# 训练模型
model = MLP(layer_sizes=two_layer_architecture, learning_rate=learning_rate, epochs=epochs)
model.fit(X_train, y_train)

# 可视化损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), model.loss_history)
plt.title(f'训练损失曲线 (学习率={learning_rate}, 迭代={epochs})')
plt.xlabel('迭代次数')
plt.ylabel('损失 (MSE)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('loss_curve.png')
plt.show()

# 评估模型
y_pred = model.predict(X_test)
mse = np.mean((y_pred - y_test) ** 2)

# 计算原始尺度的MSE
y_pred_orig = y_pred * y_std + y_mean
y_test_orig = y_test * y_std + y_mean
orig_mse = np.mean((y_pred_orig - y_test_orig) ** 2)

print(f"\nMLP模型评估结果:")
print(f"学习率: {learning_rate}, 迭代: {epochs}, "
      f"标准化MSE: {mse:.6f}")

# 计算R²
ss_total = np.sum((y_test_orig - np.mean(y_test_orig)) ** 2)
ss_residual = np.sum((y_test_orig - y_pred_orig) ** 2)
r_squared = 1 - (ss_residual / ss_total)
print(f"R²(决定系数): {r_squared:.4f}")

# 可视化预测结果
plt.figure(figsize=(12, 10))
# 真实与预测房价比较
plt.subplot(2, 2, 1)
plt.scatter(y_test_orig, y_pred_orig, alpha=0.5)
plt.plot([min(y_test_orig), max(y_test_orig)], [min(y_test_orig), max(y_test_orig)], 'r--')
plt.title(f'房价预测结果 (学习率={learning_rate}, 迭代={epochs})')
plt.xlabel('真实房价')
plt.ylabel('预测房价')

# 收入与房价关系
plt.subplot(2, 2, 2)
plt.scatter(X_clean[test_indices, 3], y_test_orig, alpha=0.5, label='真实值')
plt.scatter(X_clean[test_indices, 3], y_pred_orig, alpha=0.5, label='预测值')
plt.title('收入与房价关系')
plt.xlabel('收入')
plt.ylabel('房价')
plt.legend()

# 房龄与房价关系
plt.subplot(2, 2, 3)
plt.scatter(X_clean[test_indices, 2], y_test_orig, alpha=0.5, label='真实值')
plt.scatter(X_clean[test_indices, 2], y_pred_orig, alpha=0.5, label='预测值')
plt.title('房龄与房价关系')
plt.xlabel('房龄')
plt.ylabel('房价')
plt.legend()

# 地理位置与房价关系
plt.subplot(2, 2, 4)
sc = plt.scatter(X_clean[test_indices, 0], X_clean[test_indices, 1], c=y_pred_orig, cmap='viridis')
plt.colorbar(sc, label='预测房价')
plt.title('地理位置与房价关系')
plt.xlabel('经度')
plt.ylabel('纬度')

plt.tight_layout()
plt.savefig('房价预测结果.png')
plt.show()

# 可视化网络架构
def visualize_network_architecture(model, title='网络架构'):
    plt.figure(figsize=(10, 6))
    
    layer_sizes = model.layer_sizes
    n_layers = len(layer_sizes)
    
    layer_names = ['输入层'] 
    for i in range(1, n_layers-1):
        if i == 1:
            layer_names.append('隐层1\n(tanh)')
        else:
            layer_names.append(f'隐层{i}\n(ReLU)')
    layer_names.append('输出层\n(线性)')
    
    for i, (size, name) in enumerate(zip(layer_sizes, layer_names)):
        x = i/(n_layers-1)
        neurons_y = np.linspace(0, 1, size+2)[1:-1]
        
        for j, y in enumerate(neurons_y):
            circle = plt.Circle((x, y), 0.02, color='blue', fill=True)
            plt.gca().add_patch(circle)
            
            if i < n_layers-1:
                next_size = layer_sizes[i+1]
                next_neurons_y = np.linspace(0, 1, next_size+2)[1:-1]
                next_x = (i+1)/(n_layers-1)
                for k, next_y in enumerate(next_neurons_y):
                    plt.plot([x, next_x], [y, next_y], 'gray', alpha=0.3)
        plt.text(x, -0.05, name, ha='center')
        plt.text(x, 1.05, f'{size}个节点', ha='center')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.axis('off')
    plt.title(title)
    plt.savefig('网络结构.png')
    plt.show()
