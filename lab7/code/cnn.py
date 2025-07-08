# #使用说明：请将code文件夹放置于与train文件夹和test文件夹同一目录下
######################################################################
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from PIL import Image
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import os

# 图像预处理流水线：对输入图像进行变换操作
Data_transform=transforms.Compose([
    # 将所有图像调整为统一尺寸224x224像素
    transforms.Resize((224,224)),
    # 将图像转换为PyTorch张量格式，并归一化像素值到[0,1]区间
    transforms.ToTensor(),
    # 标准化处理：使用ImageNet数据集的均值和标准差进行归一化
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

# 图像数据集类：用于管理图像数据的加载和预处理
class MyDataset(Dataset):
    def __init__(self, txt_path, data_dir=""):
        imgs=[]
        labels=[]
        # 从索引文件读取图像数据信息
        Read=open(txt_path,'r')
        for i in Read:
            i=i.rstrip()
            img=i.split(':')
            img_path = os.path.join(data_dir, img[0]) if data_dir else img[0]
            imgs.append(img_path)
            labels.append(int(img[1]))
        self.imgs=imgs
        self.labels=labels
    
    # 返回数据集中样本总数
    def __len__(self):
        return len(self.imgs)
    
    # 根据索引获取数据样本及其对应标签
    def __getitem__(self, index):
        # 使用PIL库加载指定索引的图像文件
        img=Image.open(self.imgs[index])
        # 应用预定义的数据转换操作（缩放、转换张量、归一化）
        img=Data_transform(img)
        # 获取对应的分类标签
        label=self.labels[index]
        return img,label
    
# 定义卷积神经网络（CNN）模型
class CNN(nn.Module):
    def __init__(self):
        # 初始化父类属性
        super(CNN,self).__init__()
        # 定义卷积层
        # 第一层卷积
        self.conv1=nn.Sequential(
            # 卷积操作：提取特征
            nn.Conv2d(
                in_channels=3, # 输入通道数（灰度图为1，彩色图为3）
                out_channels=64, # 卷积核数量，决定输出通道数
                kernel_size=3, # 卷积核大小3x3
                stride=1, # 步长为1
                padding=1, # 填充以保持输出尺寸与输入相同
            ),
            # 批量归一化：加速训练并稳定模型
            nn.BatchNorm2d(num_features=64),
            # 激活函数：ReLU
            nn.ReLU(inplace=True),
            # 最大池化：降维并保留重要特征
            nn.MaxPool2d(
                kernel_size=2, # 池化核大小2x2
                stride=2, # 步长为2
            ),
        )

        # 第二层卷积
        self.conv2=nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )

        # 第三层卷积
        self.conv3=nn.Sequential(
            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4,4),
        )

        # 全连接层（输出层）
        self.output=nn.Sequential(
            # Dropout：以0.5的概率随机丢弃神经元，防止过拟合
            nn.Dropout(0.5),
            nn.Linear(256*14*14,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Dropout(0.5),
            nn.Linear(256,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            # 最终输出类别数为5
            nn.Linear(256,5),
        )

    # 前向传播：定义数据流经网络的方式
    def forward(self,x):
        # 卷积层提取特征
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        # 展平：将多维张量转换为一维向量
        x=x.flatten(1)
        # 全连接层分类
        x=self.output(x)
        return x
# 使用训练集进行模型训练
def TrainNetwork(epoch, model, device, criterion, optim, lr_scheduler, train_loader, test_loader):
    # 存储训练过程中的各项指标
    train_losses = []        # 每批次训练损失
    train_epoch_losses = []  # 每个epoch的平均训练损失
    train_accuracies = []    # 每个epoch的训练准确率
    test_losses = []         # 每个epoch的测试损失
    test_accuracies = []     # 每个epoch的测试准确率
    
    # 对训练集进行多轮迭代训练
    for i in range(epoch):
        # 训练模式：启用BN和Dropout层
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for j,(imgs,labels) in enumerate(train_loader):
            print("epoch:{} batch:{}".format(i+1,j+1))
            imgs=imgs.to(device)
            labels=labels.to(device)
            imgs=imgs.reshape(-1,3,224,224)
            # 输入数据通过模型得到预测输出
            outputs=model(imgs)
            # 计算损失值（交叉熵损失）
            loss=criterion(outputs,labels)
            train_losses.append(loss.item())
            epoch_loss += loss.item()
            batch_count += 1
            # 清空上一次的梯度
            optim.zero_grad()
            # 反向传播计算梯度
            loss.backward()
            # 更新模型参数
            optim.step()
        
        # 计算该epoch的平均训练损失
        avg_epoch_loss = epoch_loss / batch_count
        train_epoch_losses.append(avg_epoch_loss)
        
        # 更新学习率
        lr_scheduler.step()

        # 评估模式：禁用BN和Dropout层
        model.eval()
        
        # 计算训练集准确率和平均损失
        correct_train = 0
        with torch.no_grad():
            for imgs, labels in train_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                imgs = imgs.reshape(-1, 3, 224, 224)
                # 通过模型得到预测输出
                outputs = model(imgs)
                # 获取预测标签
                predict = outputs.data.max(1, keepdim=True)[1]
                correct_train += predict.eq(labels.data.view_as(predict)).cpu().sum()
        
        # 计算当前epoch的训练准确率
        train_accuracy = correct_train / len(train_loader.dataset)
        train_accuracies.append(train_accuracy)
        
        # 计算测试集准确率和损失
        correct_test = 0
        test_loss = 0.0
        test_batch_count = 0
        
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                imgs = imgs.reshape(-1, 3, 224, 224)
                # 通过模型得到预测输出
                outputs = model(imgs)
                # 计算测试损失
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                test_batch_count += 1
                # 获取预测标签
                predict = outputs.data.max(1, keepdim=True)[1]
                correct_test += predict.eq(labels.data.view_as(predict)).cpu().sum()
        
        # 计算当前epoch的测试准确率和平均损失
        test_accuracy = correct_test / len(test_loader.dataset)
        test_accuracies.append(test_accuracy)
        avg_test_loss = test_loss / test_batch_count
        test_losses.append(avg_test_loss)
        
        # 输出每个epoch的训练和测试指标
        print('Epoch: {} Train Loss: {:.4f}, Train Accuracy: {}/{} ({:.3f}%)'.format(
            i+1, avg_epoch_loss, correct_train, len(train_loader.dataset),
            100. * train_accuracy))
        print('Epoch: {} Test Loss: {:.4f}, Test Accuracy: {}/{} ({:.3f}%)'.format(
            i+1, avg_test_loss, correct_test, len(test_loader.dataset),
            100. * test_accuracy))
            
    # 返回所有收集的指标
    metrics = {
        'batch_losses': train_losses,          # 每批次训练损失
        'train_losses': train_epoch_losses,    # 每个epoch的训练损失
        'train_accuracies': train_accuracies,  # 每个epoch的训练准确率
        'test_losses': test_losses,            # 每个epoch的测试损失
        'test_accuracies': test_accuracies     # 每个epoch的测试准确率
    }
    
    return metrics


# 计算和保存可视化结果
def save_visualization(metrics, params):
    # 获取并创建保存目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(os.path.dirname(current_dir), 'results')
    os.makedirs(save_dir, exist_ok=True)
    
    # 构建包含参数信息的文件名
    filename = f"train_results_lr{params['lr']}_bs{params['batch_size']}_ep{params['epochs']}_time{params['time']}s"
    
    # 设置画布大小
    plt.figure(figsize=(15, 10))
    
    # 1. 绘制批次训练损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(range(len(metrics['batch_losses'])), metrics['batch_losses'])
    plt.title(f'Training Batch Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2. 绘制每个epoch的训练和测试损失曲线
    plt.subplot(2, 2, 2)
    epochs = range(1, len(metrics['train_losses']) + 1)
    plt.plot(epochs, metrics['train_losses'], 'b-', label='Train Loss')
    plt.plot(epochs, metrics['test_losses'], 'r-', label='Test Loss')
    plt.title('Train vs Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 3. 绘制训练和测试准确率曲线
    plt.subplot(2, 2, 3)
    plt.plot(epochs, metrics['train_accuracies'], 'b-o', label='Train Accuracy')
    plt.plot(epochs, metrics['test_accuracies'], 'r-o', label='Test Accuracy')
    plt.title('Train vs Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 4. 训练和测试准确率对比表格
    plt.subplot(2, 2, 4)
    plt.axis('off')  # 关闭坐标轴
    
    # 创建表格数据
    table_data = []
    headers = ['Epoch', 'Train Loss', 'Train Acc', 'Test Loss', 'Test Acc']
    for i in range(len(epochs)):
        table_data.append([
            f"{i+1}",
            f"{metrics['train_losses'][i]:.4f}",
            f"{metrics['train_accuracies'][i]:.4f}",
            f"{metrics['test_losses'][i]:.4f}",
            f"{metrics['test_accuracies'][i]:.4f}"
        ])
    
    # 表格位置和样式
    table = plt.table(cellText=table_data, 
                      colLabels=headers,
                      loc='center',
                      cellLoc='center',
                      bbox=[0.0, 0.0, 1.0, 1.0])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)
    
    # 添加总标题
    plt.suptitle(f'Training Results (Time: {params["time"]}s)\nLR={params["lr"]}, Batch Size={params["batch_size"]}, Epochs={params["epochs"]}', fontsize=16)
    
    # 保存图像
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt_path = os.path.join(save_dir, f"{filename}.png")
    plt.savefig(plt_path)
    print(f"可视化结果已保存至result文件夹")
    
    # 显示图像
    plt.show()


# 计算准确率
def Accurary(model,device,train_loader,test_loader):
    # 测试模式：禁用BN和Dropout层
    model.eval()
    correct_Train=0
    correct_Test=0
    # 禁用梯度计算以节省内存
    with torch.no_grad():
        for imgs,labels in train_loader:
            imgs=imgs.to(device)
            labels=labels.to(device)
            imgs=imgs.reshape(-1,3,224,224)
            # 通过模型得到预测输出
            outputs=model(imgs)
            # 获取预测标签
            predict=outputs.data.max(1,keepdim=True)[1]
            correct_Train+=predict.eq(labels.data.view_as(predict)).cpu().sum()

        for imgs,labels in test_loader:
            imgs=imgs.to(device)
            labels=labels.to(device)
            imgs=imgs.reshape(-1,3,224,224)
            # 通过模型得到预测输出
            outputs=model(imgs)
            # 获取预测标签
            predict=outputs.data.max(1,keepdim=True)[1]
            correct_Test+=predict.eq(labels.data.view_as(predict)).cpu().sum()

    print('Train Accuracy: {}/{} ({:.3f}%)'.format(correct_Train, len(train_loader.dataset),
    100. * correct_Train / len(train_loader.dataset)))

    print('Test Accuracy: {}/{} ({:.3f}%)\n'.format(correct_Test, len(test_loader.dataset),
    100. * correct_Test / len(test_loader.dataset)))


if __name__ == '__main__':
    # 获取当前脚本所在目录作为基础路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # train.txt和test.txt与脚本在同一目录下
    train_txt = os.path.join(current_dir, 'train.txt')
    test_txt = os.path.join(current_dir, 'test.txt')
    
    # 使用时图像文件放置于父目录中！！！！！！！！！！！！
    data_dir = os.path.dirname(current_dir)
    
    # 定义训练参数
    batch_size = 128
    learning_rate = 0.001
    epochs = 10
    
    dataset_train = MyDataset(train_txt, data_dir)
    dataset_test = MyDataset(test_txt, data_dir)
    
    # 使用DataLoader创建批处理迭代器
    loader_train=torch.utils.data.DataLoader(dataset_train,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True)
    loader_test=torch.utils.data.DataLoader(dataset_test,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=False)
    # 检测并选择可用的计算硬件设备
    hardware=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化CNN网络实例
    net=CNN().to(hardware)
    # 使用交叉熵作为损失度量
    loss_fn=nn.CrossEntropyLoss().to(hardware)
    # 配置Adam优化算法
    optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate)
    # 设置学习率动态调整策略
    scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
    
    # 开始计时
    import time
    time_begin = time.time()
    
    # 执行网络训练流程
    metrics = TrainNetwork(epochs, net, hardware, loss_fn, optimizer, scheduler, loader_train, loader_test)
    
    # 结束计时并计算训练耗时
    time_finish = time.time()
    duration = time_finish - time_begin
    duration_str = f"{duration:.2f}"
    print(f"模型训练完成，总计用时: {duration_str} 秒")
    
    # 生成训练结果可视化
    train_params = {
        'lr': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'time': duration_str
    }
    save_visualization(metrics, train_params)
    
    # 计算准确率
    Accurary(net, hardware, loader_train, loader_test)
