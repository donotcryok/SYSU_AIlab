class StuData:
    def __init__(self, filename):
        """
        构造函数，初始化 StuData 类的实例。
        :param filename: str，包含学生信息的文本文件名，文件扩展名为 .txt。
        """
        self.data = []  # 初始化一个空列表，用于存储学生信息
        self.filename = filename  # 存储文件名
        self.read_data()  # 调用 read_data 方法，从文件中读取学生信息

    def read_data(self):
        """
        从文件中读取学生信息，并存储到 data 列表中。
        """
        with open(self.filename, 'r', encoding='utf-8') as file:  # 打开文件进行读取
            for line in file:  # 逐行读取文件内容
                parts = line.strip().split()  # 去除行首尾空白字符，然后分割字符串
                if len(parts) == 4:  # 确保每行数据包含四个属性
                    name, stu_num, gender, age = parts  # 解包属性
                    age = int(age)  # 将年龄转换为整数类型
                    self.data.append([name, stu_num, gender, age])  # 将学生信息添加到 data 列表

    def add_data(self, name, stu_num, gender, age):
        """
        向 data 列表中添加一个新的学生信息。
        :param name: str，学生的姓名。
        :param stu_num: str，学生的学号。
        :param gender: str，学生的性别，"M" 表示男性，"F" 表示女性。
        :param age: int，学生的年龄。
        """
        self.data.append([name, stu_num, gender, int(age)])  # 将新学生信息添加到 data 列表

    def sort_data(self, attribute):
        """
        根据指定的属性对学生信息进行排序。
        :param attribute: str，用于排序的属性名称，可以是 'name'、'stu_num'、'gender' 或 'age'。
        """
        if attribute in ['name', 'stu_num', 'gender', 'age']:  # 检查属性是否有效
            self.data.sort(key=lambda x: x[attribute.index(attribute)] if attribute != 'age' else int(x[attribute.index(attribute)]))  # 根据属性排序

    def export_file(self, export_filename):
        """
        将 data 列表中的学生信息导出到一个新的文本文件中。
        :param export_filename: str，导出文件的文件名，文件扩展名为 .txt。
        """
        with open(export_filename, 'w', encoding='utf-8') as file:  # 打开文件进行写入
            for student in self.data:  # 遍历 data 列表中的学生信息
                file.write(f"{student[0]} {student[1]} {student[2]} {student[3]}\n")  # 将学生信息写入文件

# 使用示例
# 假设 student_data.txt 文件位于当前目录下
student_data = StuData('student_data.txt')
print("Initial data:", student_data.data)

# 添加新学生数据
student_data.add_data("Eric", "249", "M", 19)
print("Data after adding Eric:", student_data.data)

# 按学号排序
student_data.sort_data('stu_num')
print("Data after sorting by student number:", student_data.data)

# 导出到新文件
student_data.export_file('new_stu_data.txt')