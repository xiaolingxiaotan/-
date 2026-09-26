# 问题一：遍历value列时把键值写成“Value”
- 定位与现象：读取完csv表后对得到的迭代器进行遍历时，v = float(row["Value"])显示键值错误KeyError: 'Value'
- 原因：在csv文件中第二列的表头为小写的“value”，而不是“Value”
- 修改：将“Value”改为“value”

# 问题二：没有自动创建输出文件
- 定位与现象：在输出清洗的数据时，在f = open(output_path, "w")行显示文件未找到错误FileNotFoundError: [Errno 2] No such file or directory: '/out/cleaned_data.csv'
- 原因：在打开这个路径之前，只是将文件路径拼接，没有创建这个输出文件夹，而open函数的write模式只能创建文件不能创建文件夹
- 修改：将输出文件创建出来————用os.makedirs()函数（使用绝对路径）

# 问题三：标准差的计算结果为0
- 定位与现象：结果输出的标准差为0，不符合表中数据的实际，定位到标准差的代码：acc += (v - mean)
- 原因：根据标准差的计算公式，这里加的应该是(v - mean)的平方，且就算加上了平法，得到的std也是方差，还要取根号
- 修改：将(v - mean)带上平方，且std加上取根号操作（为了更好的维护代码，将标准差和均值的计算封装为函数）
