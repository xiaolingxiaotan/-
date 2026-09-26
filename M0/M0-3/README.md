# 问题一：遍历value列时把键值写成“Value”
- 定位与现象：读取完csv表后对得到的迭代器进行遍历时，v = float(row["Value"])显示键值错误KeyError: 'Value'
- 原因：在csv文件中第二列的表头为小写的“value”，而不是“Value”
- 修改：将“Value”改为“value”

# 问题二：没有自动创建输出文件
- 定位与现象：在输出清洗的数据时，在f = open(output_path, "w")行显示文件未找到错误FileNotFoundError: [Errno 2] No such file or directory: '/out/cleaned_data.csv'
- 原因：在打开这个路径之前，只是将文件路径拼接，没有创建这个输出文件夹，而open函数的write模式只能创建文件不能创建文件夹
- 修改：将输出文件创建出来————用os.makedirs()函数（使用绝对路径）

# 问题三：