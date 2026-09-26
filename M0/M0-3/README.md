# 问题一：遍历value列时把键值写成“Value”
- 定位与现象：读取完csv表后对得到的迭代器进行遍历时，v = float(row["Value"])显示键值错误KeyError: 'Value'
- 原因：在csv文件中第二列的表头为小写的“value”，而不是“Value”
- 修改：将“Value”改为“value”