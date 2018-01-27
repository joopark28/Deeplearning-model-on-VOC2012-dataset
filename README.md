# Deeplearning-model-on-VOC2012-dataset
从 VOC2012/AnnotationsTmp/ 的.xml 中 提取 pictureName width depth className
字符串拼接 找到 pictureName  并将图片转化为数组
依据className 建立 tag array 
并 将数组 和 和tag array 保存为2进制文件

模型从2进制文件中读取数组 并做相应运算
