import pic2binary
import readXML
import batchReadXml
import numpy as np

JREfolderPath = '/home/jackey/private/project/ML/data/VOC2012/JPEGImagesTmp'
AnnotationsPath = '/home/jackey/private/project/ML/data/VOC2012/AnnotationsTmp'

def load_binary_file(x_path,tag_path):
 #   ys = pic2binary.bin2tensor2('/home/jackey/private/project/ML/data/VOC2012/Binary/input_tag.txt', isTag=True)
  #  xs = pic2binary.bin2tensor2('/home/jackey/private/project/ML/data/VOC2012/Binary/input_x.txt', isTag=False)
    xs = pic2binary.bin2tensor2(x_path, isTag=False)
    ys = pic2binary.bin2tensor2(tag_path, isTag=True)
    ys = np.reshape(ys,[-1,24])
    return xs,ys
"""
xs,ys =load_binary_file()

print(xs)
print(ys)
print(xs.shape)
print(ys.shape)
"""
