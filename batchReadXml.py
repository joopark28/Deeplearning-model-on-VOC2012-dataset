import readXML
import os
import numpy as np
import pic2binary
def batchReadXml(AnnotationsPath,LastXmlNum):
    xs = np.random.random((1,500,500,3))
    ys = np.zeros([1,24])
    print(ys)
    JREFold = "/home/jackey/private/project/ML/data/VOC2012/JPEGImagesTmp/"
   # Annotations = "/home/jackey/private/project/ML/data/VOC2012/AnnotationsTmp"
    for i in range(500):
        fileName = AnnotationsPath
        listStr = [AnnotationsPath,"/2009_000"+str(i)+'.xml']
        xmlPath = ''.join(listStr)
        if not os.path.exists(xmlPath):
            continue
        print(xmlPath)
        imageName, widthPixels, heightPixels, className = readXML.getXMLAttribute(xmlPath)
        print('batch read xmlfile success')
        JREPath = JREFold+imageName
        #ys = pic2binary.tagNUM(className)
        input_x,input_y =pic2binary.image_to_array(
            JREPath,width=int(widthPixels),depth=int(heightPixels),className=className)
        print(i)
        xs = np.concatenate((xs,input_x),axis=0)
        ys = np.concatenate((ys,input_y),axis=0)
        print(ys.shape)
    xs.tofile("/home/jackey/private/project/ML/data/VOC2012/Binary/input_xs.txt")
    ys.tofile("/home/jackey/private/project/ML/data/VOC2012/Binary/input_ys.txt")
    print('array 2 bin file success')


#batchReadXml("/home/jackey/private/project/ML/data/VOC2012/AnnotationsTmp",5311+1)


