#coding=utf-8
import  xml.dom.minidom

def getXMLAttribute(filename):
    dom = xml.dom.minidom.parse(filename)
    root = dom.documentElement
    image = root.getElementsByTagName('filename')[0]
    imageName= image.childNodes[0].nodeValue
    tmp = root.getElementsByTagName('object')[0]
    tmp = tmp.getElementsByTagName('name')[0]
    className = tmp.childNodes[0].nodeValue

    tmp = root.getElementsByTagName('size')[0]
    tmp = tmp.getElementsByTagName('height')[0]
    heightPixels =tmp.childNodes[0].nodeValue
    tmp = root.getElementsByTagName('size')[0]
    tmp = tmp.getElementsByTagName('width')[0]
    widthPixels =tmp.childNodes[0].nodeValue
    return imageName,widthPixels,heightPixels,className

"""
getXMLAttribute('/home/jackey/private/project/ML/data/VOC2012/AnnotationsTmp/2012_001000.xml')
"""
