# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 21:17:34 2018

@author: zyz
"""
'''
轮廓检测
OpenCV的imread()只能加载本地的图片，并不能通过网址加载图片。
'''
#  ----------------------------------------------       灰度化、二值化、增强、去噪、图像分割
import os,base64
import io 
import logging,IPython
from io import BytesIO,StringIO
import requests as req

import aircv as ac
import pylab
from PIL import Image
from PIL import ImageEnhance
import numpy as np
import cv2
import imutils
import argparse
from imutils import contours 
from PIL import Image
from skimage import filters

from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=r"d:/msyh.ttf", size=12) 

#-------------------------------------------------------------空间流量已用完
'''
url='http://img.027cgb.com/607871/20180917-0miss.jpg'
#response = req.get(url) # 将这个图片保存在内存
tuple_temp = urllib.request.urlretrieve(url)# 访问URL并保存临时文件
img_down = Image.open(tuple_temp[0])  # 打开临时文件
'''
#-------------------------------------------------------------空间流量已用完
'''
 python中， 每一个变量在内存中创建，我们可以通过变量来查看内存中的值
'''
#img_down = cv2.imread('D:/Software/Python/SaveDone/ImageRecognition/20180917-0miss.jpg',0)#打开为灰度图像

#
#plt.figure()
#plt.imshow(img_down,'gray') #必须规定为显示的是什么图像
#plt.xticks([]),plt.yticks([]) #隐藏坐标线 
#plt.title(u'原图',fontproperties=myfont)
#plt.show() #显示
#img=np.asarray(img_down)
img_open = Image.open('D:/Software/Python/SaveDone/ImageRecognition/20180917-0origin.jpg')
#img_open = Image.open('D:/Software/Python/SaveDone/ImageRecognition/20180917-0origin.jpg')
#-------------------------------------------------------------------------------------图片切割

#-------------------------------------------------------------------------------- 1 灰度化
img=np.asarray(img_open)
#plt.figure()
#plt.imshow(img,'gray') #必须规定为显示的是什么图像
#plt.xticks([]),plt.yticks([]) #隐藏坐标线 
#plt.title(u'原图',fontproperties=myfont)
#plt.show() #显示
'''
目标区
square = np.array([ [ (1332, 930) ,(1332, 375),(1220, 375),(1220, 930)] ])
'''
#square1 = np.array([ [ (2048, 0) ,(2048, 375),(0, 375),(0, 0)] ])
#square2 = np.array([ [ (1220, 375) ,(1220, 1400),(0, 1400),(0, 375)] ])
#square3 = np.array([ [ (2048, 375) ,(2048, 930),(1330, 930),(1330, 375)] ])
#square4 = np.array([ [ (2048, 930) ,(2048, 1400),(1220, 1400),(1220, 930)] ])
#cv2.fillPoly(img, [ square1,square2,square3,square4], 1)#(169,169,169,0.2)
#plt.figure()
#plt.imshow(img)
#plt.axis('off')
#plt.show()


#img_gray0 = img_open.convert('L')   #转化为灰度图
#L = I.convert('1')   #转化为二值化图
#plt.figure()
#plt.imshow(img_gray0) #必须规定为显示的是什么图像
#plt.xticks([]),plt.yticks([]) #隐藏坐标线 
#plt.title(u'灰度图',fontproperties=myfont)
#plt.show() #显示
'''
 彩色图像所包含的数据量要远远大于黑白图像
'''
'''
     cv2.cvtColor() 用这个函数把图像j进行颜色空间转换
     经常用到的颜色空间转换是: BGR<->Gray 和 BGR<->HSV
'''
if len(img.shape) == 3 or len(img.shape) == 4:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    img_gray0=img_open.convert('L')
    img_gray = np.asarray(img_gray0)
'''
灰度图像二值化的目的在于将图像中每个像素点所具有的各自不同的多种灰度值进行重新赋值，重新赋值的结果为 0 或 255。
如果是 0，就划为背景；如果是 255，就划为目标。
'''  

#--------------------------------------------------------------------------------- 2-1 固定阈值二值化    
'''
ret,thresh1 = cv2.threshold()
'''
#rectKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(2,3))
#tophat=cv2.morphologyEx(img_gray,cv2.MORPH_TOPHAT,rectKernel)
#plt.figure()
#plt.imshow(tophat,'gray')
#plt.title('tophat')
#plt.xticks([]),plt.yticks([])
#plt.show()


#ret,thresh1 = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
#ret,thresh2 = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img_gray,127,255,cv2.THRESH_TRUNC)
#ret,thresh4 = cv2.threshold(img_gray,127,255,cv2.THRESH_TOZERO)
#ret,thresh5 = cv2.threshold(img_gray,127,255,cv2.THRESH_TOZERO_INV)
#titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
#images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

#plt.figure()
#for i in np.arange(6):
#    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
#    plt.title(titles[i])
#    plt.xticks([]),plt.yticks([])
#plt.show()

#plt.figure()
#plt.imshow(thresh3) #必须规定为显示的为什么图像
#plt.imshow(thresh3, 'binary') #必须规定为显示的为什么图像
#plt.title(u'固定阈值二值化',fontproperties=myfont)
#plt.xticks([]),plt.yticks([]) #隐藏坐标线 
#plt.show() #显示

#--------------------------------------------------------------------------------- 2-2 自适应阈值二值化
'''
 dst = cv2.adaptiveThreshold()  , 该函数有6个参数：
 第一个原始图像
 第二个像素值上限
 第三个自适应方法Adaptive Method: 
    — cv2.ADAPTIVE_THRESH_MEAN_C ：领域内均值 
    —cv2.ADAPTIVE_THRESH_GAUSSIAN_C ：领域内像素点加权和，权重为一个高斯窗口
 第四个值的赋值方法：只有cv2.THRESH_BINARY 和cv2.THRESH_BINARY_INV
 第五个Block size:规定领域大小（一个正方形的领域）
 第六个常数C，阈值等于均值或者加权值减去这个常数（为0相当于阈值 就是求得领域内均值或者加权值） 
这种方法理论上得到的效果更好，相当于在动态自适应的调整属于自己像素点的阈值，而不是整幅图像都用一个阈值。
'''
'''
ret, dst = cv2.threshold(src, thresh, maxval, type)
src： 输入图，只能输入单通道图像，通常来说为灰度图
dst： 输出图
thresh： 阈值
maxval： 当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
type：二值化操作的类型，包含以下5种类型： 
      cv2.THRESH_BINARY； cv2.THRESH_BINARY_INV； cv2.THRESH_TRUNC； cv2.THRESH_TOZERO；cv2.THRESH_TOZERO_INV
'''
#img_at_mean = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 3)
#img_at_mean2 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
#retval,img_at_fixed = cv2.threshold(img_gray, 25, 255, cv2.THRESH_BINARY)

#plt.figure()
#plt.subplot(2, 2, 1),plt.imshow(img_gray,'gray')
#plt.subplot(2, 2, 2),plt.imshow(img_at_mean,'gray')
#plt.subplot(2, 2, 3),plt.imshow(img_at_mean2,'gray')
#plt.subplot(2, 2, 4),plt.imshow(img_at_fixed,'gray')
#plt.show() #显示

#plt.figure()
#plt.imshow(img_at_fixed, 'gray') #必须规定为显示的为什么图像
#plt.title(u'自适应阈值二值化',fontproperties=myfont)
#plt.xticks([]),plt.yticks([]) #隐藏坐标线 
#plt.show() #显示

img2v=thresh3
#---------------------------------------------------------------------------------- 3 图像增强
'''
图像增强是以突出图像中的有意义信息，减少以至剔除冗余信息为目的，利用相关
的方法获取对具体应用有价值的图像，或是对人的视觉响应、机器处理更容易接受的图
像的处理技术
图像增强 包括【空间域法（灰度级校正、灰度变换和直方图修正）、频率域法（图像平滑和图像锐化）】
--
空间域法的操作对象不同，所以图像增强操作可分为点操作和邻域操作两种。
【空间域法】可以使图像成像均匀，或扩大图像动态范围，提高对比度，包括灰度级校正、灰度变换和直方图修正等方法。
【频率域法】分为图像平滑和图像锐化两种。
图像平滑主要用于除去噪声，但在去噪的同时也会导致边缘模糊，包括均值滤波、中值滤波等方法；
图像锐化主要在于图像识别时突出目标的边缘轮廓，包括梯度法、统计差值法等。常用的算子为Sobel和Laplacian
--
'''
#---------------------------------------------------------------------------------图像锐化
'''
要使用16位有符号的数据类型，即cv2.CV_16S，dx和dy表示的是求导的阶数，0表示这个方向上没有求导，一般为0、1、2
'''

#--------------------------------------------------------------------------------边缘增强滤波
#from PIL import ImageFilter
#
#img2vI = Image.fromarray(img2v) 
#img_edge_enhance = img2vI.filter(ImageFilter.EDGE_ENHANCE)
#plt.figure()
#plt.imshow(img_edge_enhance,'gray') #必须规定为显示的为什么图像
#plt.title(u'边缘增强',fontproperties=myfont)
#plt.xticks([]),plt.yticks([]) #隐藏坐标线 
#plt.show() #显示

#---------------------------------------------------------------------------------直方图增强
lut = np.zeros(256, dtype = img2v.dtype )#创建空的查找表  
hist= cv2.calcHist([img2v], #计算图像的直方图  
    [0], #使用的通道  
    None, #没有使用mask  
    [256], #it is a 1D histogram  
    [0.0,255.0])  
      
minBinNo, maxBinNo = 0, 255 
#计算从左起第一个不为0的直方图柱的位置  
for binNo, binValue in enumerate(hist):  
    if binValue != 0:  
        minBinNo = binNo  
        break  
#计算从右起第一个不为0的直方图柱的位置  
for binNo, binValue in enumerate(reversed(hist)):  
    if binValue != 0:  
        maxBinNo = 255-binNo  
        break  
#print minBinNo, maxBinNo  
  
#生成查找表
for i,v in enumerate(lut):  
#    print i  
    if i < minBinNo:  
        lut[i] = 0  
    elif i > maxBinNo:  
        lut[i] = 255  
    else:  
        lut[i] = int(255.0*(i-minBinNo)/(maxBinNo-minBinNo)+0.5)  
  
#计算,调用OpenCV cv2.LUT函数,参数 image --  输入图像，lut -- 查找表 
result = cv2.LUT(img2v, lut)  
#plt.figure()
#plt.imshow(result,'gray') 
#plt.title(u'直方图增强',fontproperties=myfont)
#plt.xticks([]),plt.yticks([]) #隐藏坐标线 
#plt.show() #显示
#--------------------------------------------------------------------------------  4  图像去噪  开始
'''
不同的噪声与图像本身的有用信号之间关系也不尽相同，噪声与图像信号之间有的是相关的，有的是无关的，
但不管是什么样的关系，噪声的存在不可避免。因此，为了更好地完成图像识别，去噪则是必要的一步操作过程，
合理的去噪方法也必将对图像准确识别起到事半功倍的作用。
---
图像去噪处理就是滤波处理，滤波方法的选取要考虑实际的噪声类别，经去噪处理的图像要保持原图像自身的真实度。
---
'''

'''
img_smooth = cv2.filter2D() 来对图像进行卷积操作(图像滤波)
'''
#----------------------------------------------------------------------------------  4-1 卷积操作  去噪
#卷积去噪
#img_temp=np.asarray(img_edge_enhance)
img_temp=np.asarray(result)
kernel = np.ones((5,5),np.float32)/25
#kernel = np.ones((3,3),np.float32)/9              #  当设定为  kernel = np.ones((3,3),np.float32)/9  ,  滤波结果和 高斯滤波 sigma=0.8 时  效果一样 
img_2D = cv2.filter2D(img_temp,-1,kernel)#图像卷积运算函数，#输出图像与输入图像大小相同

plt.figure()
plt.imshow(img_2D,'gray') #必须规定为显示的为什么图像
#plt.imshow(img_smooth) #or
plt.title(u'卷积滤波',fontproperties=myfont)
plt.xticks([]),plt.yticks([]) #隐藏坐标线 
plt.show() #显示

#--------------------------------------------------------------------------------   4-2 高斯滤波
# 高斯滤波，sigma越大越模糊
#img_gaussian = filters.gaussian(img_temp,sigma=0.9)
#plt.figure()
#plt.imshow(img_gaussian,'gray') #必须规定为显示的为什么图像
##plt.imshow(img_smooth) #or
#plt.title(u'高斯滤波',fontproperties=myfont)
#plt.xticks([]),plt.yticks([]) #隐藏坐标线 
#plt.show() #显示

#--------------------------------------------------------------------------------  4-3 深度边缘增强滤波
# 深度边缘增强滤波
#edge_enhance_more = img_edge_enhance.filter(ImageFilter.EDGE_ENHANCE_MORE)
#img_contour = img_edge_enhance.filter(ImageFilter.CONTOUR)
#plt.figure()
#plt.imshow(img_contour,'gray') #必须规定为显示的为什么图像
#plt.title(u'深度边缘滤波',fontproperties=myfont)
#plt.xticks([]),plt.yticks([]) #隐藏坐标线 
#plt.show() #显示

##--------------------------------------------------------------------------------  4-4 轮廓滤波，将图像中的轮廓信息全部提取出来
##轮廓滤波

#img_contour = img2vI.filter(ImageFilter.CONTOUR)
#plt.figure()
#plt.imshow(img_contour,'gray') #必须规定为显示的为什么图像
#plt.title(u'轮廓滤波',fontproperties=myfont)
#plt.xticks([]),plt.yticks([]) #隐藏坐标线 
#plt.show() #显示
#---------------------------------------------------------------------------------- 4-5  图像去噪  ---  结束

'''
cv2.fastNlMeansDenoising()   使用对象为灰度图。
cv2.fastNlMeansDenoisingColored()  使用对象为彩色图。
cv2.fastNlMeansDenoisingMulti() 适用于短时间的图像序列(灰度图像)
cv2.fastNlMeansDenoisingColoredMulti() 适用于短时间的图像序列(彩色图像)
共有参数：  
h : 决定过滤器强度。h 值高可以很好的去除噪声,但也会把图像的细节抹去。(取 10 的效果不错) 
hForColorComponents : 与 h 相同,但使用与彩色图像。(与 h 相同) 
templateWindowSize : 奇数。(推荐值为 7) 
searchWindowSize : 奇数。(推荐值为 21)
'''
#----------------------------------------------------------------------------------  4-2   均值去噪
#dst = cv2.fastNlMeansDenoisingMulti(sharpen,2, 5, None, 4, 7, 35)
#plt.figure()
#plt.imshow(dst)
#plt.show()


#from scipy.ndimage import filters
#import rof
#U,T = rof.denoise(sharpen,sharpen)
#G = filters.gaussian_filter(sharpen,8)
# 
## 保存生成结果
##from scipy.misc import imsave
##imsave('synth_rof.pdf',U)
##imsave('synth_gaussian.pdf',G)
#
#plt.figure()
#plt.imshow(G,'gray') 
#plt.title(u'高斯滤波',fontproperties=myfont)
#plt.xticks([]),plt.yticks([]) #隐藏坐标线 
#plt.show() #显示

#----------------------------------------------------------图像边缘检测
#img_edge=edge(img_smooth)
#图像梯度
#xgrad=cv2.Sobel(img_2Dnoise,cv2.CV_16SC1,1,0)
#ygrad=cv2.Sobel(img_2Dnoise,cv2.CV_16SC1,0,1)
##计算边缘
##50和150参数必须符合1：3或者1：2
#edge_output=cv2.Canny(xgrad,ygrad,50,150)
#img_edge=cv2.bitwise_and(img,img,mask=edge_output)
#
#plt.figure()
#plt.imshow(img_edge,'gray') #必须规定为显示的为什么图像
##plt.imshow(img_smooth) #or
#plt.title(u'边缘',fontproperties=myfont)
#plt.xticks([]),plt.yticks([]) #隐藏坐标线 
#plt.show() #显示
#
##-------------------------------------------------------  提取 对象 梯度
#from skimage import img_as_ubyte
#img_2D1 = img_as_ubyte(img_2D)
'''
cv2.CV_64F

梯度是求导，三种梯度滤波器，（高通滤波器）:Sobel,Scharr和Laplacian
SobelScharr:求一阶或二阶导数，Scharr是对Sobel(使用小的卷积核求解梯度角度时)的优化。
Laplacian:是求二阶导数
Sobel 算子是高斯平滑u微分操作的结合体，抗噪声能力很好，可以设定求导的方向(xorder或yorder)
还可以设定使用的卷积核的大小(ksize)。如果ksize=-1,会使用3*3的Scharr滤波器，它的效果要
比3*3的Sobel滤波器好，3*3的Scharr滤波器卷积核如下:

    Sobel 和 Canny 结果对比
 https://www.cnblogs.com/mfmdaoyou/p/6781321.html   
'''
gradX = cv2.Sobel(img_2D, ddepth=cv2.CV_64F, dx=1, dy=0,ksize=3)
gradY = cv2.Sobel(img_2D, ddepth=cv2.CV_64F, dx=0, dy=1,ksize=3)
#gradient = cv2.subtract(gradX, gradY) ##将图像 gradX 与 gradY 相减
gradient = cv2.addWeighted(gradX, 0.5, gradY, 0.5, 0) # 用cv2.addWeighted(...)函数将其组合起来。
gradient = cv2.convertScaleAbs(gradient) # 用convertScaleAbs()函数将其转回原来的uint8形式。否则将无法显示图像，而只是一副灰色的窗口。

plt.figure()
plt.imshow(gradient,'gray') 
plt.title(u'Sobel 边缘检测',fontproperties=myfont)
plt.xticks([]),plt.yticks([]) #隐藏坐标线 
plt.show() #显示


'''

'''
#gray_lapa = cv2.Laplacian(img_2D,cv2.CV_64F,ksize = 3)
##gray_lapa = cv2.Laplacian(img_2D,cv2.CV_16S,ksize = 3)

#plt.figure()
#plt.imshow(dst,'gray') 
#plt.title(u'拉普拉斯',fontproperties=myfont)
#plt.xticks([]),plt.yticks([]) #隐藏坐标线 
#plt.show() #显示

'''
Canny Edge
'''
#edge_output = cv2.Canny(img2v, 130, 50)#-----------------OK
#edge_output = cv2.Canny(img_2D, 45, 180)

#plt.figure()
#plt.imshow(edge_output,'gray') 
#plt.title(u'Canny Edge',fontproperties=myfont)
#plt.xticks([]),plt.yticks([]) #隐藏坐标线 
#plt.show() #显示










#------------------------------------------------------------------------------------数据存入数据库
import sqlserver
import datetime
server="127.0.0.1"  #服务器IP或服务器名称
user="sa"           #登陆数据库所用账号
password="pio-tech2012" #该账号密码
database="ImageRec"  #数据库名称
dt=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
ImageID='20180924-06'
position1='正常'
position2='正常'
position3='正常'
position4='正常'
position5='正常'

ms = sqlserver.MSSQL(host=server,user=user,pwd=password,db=database)
insql ="insert into bogie values ('%s','%s','%s','%s','%s','%s','%s')"%(dt,ImageID,position1,position2,position3,position4,position5)
ms.ExecNonQuery(insql)





