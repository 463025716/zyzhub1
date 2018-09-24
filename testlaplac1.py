import Image , ImageChops
import math


def neglaplacian(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    newimg = Image.new(img.mode,img.size,None)
    width , height = img.size

    mask = {}
    mask[(0,0)] = 0
    mask[(0,1)] = 1
    mask[(0,2)] = 0
    mask[(1,0)] = 1
    mask[(1,1)] = -4
    mask[(1,2)] = 1
    mask[(2,0)] = 0
    mask[(2,1)] = 1
    mask[(2,2)] = 0

    for row in range(1,width-1,1):
        for col in range(1,height-1,1):
            pixelmask = ct = 0
            for i in range(0,3,1):
                for j in range(0,3,1):
                    r,g,b = img.getpixel((row+i-1,col+j-1))
                    value = (r+g+b)/3
                    c = value*mask[(i,j)]
                    ct = ct+c
        
            newimg.putpixel((row+1,col+1),(ct,ct,ct))
    img.show()
    newimg.show()
    newimg.save("faceneg6.jpg")
    #newimg = ImageChops.invert(newimg)
    newim = ImageChops.difference(img,newimg)
    newim.show()
    width,height = newim.size
    for i in range(width):
        for j in range(height):
            r,g,b = newim.getpixel((i,j))
            c = (r+g+b)/3
            if c >70.6:
                c = 255
            else :
                c = 0
            newim.putpixel((i,j),(c,c,c))
    newim.show()
    newim.save("fce2logg.jpg")

img = Image.open("face.jpg")
neglaplacian(img)


            
