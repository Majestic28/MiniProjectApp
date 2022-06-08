from http.client import IM_USED
from PIL import Image

im = Image.open("vish.jpeg")
width, height = im.size
print(width,height)
newsize = (100,100)
im = im.resize(newsize)
im.save("vish1.jpg")
width, height = im.size
