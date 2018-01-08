from textRegion import get_textRegion
try:
    import Image
except ImportError:
    from PIL import Image,ImageFilter
import pytesseract
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'D:\installed\Tesseract-OCR\tesseract.exe'
# Include the above line, if you don't have tesseract executable in your PATH
# Example tesseract_cmd: 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'


imagePath = r'card2.png'
img = cv2.imread(imagePath)
regions = get_textRegion(img)
count=0
for reg in regions:
    # ret, reg = cv2.threshold(reg, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    i = cv2.cvtColor(reg, cv2.COLOR_BGR2RGB)
    i = Image.fromarray(i, mode="RGB")
    # i = i.resize((400,60))
    # i = i.filter(ImageFilter.SHARPEN)
    # i.show()


    i.save(r'temp/{}.jpg'.format(count))
    count+=1

    print(pytesseract.image_to_string(i,lang='chi_sim+eng'))
    # print(pytesseract.image_to_string(Image.open('test.png')))
