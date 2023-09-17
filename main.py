import digit
import symbol
import cv2
import openAI
from PIL import Image

#digit.create_model()
#symbol.create_model_symbol()

img = cv2.imread('C:\\Users\\spwal\\Downloads\\equ2.png')

s1 = str(digit.predictDigit(img[:,0:img.shape[1]//3]))
s1 += symbol.predictSymbol(img[:,img.shape[1]//3:2*img.shape[1]//3])
s1 += str(digit.predictDigit(img[:,2*img.shape[1]//3:img.shape[1]]))

print(s1)
#print(openAI.entry(s1))
