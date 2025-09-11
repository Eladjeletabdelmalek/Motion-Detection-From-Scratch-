import numpy as np 
import cv2 as cv 

my_image="C:/Users/lapto/OneDrive/Desktop/Abdelmalek/photo_2025-01-10_17-25-35.jpg"
image=cv.imread(my_image)
#image=cv.resize(image,(500,500))
npimage=np.array(image)
print(npimage)

# cv.imshow('Abdelmalek',npimage)
# cv.waitKey(0)
# cv.destroyAllWindows()
