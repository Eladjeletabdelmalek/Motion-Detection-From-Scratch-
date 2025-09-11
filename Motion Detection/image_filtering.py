import numpy as np 
import cv2 as cv 


def grey_scale(image):
    weights = np.array([0.299, 0.587, 0.114])
    npimage = np.array(image, dtype=float)
    gray = np.dot(npimage[..., :3], weights)  
    return gray.astype(np.uint8)




# my_image="C:/Users/lapto/OneDrive/Desktop/Abdelmalek/photo_2025-01-10_17-25-35.jpg"
# image=cv.imread(my_image)
# npimage=grey_scale(image)  
# cv.imshow('Abdelmalek',npimage)
# cv.waitKey(0)
# cv.destroyAllWindows()     