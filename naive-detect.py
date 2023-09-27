import cv2
import numpy as np  
image = cv2.imread('./src/demo.jpg')
# A better way to do this is to set color in BGR format.
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
color = np.array([26,230,229])

black = np.array([0,0,0])
threshold = 150

centered = image - color

centered = np.linalg.norm(centered,axis=2)

index = centered < threshold
image[index] = black
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
cv2.imwrite('./src/demo1.jpg',image)
