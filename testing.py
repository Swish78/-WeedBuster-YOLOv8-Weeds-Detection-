from PIL import Image
import matplotlib.pyplot as plt

image_path = "/Users/swayampatil/runs/detect/predict2/agri_0_550.jpeg"
image = Image.open(image_path)

plt.imshow(image)
plt.axis('off')
plt.show()

model='/Users/swayampatil/runs/detect/train2/weights/best.pt'