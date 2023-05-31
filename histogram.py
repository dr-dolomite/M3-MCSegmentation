import matplotlib.pyplot as plt

epochs = list(range(1, 11))

# Values for accuracy, dice score, and loss
accuracy = [89.42, 94.87, 94.55, 86.46, 91.17, 94.62, 95.43, 94.40, 95.79, 95.27]
dice_score = [1.7853, 1.8305, 1.8283, 1.7640, 1.7991, 1.8291, 1.8355, 1.8250, 1.8402, 1.8328]
loss = [-262, -321, -322, -327, -320, -320, -325, -324, -329, -344]

# Values for validation mean DSC, IoU, and pixel accuracy
validation_mean_dsc = [0.1287, 0.1280, 0.1247, 0.1291, 0.1280, 0.1302, 0.1315, 0.1280, 0.1313, 0.1275]
validation_mean_iou = [0.7731, 0.7741, 0.5772, 0.6847, 0.7681, 0.7982, 0.7590, 0.8119, 0.7925, 0.7713]
validation_mean_pixel_accuracy = [0.9487, 0.9455, 0.8646, 0.9117, 0.9462, 0.9543, 0.9440, 0.9579, 0.9527, 0.9477]

# Line graph for accuracy, dice score, and loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracy, label='Accuracy')
plt.plot(epochs, dice_score, label='Dice Score')
plt.plot(epochs, loss, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Accuracy, Dice Score, and Loss per Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Line graph for validation mean DSC, IoU, and pixel accuracy
plt.figure(figsize=(10, 6))
plt.plot(epochs, validation_mean_dsc, label='Validation Mean DSC')
plt.plot(epochs, validation_mean_iou, label='Validation Mean IoU')
plt.plot(epochs, validation_mean_pixel_accuracy, label='Validation Mean Pixel Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Validation Mean DSC, IoU, and Pixel Accuracy per Epoch')
plt.legend()
plt.grid(True)
plt.show()
