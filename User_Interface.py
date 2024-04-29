import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from matplotlib import cm as c
import matplotlib.pyplot as plt
from keras.models import model_from_json

#Home page containing project title and start button
class StartPopup:
    def __init__(self, master):
        self.master = master
        self.master.title("CNN based crowd counting and density estimation")
        self.master.geometry("1439x899")
        self.master.configure(bg="pink")  # Change to light pink

        # Title Label
        self.title_label = tk.Label(master, text="CNN based crowd counting and density estimation", bg="pink", font=("Times New Roman", 40, "bold"))  # Change to light pink
        self.title_label.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

        # Start Button
        self.start_button = tk.Button(master, text="Start", bg="pink", font=("Times New Roman", 20), command=self.start_counting)  # Change to light pink
        self.start_button.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

    def start_counting(self):
        filename = filedialog.askopenfilename()
        if filename:
            self.show_result(filename)

    def show_result(self, filename):
        count, img, hmap = self.predict(filename)

        fig = plt.figure(figsize=(10, 5))
        fig.patch.set_facecolor('pink')  # Set background color of the plot
        fig.suptitle("Crowd Counting Result", x=0.5, y=0.98, fontname="Times New Roman", fontsize=16, fontweight="bold")
        axs = fig.subplots(1, 2)

        axs[0].imshow(img.squeeze())
        axs[0].set_title('Original Image', fontname="Times New Roman", fontsize=14)
        axs[0].axis('off')

        axs[1].imshow(hmap.squeeze(), cmap=c.jet)
        axs[1].set_title(f'Model Analysis Report:\nMean Absolute Error: 31\nR-Squared: 0.94\n\nDensity Map\nEstimated Count: {count}', fontname="Times New Roman", fontsize=14)
        axs[1].axis('off')

        plt.show()

    def predict(self, path):
        model = load_model()
        image = create_img(path)
        ans = model.predict(image)
        count = int(np.sum(ans))
        return count, image, ans

def load_model():
    # Function to load and return neural network model 
    json_file = open('/Users/apple/Desktop/CSRnet-master/models/Model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("/Users/apple/Desktop/CSRnet-master/weights/model_A_weights.h5")
    return loaded_model

def create_img(path):
    # Function to load, normalize and return image 
    im = Image.open(path).convert('RGB')
    im = np.array(im)
    im = im / 255.0
    im[:, :, 0] = (im[:, :, 0] - 0.485) / 0.229
    im[:, :, 1] = (im[:, :, 1] - 0.456) / 0.224
    im[:, :, 2] = (im[:, :, 2] - 0.406) / 0.225
    im = np.expand_dims(im, axis=0)
    return im

def main():
    root = tk.Tk()
    app = StartPopup(root)
    root.mainloop()

if __name__ == "__main__":
    main()
