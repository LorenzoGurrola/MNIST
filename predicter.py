import tkinter as tk
from tkinter import *
import cv2
import PIL.Image
import PIL.ImageDraw
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model('mnist_model.h5')


def predict_digit(img):
    # Resize image to 28x28 pixels
    img = img.resize((28, 28))
    # Convert to grayscale
    img = img.convert('L')
    img = np.array(img)
    # Reshape to form a batch of size 1
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0  # Normalize to [0, 1]
    # Predicting the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0

        # Creating elements
        self.canvas = tk.Canvas(self, width=280, height=280, bg='white')
        self.label = tk.Label(self, text="Thinking...", font=("Helvetica", 48))
        self.classify_btn = tk.Button(
            self, text="Recognise", command=self.classify_handwriting)
        self.button_clear = tk.Button(
            self, text="Clear", command=self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        # Canvas setup
        self.canvas.bind("<B1-Motion>", self.draw_lines)

        self.image = PIL.Image.new("RGB", (280, 280), (255, 255, 255))
        self.draw = PIL.ImageDraw.Draw(self.image)

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(
            self.x-r, self.y-r, self.x+r, self.y+r, fill='black')
        self.draw.ellipse(
            [self.x-r, self.y-r, self.x+r, self.y+r], fill='black')

    def classify_handwriting(self):
        digit, acc = predict_digit(self.image)
        self.label.configure(text=str(digit)+', ' + str(int(acc*100))+'%')

    def clear_all(self):
        self.canvas.delete("all")
        self.image = PIL.Image.new("RGB", (280, 280), white)
        self.draw = PIL.ImageDraw.Draw(self.image)
        self.label.config(text="Draw...")


if __name__ == "__main__":
    app = App()
    mainloop()
