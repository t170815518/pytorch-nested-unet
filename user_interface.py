import tkinter as tk
import customtkinter
from tkinter import filedialog, simpledialog, Label
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
import os
import cv2
from PIL import ImageTk, Image
from val import compute_an_image


customtkinter.set_appearance_mode("dark")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green


def image_inference():
    filename = filedialog.askopenfilename()  # ask the user to choose a file
    if filename:
        fig = compute_an_image(filename)

        segmentation_displayer = customtkinter.CTk()  # create CTk window like you do with the Tk window
        segmentation_displayer.geometry("580x380")
        segmentation_displayer.resizable(False, False)
        segmentation_displayer.title('矿石分割结果')

        canvas = FigureCanvasTkAgg(fig, master=segmentation_displayer)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=None, expand=False)

        segmentation_displayer.mainloop()


root = customtkinter.CTk()  # create CTk window like you do with the Tk window
root.geometry("580x380")
root.resizable(False, False)
root.title('矿石分割')
# root.geometry('500x555')
FONT = ('黑体', 35)
TITLE_FONT = ('黑体', 35, 'bold')

# Load the icons for the buttons
# image_open = Image.open("assets/camera.png").resize([25, 25])
# real_time_icon = customtkinter.CTkImage(image_open)
# image = Image.open("assets/image.png").resize([25, 25])
# image_icon = customtkinter.CTkImage(image)
# image = Image.open("assets/movie.png").resize([25, 25])
# video_icon = customtkinter.CTkImage(image)


title_label = customtkinter.CTkLabel(root, text='矿石分割', font=FONT)
title_label.place(relx=0.5, rely=0.1, anchor='center')

button2 = customtkinter.CTkButton(root, text="图片检测", command=image_inference, compound="left",
                                  font=FONT)
button2.place(relx=0.5, rely=0.5, anchor='center')

root.mainloop()
