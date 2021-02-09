from image_processing import ProcesssedImage

from tkinter import Tk, Label, Button, Canvas,Frame, NW, Toplevel, IntVar, Scale,HORIZONTAL, RIGHT,LEFT
from PIL import ImageTk

class ImageEditorGUI:
    def __init__(self, master):
        self.master = master
        
        self.rightframe = Frame(root)
        self.rightframe.pack( side = RIGHT )
        self.leftframe = Frame(root)
        self.leftframe.pack( side = LEFT )


        self.workspace_image = ProcesssedImage("images/Gramado_22k.jpg")

        self.label_original = Label(self.leftframe, text="Original Image")
        self.label_original.pack()
        self.update_image(update_original=True)

        self.label = Label(self.leftframe, text="Processed Image")
        self.label.pack()
        self.update_image()




        self.yflip_button = Button(self.rightframe, text="Horizontal Flip", command=self.yflip)
        self.yflip_button.pack()

        self.xflip_button = Button(self.rightframe, text="Vertical Flip", command=self.xflip)
        self.xflip_button.pack()

        self.xflip_button = Button(self.rightframe, text="Greyscale", command=self.grey_scale)
        self.xflip_button.pack()

        self.var = IntVar()
        self.scale = Scale(self.rightframe, variable = self.var,to=255, orient=HORIZONTAL)
        self.scale.pack()

        self.reset_button = Button(self.rightframe, text="Quantize", command=self.quantize_tones)
        self.reset_button.pack()

        self.reset_button = Button(self.rightframe, text="Reset", command=self.reset)
        self.reset_button.pack()

        self.close_button = Button(self.rightframe, text="Close", command=master.quit)
        self.close_button.pack()


    def yflip(self):
        self.workspace_image.flip()
        self.update_image()

    def xflip(self):
        self.workspace_image.flip(vertical_axis=True)
        self.update_image()

    def grey_scale(self):
        self.workspace_image.grey_scale()
        self.update_image()

    def quantize_tones(self):
        self.workspace_image.quantize_tones(max_tones=self.var.get())
        self.update_image()

    def reset(self):
        self.processed_image = ImageTk.PhotoImage(image=self.workspace_image.original_image)
        self.label.config(image=self.processed_image)
        self.master.update_idletasks()
        self.label.photo = self.processed_image


    def update_image(self, update_original: bool = False):
        self.processed_image = ImageTk.PhotoImage(image=self.workspace_image.processed_image)

        if update_original:
            self.label_original.config(image=self.processed_image)
            self.master.update_idletasks()
            self.label_original.originalphoto = self.processed_image
        else:
            self.label.config(image=self.processed_image)
            self.master.update_idletasks()
            self.label.photo = self.processed_image

    


root = Tk()
gui = ImageEditorGUI(root)
root.mainloop()