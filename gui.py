import sys 
from image_processing import ProcesssedImage
from tkinter import Tk, Label, Button, Canvas,Frame, NW, Toplevel, IntVar, Scale, Entry,HORIZONTAL, RIGHT,LEFT,simpledialog
from PIL import ImageTk
import matplotlib.pyplot as plt
import numpy as np
class ImageEditorGUI:
    def __init__(self, master, filename):
        master.title("Editor de Imagens")
        self.master = master
        
        self.rightframe = Frame(root)
        self.rightframe.pack( side = RIGHT )
        self.leftframe = Frame(root)
        self.leftframe.pack( side = LEFT )
        self.left_right_frame = Frame(self.leftframe)
        self.left_right_frame.pack( side = RIGHT )

        self.workspace_image = ProcesssedImage(filename)

        self.label_original = Label(self.leftframe, text="Original Image")
        self.label_original.pack()
        self.update_image(update_original=True)

        self.label = Label(self.left_right_frame, text="Processed Image")
        self.label.pack()
        self.update_image()

        self.yflip_button = Button(self.rightframe, text="Horizontal Flip", command=self.yflip)
        self.yflip_button.pack()

        self.xflip_button = Button(self.rightframe, text="Vertical Flip", command=self.xflip)
        self.xflip_button.pack()

        self.xflip_button = Button(self.rightframe, text="Rotate 90º", command=self.rotate)
        self.xflip_button.pack()

        self.xflip_button = Button(self.rightframe, text="Greyscale", command=self.grey_scale)
        self.xflip_button.pack()

        self.int_var = IntVar()
        self.awnser = Entry(self.rightframe,width=5)
        self.awnser.pack()

        self.reset_button = Button(self.rightframe, text="Brightness", command=self.brightness)
        self.reset_button.pack()

        self.contrast_button = Button(self.rightframe, text="Contrast", command=self.contrast)
        self.contrast_button.pack()

        self.equalize_button = Button(self.rightframe, text="Equalize", command=self.equalize)
        self.equalize_button.pack()

        self.negative_button = Button(self.rightframe, text="Negative", command=self.negative)
        self.negative_button.pack()

        self.var = IntVar()
        self.scale = Scale(self.rightframe, variable = self.var,to=255, orient=HORIZONTAL)
        self.scale.pack()

        self.quantize_button = Button(self.rightframe, text="Quantize", command=self.quantize_tones)
        self.quantize_button.pack()

        self.histogram_button = Button(self.rightframe, text="Histogram", command=self.histogram)
        self.histogram_button.pack()

        self.reset_button = Button(self.rightframe, text="Reset", command=self.reset)
        self.reset_button.pack()

        self.close_button = Button(self.rightframe, text="Close", command=master.quit)
        self.close_button.pack()

        self.save_button = Button(self.rightframe, text="Save", command=self.save)
        self.save_button.pack()

        self.zoomout_button = Button(self.rightframe, text="Zoom Out", command=self.zoom_out)
        self.zoomout_button.pack()

        self.zoomin_button = Button(self.rightframe, text="Zoom In", command=self.zoom_in)
        self.zoomin_button.pack()

        self.conv_button = Button(self.rightframe, text="Convolution", command=self.conv)
        self.conv_button.pack()

        self.reset_button = Button(self.rightframe, text="Histogram Matching", command=self.matchhistogram)
        self.reset_button.pack()
        self.reset_button = Button(self.rightframe, text="Target Histogram", command=self.histogram_target)
        self.reset_button.pack()
        self.close_button = Button(self.rightframe, text="Load Target", command=self.loadtarget)
        self.close_button.pack()


    def yflip(self):
        self.workspace_image.flip()
        self.update_image()

    def xflip(self):
        self.workspace_image.flip(vertical_axis=True)
        self.update_image()

    def rotate(self):
        self.workspace_image.rotate_90()
        self.update_image()

    def grey_scale(self):
        self.workspace_image.grey_scale()
        self.update_image()

    def histogram(self):
        hist_red, hist_green, hist_blue = self.workspace_image.get_histogram(normalized=True)
        plt.bar(list(range(1,257)),hist_red)
        plt.show()

    def histogram_target(self):
        hist_red, hist_green, hist_blue = self.workspace_image.get_histogram(target=True, normalized=True)
        plt.bar(list(range(1,257)), hist_red,)
        plt.show()

    def loadtarget(self):
        self.workspace_image.load_target("image_editor/images/Gramado_22k.jpg")

    def matchhistogram(self):
        self.workspace_image.match_histogram()
        self.update_image()

    def negative(self):
        self.workspace_image.negative()
        self.update_image()

    def brightness(self):
        self.workspace_image.brightness(int(self.awnser.get()))
        self.update_image()

    def equalize(self):
        self.workspace_image.equalize()
        self.update_image()

    def contrast(self):
        self.workspace_image.contrast(float(self.awnser.get()))
        self.update_image()

    def negative(self):
        self.workspace_image.negative()
        self.update_image()

    def quantize_tones(self):
        self.workspace_image.quantize_tones(max_tones=self.var.get())
        self.update_image()

    def reset(self):
        self.processed_image = ImageTk.PhotoImage(image=self.workspace_image.original_image)
        self.label.config(image=self.processed_image)
        self.master.update_idletasks()
        self.label.photo = self.processed_image

    def zoom_out(self):
        answer1 = simpledialog.askinteger("Input", "Sx",
                                 parent=root,
                                 minvalue=0, maxvalue=100)

        answer2 = simpledialog.askinteger("Input", "Sy",
                                 parent=root,
                                 minvalue=0, maxvalue=100)

        self.workspace_image.zoom_out(answer1,answer2)
        self.update_image()

    def conv(self):
        answconv = simpledialog.askstring("Input", "3x3 filter separated by comma",
                                 parent=root)

        values = answconv.split(',')
        values = [float(x) for x in values]

        kernel_filter = np.array(values).reshape(3,3)

        self.workspace_image.conv_filter(kernel_filter)
        self.update_image()

    def zoom_in(self):
        self.workspace_image.zoom_in()
        self.update_image()

    def save(self):
        self.workspace_image.save_image()

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

    


try:
    filename =  str(sys.argv[1])
    print("Opening:",filename) 
except:
    filename = "image_editor/images/Gramado_22k.jpg"

root = Tk()
gui = ImageEditorGUI(root, filename)
root.mainloop()