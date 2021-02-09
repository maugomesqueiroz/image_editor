from math import floor
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

class ProcesssedImage(object):
    '''
    '''
    def __init__(self, path_to_file: str):
            self.original_image = Image.open(path_to_file)
            self.processed_image = self.original_image.copy()

            self.px_data = np.asarray(self.processed_image).copy()

        
    def save_image(self, file_name: str = 'output.jpg'):
        ''' Saves an image

        Keyword Arguments:
        file_name - str representing the filename, defaults to 
                    "output.jpg"
        '''

        file_format = file_name.split('.')[1]
        file_format = file_format.upper()

        return self.processed_image.save(file_name, file_format)

    def flip(self, vertical_axis: bool = False):
        ''' Flips an image in relation to a axis. Axis can be
        vertical or horizontal.

        Keyword Arguments:
        vertical_axis - bool indicatig axis, defaults to False
        '''


        image = self.px_data

        # Swaping axes x and y for iteration
        if vertical_axis:
            image = image.swapaxes(0,1)

        flipped = np.array([np.flip(row, axis=0) for row in image])

        # Reverting back to normal
        if vertical_axis:
            flipped = flipped.swapaxes(0,1)

        self.px_data = np.array(flipped)
        self.update_processed_image()        

    def grey_scale(self):
        ''' Converts the rgb image to grayscale.
        '''

        image = self.px_data
        luminance_values_2D = np.array([self.luminance(row) for row in image]).astype(np.uint8)
        luminance_values_3D = np.repeat(luminance_values_2D[:, :, np.newaxis], 3, axis=2)

        self.px_data = luminance_values_3D
        self.update_processed_image()

    def quantize_tones(self, max_tones: int = 255):
        ''' Performs the quantization of image tones

        Keyword Arguments:
        max_tones - maximum number of tones, defaults to 255.
        '''
        image = self.px_data
        mapped_values_2D = np.array([self.quantize_tone(row, max_tones) for row in image]).astype(np.uint8)
        mapped_values_3D = np.repeat(mapped_values_2D[:, :, np.newaxis], 3, axis=2)

        self.px_data = mapped_values_3D
        self.update_processed_image()

    def update_processed_image(self):
        ''' Takes pixel data from instance and updates the processed image'''
        self.processed_image = Image.fromarray(self.px_data)

    @staticmethod
    def luminance(values: list):
        # for rgb_value in rgb_row:
        luminance_values = [int(0.299*r + 0.587*g + 0.114*b) for r, g, b in list(values)]
        return luminance_values

    @staticmethod
    def quantize_tone(values: list, n_tones: int, max_tone: int = 255) -> list:
        bin_size = int(max_tone/n_tones)
        return [floor(tone/bin_size)*bin_size for tone,*_ in values]