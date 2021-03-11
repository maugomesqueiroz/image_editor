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

    def brightness(self, brightness: int):
        ''' Adds a brightness to the image
        '''
        image = self.px_data.copy()
        image = image.astype('int')

        red = image[:,:,0]
        green = image[:,:,1]
        blue = image[:,:,2]

        red = red + brightness
        green = green + brightness
        blue = blue + brightness

        image[:,:,0] = red
        image[:,:,1] = green
        image[:,:,2] = blue

        image = np.where( image > 0, image, 0)
        image = np.where( image < 254, image, 254)

        # self.px_data = image
        self.px_data = image.astype(np.uint8)
        self.update_processed_image()

    def contrast(self, contrast: int):
        ''' Adds a brightness to the image
        '''
        image = self.px_data.copy()
        image = image.astype('int')

        red = image[:,:,0]
        green = image[:,:,1]
        blue = image[:,:,2]

        red = red*contrast
        green = green*contrast
        blue = blue*contrast

        image[:,:,0] = red
        image[:,:,1] = green
        image[:,:,2] = blue

        image = np.where( image > 0, image, 0)
        image = np.where( image < 254, image, 254)

        # self.px_data = image
        self.px_data = image.astype(np.uint8)
        self.update_processed_image()

    def equalize(self):
        ''' Adds a brightness to the image
        '''
        
        height = len(self.px_data)
        width = len(self.px_data[0])
        red, *_ = self.get_histogram(grey_scale=True)

        histogram_array = np.array(red)
        cumulative_histogram = histogram_array.cumsum()
        number_of_pixels = width*height
        alpha = 255/number_of_pixels

        cumulative_histogram = (cumulative_histogram*alpha).astype('int')


        image = self.px_data.copy()

        for i in range(height):
            for j in range(width):
                for c in range(3):
                    px_value = image[i,j,c]
                    eq_px_value = cumulative_histogram[px_value]
                    image[i,j,c] = eq_px_value
                    
        # self.px_data = image
        self.px_data = image
        self.update_processed_image()

    def match_histogram(self):
        ''' Adds a brightness to the image
        '''
        

        red, *_ = self.get_histogram(grey_scale=True)
        red_target, *_ = self.get_histogram(grey_scale=True, target=True)

        height = len(self.px_data)
        width = len(self.px_data[0])
        histogram_array = np.array(red)
        cumulative_histogram = histogram_array.cumsum()
        number_of_pixels = width*height
        alpha = 255/number_of_pixels
        cumulative_histogram = (cumulative_histogram*alpha).astype('int')

        height_target = len(self.px_data)
        width_target = len(self.px_data[0])
        histogram_array_target = np.array(red_target)
        cumulative_histogram_target = histogram_array_target.cumsum()
        number_of_pixels_target = width_target*height_target
        alpha_target = 255/number_of_pixels_target
        cumulative_histogram_target = (cumulative_histogram_target*alpha_target).astype('int')

        image = self.px_data.copy()

        dict_tone = {}
        for i in range(256):
            eq_px_value = cumulative_histogram[i]
            differences = [np.abs(eq_px_value-x) for x in cumulative_histogram_target]
            match_px_val = int(np.argmin(differences))
            dict_tone[i] = match_px_val

        print(dict_tone)
        for i in range(height):
            for j in range(width):
                for c in range(3):
                    px_value = image[i,j,c]
                    # eq_px_value = cumulative_histogram[px_value]

                    #where in target eq_px_value is equal?
                    # differences = [np.abs(eq_px_value-x) for x in cumulative_histogram_target]
                    # match_px_val = np.argmin(differences)
                    # image[i,j,c] = match_px_val
                    image[i,j,c] = dict_tone[int(px_value)]
                    
        # self.px_data = image
        self.px_data = image
        self.update_processed_image()

    def save_image(self, file_name: str = 'output.jpg'):
        ''' Saves an image

        Keyword Arguments:
        file_name - str representing the filename, defaults to 
                    "output.jpg"
        '''
        return self.processed_image.save(file_name)

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

    def get_histogram(self, normalized=False, grey_scale=False, target=False):
        ''' 

        Keyword Arguments:
        normalized - .
        '''

        if target:
            gs_px_data = self.grey_scale(return_data=True, target=True)
            image = np.asarray(Image.fromarray(gs_px_data)).copy()
        else:
            image = self.px_data
            if grey_scale:
                gs_px_data = self.grey_scale(return_data=True)
                image = np.asarray(Image.fromarray(gs_px_data)).copy()
            else:
                image = np.asarray(self.processed_image).copy()

        histogram_red = [0]*256
        histogram_green = [0]*256
        histogram_blue = [0]*256

        for row in image:
            for pixel in row:
                histogram_red[pixel[0]] += 1
                histogram_green[pixel[1]] += 1 
                histogram_blue[pixel[2]] += 1
                
        #normalize
        if normalized:
            sum_red = sum(histogram_red)
            sum_blue = sum(histogram_blue)
            sum_green = sum(histogram_green)

            histogram_red =[val/sum_red for val in histogram_red]
            histogram_blue =[val/sum_blue for val in histogram_blue]
            histogram_green =[val/sum_green for val in histogram_green]

        return histogram_red, histogram_green, histogram_blue

    def grey_scale(self, return_data=False, target=False):
        ''' Converts the rgb image to grayscale.
        '''
        if target:
            image = self.target_px_data
        else:
            image = self.px_data

        luminance_values_2D = np.array([self.luminance(row) for row in image]).astype(np.uint8)
        luminance_values_3D = np.repeat(luminance_values_2D[:, :, np.newaxis], 3, axis=2)

        if return_data:
            return luminance_values_3D

        self.px_data = luminance_values_3D
        self.update_processed_image()

    def load_target(self, file_name: str = 'output.jpg'):
        ''' Saves an image

        Keyword Arguments:
        file_name - str representing the filename, defaults to 
                    "output.jpg"
        '''
        self.target_original_image = Image.open(file_name)
        self.target_processed_image = self.target_original_image.copy()
        self.target_px_data = np.asarray(self.target_processed_image).copy()

    def negative(self):
        ''' Adds a brightness to the image
        '''
        image = self.px_data.copy()

        red = image[:,:,0]
        green = image[:,:,1]
        blue = image[:,:,2]

        red = 255-red
        green = 255-green
        blue = 255-blue

        image[:,:,0] = red
        image[:,:,1] = green
        image[:,:,2] = blue

        image = np.where( image > 0, image, 0)
        image = np.where( image < 254, image, 254)

        # self.px_data = image
        self.px_data = image
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

    def zoom_out(self,sx,sy):
        '''Performs zoom out in image
        '''
        img_height = self.px_data.shape[0]
        img_width = self.px_data.shape[1]

        s_width = int(np.ceil(img_width/sx))
        s_height = int(np.ceil(img_height/sy))

        s_matrix = np.zeros(s_width*s_height*3).reshape(s_height,s_width,3)

        for c in range(3):
            for ix in range(s_width):
                for iy in range(s_height):
                    s_matrix[iy,ix,c] = self.px_data[ iy*sy:(iy+1)*sy, ix*sx:(ix+1)*sx, c].mean()

        s_matrix = s_matrix.astype(np.uint8)
        self.px_data = s_matrix
        self.update_processed_image()
        
    def zoom_in(self):
        '''Magnifies 2x the image
        '''
        img_height = self.px_data.shape[0]
        img_width = self.px_data.shape[1]

        zin_width = int(img_width*2)
        zin_height = int(img_height*2)

        zin_matrix = np.zeros(zin_width*zin_height*3).reshape(zin_height,zin_width,3)

        #place original values
        for c in range(3):
            for ix in range(img_width):
                for iy in range(img_height):
                    zin_matrix[iy*2,ix*2,c] = self.px_data[ iy, ix, c ]

        # #interpolate col values
        for c in range(3):
            for ix in range(1,img_width-1):
                for iy in range(1,img_height-1):
                    zin_matrix[iy*2, (ix*2)-1 ,c] = (zin_matrix[ iy*2, (ix*2)-2, c] + zin_matrix[ iy*2, (ix*2), c])/2

        # #interpolate row values
        for c in range(3):
            for ix in range(1,zin_width-1):
                for iy in range(1,img_height-1):
                    zin_matrix[(iy*2)-1, ix ,c] = (zin_matrix[ (iy*2)-2, ix, c] + zin_matrix[ iy*2, ix, c])/2

        zin_matrix = zin_matrix.astype(np.uint8)
        self.px_data = zin_matrix
        self.update_processed_image()

    def rotate_90(self):
        ''' Roatate the image by 90 degrees. 
        '''
        img_height = self.px_data.shape[0]
        img_width = self.px_data.shape[1]

        r_matrix = np.zeros(img_width*img_height*3).reshape(img_width,img_height,3)

        for c in range(3):
            for ix in range(img_width):
                for iy in range(img_height):
                    r_matrix[ix,-iy,c] = self.px_data[iy,ix,c]
        
        self.px_data = r_matrix.astype(np.uint8)
        self.update_processed_image()

    def conv_filter(self, kernel_filter:np.array)->np.array:
        '''Apply convolution filter in the image
        '''
        img_height = self.px_data.shape[0]
        img_width = self.px_data.shape[1]

        filtered_matrix = np.zeros(img_height*img_width*3).reshape(img_height,img_width,3)

        #place original values
        for c in range(3):
            for ix in range(1,img_width-1):
                for iy in range(1,img_height-1):
                    window = (self.px_data[(iy-1):(iy+2),(ix-1):(ix+2),1]).astype(float)
                    filtered_matrix[iy,ix,c] = (window*kernel_filter).sum()

        filtered_matrix += 127
        filtered_matrix = np.where( filtered_matrix > 0, filtered_matrix, 0)
        filtered_matrix = np.where( filtered_matrix < 254, filtered_matrix, 254)

        filtered_matrix = filtered_matrix.astype(np.uint8)
        self.px_data = filtered_matrix
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