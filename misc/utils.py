'''
This file contains functions and classes which might result useful in multiple code locations.

Classes in this file:
    Logger --> creates an object to concurrently write to a file and to the terminal
    CAMVisualizer --> creates an object to display an image along with its related Class Activation Maps (CAMs)

Functions in this file:
    onehot() --> computes a one-hot encoding for a subset of integers, given the complete list of integers
    build_command() --> builds a command for launching scripts given a command and a parameter dictionary
'''

import numpy as np
import os
import random
import sys
import telegram
from PIL import Image
from matplotlib import pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image
from ipywidgets import Button, Output, HBox, VBox, Label, BoundedIntText
from IPython.display import display


class Logger(object):
    '''
    An object that can write at the same time on the terminal and on a specific file, 
    which should be passed as input during initialization. 
    '''
    def __init__(self, outfile):
        '''
        Constructor class. Also sets stdout to coincide with the logger. 
        All following print statements will print on the terminal and on the selected output file.

        Parameters
        ----------
        outfile : str
            The file on which to write upon future function calls on the object.
        '''
        self.terminal = sys.stdout
        self.log = open(outfile, "w")
        sys.stdout = self

    def write(self, message):
        '''
        Function to write a message on the terminal and on the log file defined at construction time.

        Parameters
        ----------
        message : str
            The message to write on both output channels.
        '''
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        '''
        Function to force flushing on the terminal. [In python, stdout is buffered and would wait for some text before printing.]
        '''
        self.terminal.flush()


class CAMVisualizer:
    '''
    A widget to display an image along with CAMs computed for it. It allows the user to select the width of the image grid to display
    with the parameter plot_cols. For visualization purposes, it requires the list of names of the categories for which CAMs were computed
    and the directories containing CAMs and predictions (the latter is optional).
    It assumes that, in the directories passed as parameters, CAMs and predictions have the same filename (with npy extension) of the image
    they are computed for.
    The widget allows to iterate over a list of images by means of three buttons (Previous, Next and Random) and a Text field.

    Parameters
    ----------
    images : [str]
        List of image paths.
    cat_names : [str]
        List of names of the categories for which CAMs were computed.
    cams_dir : str
        Path to the directory containing CAM files associated to the images passed in the first parameter. 
    preds_dir : str
        Path to the directory containing prediction files associated to the images passed in the first parameter. 
    plot_cols : int
        Number of columns of the grid of images to display
    '''
    def __init__(self, images, cat_names, cams_dir, preds_dir=None, plot_cols=3):
        # Sanity check on passed list of images. If no image is passed, raise exception
        if len(images) == 0:
            raise Exception("No images provided")
        # Sanity check on dir paths passed as parameters: check all CAMs and, in case, predictions exist
        # Also save CAM names in a dedicated variable [saved in the same order as images, corresponding indices]
        self.cam_names = []
        for path in images:
            img_name = path.split('/')[-1]
            cam_name = '.'.join(img_name.split('.')[:-1]) + '.npy'
            assert os.path.isfile(os.path.join(cams_dir, cam_name)), f'Could not locate CAM file for image {img_name}'
            if preds_dir is not None:
                assert os.path.isfile(os.path.join(preds_dir, cam_name)), f'Could not locate prediction for image {img_name}'
            self.cam_names.append(cam_name)
        # Save parameters into instance state
        self.images = images
        self.cat_names = cat_names
        self.cams_dir = cams_dir
        self.preds_dir = preds_dir
        self.plot_cols = plot_cols
        self.max_pos = len(self.images) - 1
        self.pos = 0
        # Create widgets
        # BUTTONS
        self.previous_button = self.__create_button("Previous", (self.pos == 0), self.__on_previous_clicked)
        self.next_button = self.__create_button("Next", (self.pos == self.max_pos), self.__on_next_clicked)
        self.random_button = self.__create_button("Random", False, self.__on_random_clicked)
        # TEXT FIELD WITH IMAGE INDEX
        label_total = Label(value='/ {}'.format(len(self.images)))
        self.text_index = BoundedIntText(value=1, min=1, max=len(self.images))
        self.text_index.layout.width = '80px'
        self.text_index.layout.height = '35px'
        self.text_index.observe(self.__selected_index)
        # MAIN OUTPUT
        self.out = Output()
        # Collect widgets for display
        self.all_widgets = VBox(children=[HBox([self.previous_button, 
                                                self.next_button, 
                                                self.random_button, 
                                                self.text_index,
                                                label_total]), 
                                          self.out])

    def __create_button(self, description, disabled, function):
        '''
        Helper function to create a Button widget and to assign it some basic features.

        Parameters
        ----------
        description : str
            The text to display on the button.
        disabled : bool
            The initial disabled status for the button.
        function : function
            The function to call upon pressing the button. 
        '''
        button = Button(description=description)
        button.disabled = disabled
        button.on_click(function)
        return button

    def __show_cam(self, index):
        '''
        Main function to create and display the grid of images. 
        Receives as input the index of the image to build the grid for.
        '''
        # Load image, CAM and, if required, prediction
        img_path = self.images[index]
        img = Image.open(img_path)
        cam = np.load(os.path.join(self.cams_dir, self.cam_names[index]))
        rgb_img = np.asarray(img.resize((cam.shape[1], cam.shape[2])), dtype=np.float32)/255
        if self.preds_dir is not None: 
            pred = np.load(os.path.join(self.preds_dir, self.cam_names[index]))
        else:
            pred = None
        # Compose grid and plot
        nplots = len(self.cat_names) + 1
        ncols = self.plot_cols
        nrows = max(np.ceil(nplots/ncols).astype(int),2) # always create at least 2 rows to allow indexing axes as grid
        _, axes = plt.subplots(nrows, ncols, figsize=(ncols*6, nrows*6))
        plt.tight_layout()
        # Plot image
        axes[0][0].axis('off')
        axes[0][0].imshow(img)
        axes[0][0].set_title(img_path.split('/')[-1], fontsize=15)
        # Plot CAMs
        for i,cat_name in enumerate(self.cat_names):
            plotidx = i+1
            ridx = plotidx//ncols
            cidx = plotidx%ncols
            axes[ridx][cidx].axis('off')
            axes[ridx][cidx].set_title(cat_name + ("" if pred is None else f", score: {pred[i]:.4f}"), fontsize=15)
            axes[ridx][cidx].imshow(show_cam_on_image(rgb_img, cam[i], use_rgb=True))
        # Remove unused plots
        for i in range(nplots, nrows*ncols):
            ridx = i//ncols
            cidx = i%ncols
            axes[ridx][cidx].remove()
        plt.show();

    def __update_display(self, index):
        '''
        Function to update the current display after clicking a button or changing the value in the text field.
        '''
        self.next_button.disabled = (index == self.max_pos)
        self.previous_button.disabled = (index == 0)

        with self.out:
            self.out.clear_output()
            self.__show_cam(index)

        self.text_index.unobserve(self.__selected_index)
        self.text_index.value = index + 1
        self.text_index.observe(self.__selected_index)

    def __on_previous_clicked(self, b):
        '''
        Callback function for the "Previous" button.
        '''
        self.pos -= 1
        self.__update_display(self.pos)

    def __on_next_clicked(self, b):
        '''
        Callback function for the "Next" button.
        '''
        self.pos += 1
        self.__update_display(self.pos)

    def __on_random_clicked(self, b):
        '''
        Callback function for the "Random" button.
        '''
        self.pos = random.randint(0, self.max_pos)
        self.__update_display(self.pos)

    def __selected_index(self, t):
        '''
        Callback function for updating the input text field. 
        '''
        if t['owner'].value is None or t['name'] != 'value':
            return
        self.pos = t['new'] - 1
        self.__update_display(self.pos)

    def start(self):
        '''
        Function to start the widget and display all buttons and input fields.  
        '''
        if self.max_pos < self.pos:
            print("No available images")
            return
        display(self.all_widgets)
        self.__update_display(self.pos)


def onehot(input, ref):
    '''
    Function to compute a one-hot encoding for a subset of integer numbers given the complete list of integers.
    Throws an exception if an integer in the list to encode is not in the complete list of integers.

    Example: 
        - list of integers to encode: [1, 3, 7]
        - complete list of integers: [1, 2, 3, 4, 5, 7]
        - output: [1, 0, 1, 0, 0, 1]

    Parameters
    ----------
    input : [int]
        The input list of integer to encode.
    ref : [int]
        The reference list of integers to use for encoding.

    Returns
    -------
    code : [int]
        The one-hot encoding of the input integer list.
    '''
    # Check all elements in the array to encode are in complete list
    if not set(input) <= set(ref):
        raise ValueError("Input list to encode is not entirely contained in reference list.")
    # Compute and return encoding
    code = [(ref[i] in input)*1 for i in range(len(ref))]
    return code

def build_command(command, params):
    '''
    Function to create a command for the scripts in this folder. Parameters are assumed to be passed in a specific dictionary wh

    Parameters
    ----------
    command : str
        The main command to launch, followed by positional arguments such as the name of the script to launch.
    params : dict
        A dictionary of parameters. Parameters are assumed to be name in the command as --<param-name> and to be present in the 
        dictionary with the key <param> storing the parameter value.

    Returns
    -------
    command : str
        The output command, ready to be launched.     
    '''
    # Iterate on parameters in the given dictionary
    for key in params:
        if params[key] != None:
            # Specific format for bool parameters
            if isinstance(params[key], bool):
                if params[key]:
                    command += f" --{key}"
            # Specific format for list parameters
            elif isinstance(params[key], list):
                command += f" --{key}"
                for value in params[key]:
                    command += f" {value}"
            # Other parameters
            else:
                command += f' --{key} {params[key]}'
    return command