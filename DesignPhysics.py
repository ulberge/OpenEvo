import numpy as np
import cv2
import os
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, initializers
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
import math


def load_pretrained(filepath):
    """
    Loads the pretrained weights and biases from the pretrained model available
    on http://www.eecs.qmul.ac.uk/~tmh/downloads.html
    Copied from: https://github.com/ayush29feb/Sketch-A-XNORNet
    Args:
        Takes in the filepath for the pretrained .mat filepath

    Returns:
        Returns the dictionary with all the weights and biases for respective layers
    """
    if filepath is None or not os.path.isfile(filepath):
        print('Pretrained Model Not Available!')
        return None, None

    data = sio.loadmat(filepath)
    weights = {}
    biases = {}
    conv_idxs = [0, 3, 6, 8, 10, 13, 16, 19]
    for i, idx in enumerate(conv_idxs):
        weights['conv' + str(i + 1)] = data['net']['layers'][0][0][0][idx]['filters'][0][0]
        biases['conv' + str(i + 1)] = data['net']['layers'][0][0][0][idx]['biases'][0][0].reshape(-1)

    print('Pretrained Model Loaded!')
    return (weights, biases)


def load_layers():
    """
    Loads the layers for the Sketch-A-Net architecture and populates the
    weights and biases with the pretrained values.
    Inspiration from: https://github.com/ayush29feb/Sketch-A-XNORNet
    Returns:
        Returns a list of the model layers
    """
    weights, biases = load_pretrained('model_without_order_info_224.mat')

    model_layers = []

    # L1
    model_layers.append(layers.Conv2D(
        64,
        (15, 15),
        strides=3,
        padding='valid',
        kernel_initializer=initializers.Constant(weights['conv1']),
        bias_initializer=initializers.Constant(biases['conv1']),
        activation='relu',
        input_shape=(225, 225, 1),
        name='L1conv'
    ))
    model_layers.append(layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=2,
        padding='valid',
        name='Pool1'
    ))

    # L2
    model_layers.append(layers.Conv2D(
        128,
        (5, 5),
        strides=1,
        padding='valid',
        kernel_initializer=initializers.Constant(weights['conv2']),
        bias_initializer=initializers.Constant(biases['conv2']),
        activation='relu',
        name='L2conv'
    ))
    model_layers.append(layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=2,
        padding='valid',
        name='Pool2'
    ))

    # L3
    model_layers.append(layers.Conv2D(
        256,
        (3, 3),
        strides=1,
        padding='same',
        kernel_initializer=initializers.Constant(weights['conv3']),
        bias_initializer=initializers.Constant(biases['conv3']),
        activation='relu',
        name='L3conv'
    ))

    # L4
    model_layers.append(layers.Conv2D(
        256,
        (3, 3),
        strides=1,
        padding='same',
        kernel_initializer=initializers.Constant(weights['conv4']),
        bias_initializer=initializers.Constant(biases['conv4']),
        activation='relu',
        name='L4conv'
    ))

    # L5
    model_layers.append(layers.Conv2D(
        256,
        (3, 3),
        strides=1,
        padding='same',
        kernel_initializer=initializers.Constant(weights['conv5']),
        bias_initializer=initializers.Constant(biases['conv5']),
        activation='relu',
        name='L5conv'
    ))
    model_layers.append(layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=2,
        padding='valid',
        name='Pool5'
    ))

    # L6
    model_layers.append(layers.Conv2D(
        512,
        (7, 7),
        strides=1,
        padding='valid',
        kernel_initializer=initializers.Constant(weights['conv6']),
        bias_initializer=initializers.Constant(biases['conv6']),
        activation='relu',
        name='L6conv'
    ))
    model_layers.append(layers.Dropout(
        rate=0.5,
        name='Dropout8'
    ))

    # L7
    model_layers.append(layers.Conv2D(
        512,
        (1, 1),
        strides=1,
        padding='valid',
        kernel_initializer=initializers.Constant(weights['conv7']),
        bias_initializer=initializers.Constant(biases['conv7']),
        activation='relu',
        name='L7conv'
    ))
    model_layers.append(layers.Dropout(
        rate=0.5,
        name='Dropout7'
    ))

    # L8
    model_layers.append(layers.Conv2D(
        250,
        (1, 1),
        strides=1,
        padding='valid',
        kernel_initializer=initializers.Constant(weights['conv8']),
        bias_initializer=initializers.Constant(biases['conv8']),
        activation='relu',
        name='L8conv'
    ))

    return model_layers

# Helper methods copied from stackoverflow https://stackoverflow.com/questions/54279442/define-vectors-and-find-angles-using-python-in-3d-space
def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    return math.acos(max(-1, min(1, dotproduct(v1, v2) / (length(v1) * length(v2)))))



class Design:
    """
    A class representing a design as movement through space
    """
    def __init__(self, dim=(225, 225)):
        self.dim = dim
        self.designs = []
        self.positions = []
        self.velocity = None
        self.next_position = None

        self.layers = load_layers()

        # init position and velocity by setting two positions at origin
        self.set_design(np.asarray(Image.new('F', self.dim, 1.0)))
        self.set_design(np.asarray(Image.new('F', self.dim, 1.0)))

    def set_design(self, img):
        """
        Sets the current image of the design and calculates its position
        in activation space as well as its velocity and presumed next position.
        Args:
            Takes in the current image of the design
        """
        self.designs.append(img)
        img = self.format(img)
        position = self.get_position(img)
        self.positions.append(position)

        if len(self.positions) > 1:
            self.velocity = self.positions[-1] - self.positions[-2]
            self.next_position = self.positions[-1] + self.velocity

    def get_score(self, variation):
        """
        Calculates a score for the given images position in activation space relative to the trajectory of this design
        Args:
            Takes in an image to score
        Returns:
            A the image's score
        """
        variation = self.format(variation)
        var_position = self.get_position(variation)
        vector_to_var = var_position - self.next_position

        # calculate position difference from "optimal" based on trajectory
        diff_position = length(vector_to_var)

        # calculate angle difference from current trajectory
        diff_angle = angle(vector_to_var, self.velocity)

        score = diff_position * (1 + abs(diff_angle))
        return score

    def get_position(self, img):
        """
        Calculates the given image's position in activation space
        Args:
            Takes in an image to find a position for
        Returns:
            A the image's position
        """
        img = self.format(img)

        dna = []
        # format for tf
        curr = np.array([img, ])
        # execute the layers
        for layer_index, layer in enumerate(self.layers):
            # run layer
            curr = layer(curr)

            # save the activations from convolution layers
            # if layer.name in ['L1conv', 'L2conv', 'L3conv', 'L4conv', 'L5conv', 'L6conv', 'L7conv', 'L8conv']:
            # For now, only use the middle layers
            if layer.name in ['L3conv', 'L4conv', 'L5conv']:
                # compile scores for each filter
                channels, h, w, filter_count = curr.shape
                for f in range(filter_count):
                    sub_solution = curr[0, :, :, f]
                    # scale the value relative to the size of this layer
                    val = np.sum(sub_solution) / (h * w)
                    dna.append(val)

        return np.array(dna)

    def format(self, img):
        """
        Calculates the given image's position in activation space
        Args:
            Takes in an image to properly format
        Returns:
            A copy of the formatted image
        """
        # invert black and white
        img_copy = 1 - img
        # resize into (h, w, 1) shape
        img_copy = cv2.resize(img_copy, self.dim)
        img_copy = np.asarray(img_copy).reshape(self.dim + (1,))
        return img_copy
