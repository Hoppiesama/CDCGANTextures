#import CDCGAN as cdcgan
import os
import inspect

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing import image
from keras_preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os, sys



def parse_input(string = ""):
    s = str.lower(string)
    if (s == "exit"):
        sys.exit()
    try:
        return int(string)
    except ValueError:
        return -1


def generate_noise(nb_of_rows: int):
    #noise = np.random.uniform(0, 1, size=(nb_of_rows, 100))
    noise = np.random.normal(0, 1, size=(nb_of_rows, 100))
    return noise

def get_input():
    valid_input = False
    while valid_input == False:
        s = input('Input:')

        temp = parse_input(s)

        if temp == -5:
            import CDCGAN
        elif (temp == -1):
            print("Invalid input! Input type is a whole number (integer)")
        elif (temp < 1) or (temp >3):
            print ("The number must be within the bounds shown above.")
        elif (temp > 0) and (temp < 4):
            valid_input == True
            return temp

def save_images(label):
    r, c = 2, 2

    new_noise = generate_noise(r*c)
    sampled_labels = np.zeros(r*c)

    #Set all the sampled labels to the input label
    for i in range(0,4):
        sampled_labels[i] = label

    #To categorical for use in the generator
    sampled_labels = to_categorical(sampled_labels, 3)
    gen_imgs = generator.predict([new_noise, sampled_labels])

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    #Save each image as it's own individual texture
    for i in range(r*c):
        #im = image.array_to_img(gen_imgs[i],scale=False)
        image.save_img("./GeneratedImages/output_%d.png" %i, gen_imgs[i], data_format="channels_last", scale=True)

    print("%d textures generated and placed into the GeneratedImages folder." %(r*c))
    return



def try_to_load_generator(dir=None):
    if (os.path.exists("./ActiveGenerator/generator.h5" ) == False):
        print("There is no generator.h5 in the 'Generators' folder.")
        print("Would you like to train a new one? (y/n)")
        string = input()
        if (string == str.lower("yes") ) or (string == str.lower("y") ):
            import CDCGAN
        else:
            print("Then this is goodbye!")
            temp = input()
            sys.exit()
    else:
        #load and build model
        generator = load_model(dir, custom_objects=None, compile=False)
        return generator

def run():
    #check

    print ("Welcome to the Texture generator!\n")
    print ("5 textures from the following\noptions will be generated for you\n in the following directory...")
    print (directory)

    print ("1  ->  Bricks")
    print ("2  ->  Grass")
    print ("3  ->  Wood")
    print ("-5  -> train CDCGAN")

    while True:
        variable = get_input()
        save_images(variable-1)


directory = "./ActiveGenerator/generator.h5"
generator = try_to_load_generator(dir=directory)

run()

if (os.path.exists("./ActiveGenerator/generator.h5" ) == False):
    print("An error has occured. A generator of your choice must be placed into the ActiveGenerator folder.")
    if input()=='':
       sys.exit()

