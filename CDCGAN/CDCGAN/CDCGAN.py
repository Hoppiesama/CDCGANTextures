import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, time  
import inspect

from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
#from keras_contrib.layers.normalization import InstanceNormalization

from keras import Sequential

from keras import layers, models
from keras import utils as keras_utils

from keras.layers import *
from keras.layers import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D

from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam


#Sets the resolution for the images
#img_shape  = (128, 128, 3)
adam_optimizer = Adam(0.00007, 0.5)
i_batch_size = 20

noise_shape = (100,)

epochs = 50000

img_rows = 128
img_cols = 128
channels = 3
img_shape = (img_rows, img_cols, channels)


train_datagen = ImageDataGenerator(
            rescale=1./255,
            #rescale = (1./127.5) - 1.,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            dtype="float64")

test_datagen = ImageDataGenerator(rescale=1./255)

train_img_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(img_shape[0], img_shape[1]),
        color_mode="rgb",
        batch_size=i_batch_size,
        shuffle=True,
        seed=42,
        class_mode = 'sparse')

test_img_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(img_shape[0], img_shape[1]),
        color_mode="rgb",
        batch_size=5,
        shuffle=True,
        seed=42,
        class_mode = 'sparse')


def get_noise(shape: tuple):
    noise = np.random.uniform(0, 1, size=shape)
    return noise

def generate_noise(nb_of_rows: int):
    #noise = np.random.uniform(0, 1, size=(nb_of_rows, 100))
    noise = np.random.normal(0, 1, size=(nb_of_rows, 100))
    return noise


def make_trainable(net, val):
    """ Freeze or unfreeze layers
    """
    net.trainable = val
    for l in net.layers: l.trainable = val

def plot_generated_images(noise,path_save=None,titleadd="", _epoch = 0):
    r, c = 5, 5

    new_noise = generate_noise(r*c)
    sampled_labels = np.random.randint(0, train_img_generator.num_classes, r * c)

    sampled_labels = keras_utils.to_categorical(sampled_labels, train_img_generator.num_classes)

    gen_imgs = generator.predict([new_noise, sampled_labels])
    
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(r, c, gridspec_kw={'wspace':0, 'hspace':0.5})

    label_dict = train_img_generator.class_indices

    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            label_categories = sampled_labels[cnt]
            label_id = np.where(label_categories == 1.0)[0]
            label_name = list(label_dict.keys())[list(label_dict.values()).index(label_id)]
            axs[i,j].set_title(label_name, {'fontsize': 10})
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(path_save + "image_%d.png" % _epoch, dpi = 200)
    plt.close()
    return


def generator_model():
    initial_width = 4
    initial_height = 4

    # Prepare noise input
    input_z = layers.Input((100,))
    dense_z_1 = layers.Dense(512)(input_z)
    act_z_1 = layers.Activation("tanh")(dense_z_1)
    dense_z_2 = layers.Dense(128 * 8 *  8)(act_z_1)
    bn_z_1 = layers.BatchNormalization()(dense_z_2)
    reshape_z = layers.Reshape((8,  8, 128))(bn_z_1)
    
  #  (img_shape // 8)

    # Prepare Conditional (label) input
    input_c = layers.Input((train_img_generator.num_classes,))
    dense_c_1 = layers.Dense(512)(input_c)
    act_c_1 = layers.Activation("tanh")(dense_c_1)
    dense_c_2 = layers.Dense(8 * 8 * 128)(act_c_1)
    bn_c_1 = layers.BatchNormalization()(dense_c_2)
    reshape_c = layers.Reshape(( 8,  8, 128))(bn_c_1)


    # Combine input source - THIS METHOD POTENTIALLY HELPS PREVENT MODEL COLLAPSE
    # I.e. reshaped layers from both noise and label, making the noise less likely to be ignored as is a problem in CGANs.
    concat_z_c = layers.Concatenate()([reshape_z, reshape_c])

    # Image generation with the concatenated inputs
    up_1 = layers.UpSampling2D(size=(2, 2))(concat_z_c)
    conv_1 = layers.Conv2D(256, (5, 5), padding='same')(up_1)
    act_1 = layers.Activation("tanh")(conv_1)

    up_2 = layers.UpSampling2D(size=(4, 4))(act_1)
    conv_2 = layers.Conv2D(128, (5, 5), padding='same')(up_2)
    act_2 = layers.Activation("tanh")(conv_2)

    up_3 = layers.UpSampling2D(size=(2, 2))(act_2)
    conv_3 = layers.Conv2D(3, (3, 3), padding='same')(up_3)
    act_3 = layers.Activation("tanh")(conv_3)

    model = models.Model(inputs=[input_z, input_c], outputs=act_3)
    return model


def discriminator_model():

    input_gen_image = layers.Input(img_shape)
    conv_1_image = layers.Conv2D(64, (3, 3), padding='same')(input_gen_image)
    act_1_image = layers.Activation("tanh")(conv_1_image)
    pool_1_image = layers.MaxPooling2D(pool_size=(2, 2))(act_1_image)
    conv_2_image = layers.Conv2D(128, (5, 5))(pool_1_image)
    act_2_image = layers.Activation("tanh")(conv_2_image)
    pool_2_image = layers.MaxPooling2D(pool_size=(2, 2))(act_2_image)
    
    conv_3_image = layers.Conv2D(128, (5, 5))(pool_2_image)
    act_3_image = layers.Activation("tanh")(conv_3_image)
    pool_3_image = layers.MaxPooling2D(pool_size=(2, 2))(act_3_image)

    input_c = layers.Input((train_img_generator.num_classes,))
    dense_1_c = layers.Dense(512)(input_c)
    act_1_c = layers.Activation("tanh")(dense_1_c)
    dense_2_c = layers.Dense(13 * 13 * 128)(act_1_c)
    bn_c = layers.BatchNormalization()(dense_2_c)
    reshaped_c = layers.Reshape((13, 13, 128))(bn_c)

    concat = layers.Concatenate()([pool_3_image, reshaped_c])

    flat = layers.Flatten()(concat)
    dense_1 = layers.Dense(1024)(flat)
    act_1 = layers.Activation("tanh")(dense_1)
    dense_2 = layers.Dense(1)(act_1)
    act_2 = layers.Activation('sigmoid')(dense_2)
    model = models.Model(inputs=[input_gen_image, input_c], outputs=act_2)
    return model


def generator_containing_discriminator(g, d):
    input_z = layers.Input((100,))
    input_c = layers.Input((train_img_generator.num_classes,))
    gen_image = g([input_z, input_c])
    d.trainable = False
    is_real = d([gen_image, input_c])
    model = models.Model(inputs=[input_z, input_c], outputs=is_real)
    return model

# Create the models

print("Generator:")
generator = generator_model()
generator.summary()

print("Discriminator:")
discriminator = discriminator_model()
discriminator.summary()

print("Combined:")
combined = generator_containing_discriminator(generator, discriminator)
combined.summary()

generator.compile(loss='binary_crossentropy', optimizer=adam_optimizer)
combined.compile(loss='binary_crossentropy', optimizer=adam_optimizer)
discriminator.trainable = True
discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])

def train(models, noise_plot, dir_result="/result/", epochs=10000):

        train_batch_size = train_img_generator.batch_size
        history = []

        epochs +=1

        half_epochs = int(epochs / 2)
        if (os.path.exists("./Generators") == False):
                    directory = os.makedirs("./Generators")
        
        for epoch in range(epochs):

            #Ready the next batch of images, labels and noise.
            batch = train_img_generator.next()
            image_batch = batch[0]
            current_batch_size = len(image_batch)
            labels = batch[1]
            noise_ = generate_noise(current_batch_size)

            #Reformat image data to between -1 and 1 for tanh
            image_batch = (image_batch * 2) - 1

            valid = np.zeros((current_batch_size, 1))
            fake = np.ones((current_batch_size, 1))

            label_batch = keras_utils.to_categorical(labels, train_img_generator.num_classes)
            gen_imgs = generator.predict([noise_, label_batch], verbose=0)
            
            #Train the discriminator with real then fake images.
            d_loss_real = discriminator.train_on_batch([image_batch, label_batch], valid)
            d_loss_fake = discriminator.train_on_batch([gen_imgs, label_batch], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            
            #Train the generator as a combined model.
            noise = generate_noise(current_batch_size)

            #Prevent discriminator training.
            make_trainable(discriminator, False)
            g_loss = combined.train_on_batch([noise, label_batch], [0] * current_batch_size)
            #Enable discriminator training.
            make_trainable(discriminator, True)

            #print the epoch progress to supervise for model collapse, or overpowering discriminators/generators.
            print ("Epoch:%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            history.append({"D":d_loss,"G":g_loss})
            
            if epoch % 100 == 0:
                plot_generated_images(noise,
                            path_save=dir_result,
                            titleadd="Epoch {}".format(epoch), _epoch=epoch)
            
            if (epoch >= (half_epochs/2)) and (epoch % 1000 == 0):
                #Save model
                if (os.path.exists("./Generators" + str(epoch) ) == False):
                    os.makedirs("./Generators/" + str(epoch) )

                directory = "./Generators/" + str(epoch)

                generator.save(directory + "/generator.h5")
                print("Saved model to disk")

        return(history)

image_output_directory="./training_output_images/"

try:
    os.mkdir(dir_result)
except:
    pass
    
start_time = time.time()

_models = combined, discriminator, generator          

history = train(_models, noise, dir_result=image_output_directory,epochs=40000)
end_time = time.time()
print("-"*10)
print("Time took: {:4.2f} min".format((end_time - start_time)/60))


