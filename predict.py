import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers

import argparse

import math
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2

import os
import shutil
import gc

from tensorflow.keras.layers import ConvLSTM2D, Conv3D, Concatenate, BatchNormalization, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from PIL import Image

#return (num_video, num_frame, 224, 224, 3)
def load_images(path, seqlen=10):

    video_path_list = glob.glob(path)

    return_images = []

    min_frames = 10000

    for path in video_path_list:
        images_path = glob.glob(path + "/*")
        if min_frames > len(images_path):
            min_frames = len(images_path)
        del images_path
        gc.collect()

    print(str(min_frames))

    for path in video_path_list:
        images_path = glob.glob(path + "/*")
        image_list_list = []
        for i in range(seqlen):
            image_list_list.append([])

        print(image_list_list)
        
        for i in range(min_frames):
            image_path = images_path[i]
            image = cv2.imread(image_path)
            resized_image = cv2.resize(image, (224, 224))
            if i < seqlen-1 :
                for l in range(i):
                    print(l)
                    image_list_list[l].append(resized_image)
            else :
                for l in range(len(image_list_list)):
                    image_list_list[l].append(resized_image)
                    if len(image_list_list[l]) % seqlen == 0 and len(image_list_list[l]) != 0:
                        return_images.append(np.array(image_list_list[l]))
                        print(np.array(image_list_list[l]).shape)
                        image_list_list[l] = []
            del image
            del resized_image
            gc.collect()
        del image_list_list
        gc.collect()

    return np.array(return_images)/255.

#バッチサイズ1にしないとエラー出るからあんま意味ねえな
def image_generator(dataset,batchsize=4,seqlen=4, num_frames=20):
    while True:
      batch_x = np.zeros((batchsize,seqlen-1,224,224,3))
      batch_y = np.zeros((batchsize,1,224,224,3))
      ran = np.random.randint(dataset.shape[0],size=batchsize)
      minibatch = dataset[ran]
      #these sequences have length 20; we reduce them to seqlen
      for i in range(batchsize):
          start = 0
          end = start+seqlen-1
          batch_x[i] = minibatch[i,start:end,:,:,:]
          batch_y[i] = minibatch[i,end:end+1,:,:,:]
          yield(batch_x,batch_y)

def save_list(images, num): #takes in input a list of images and plot them
    size = len(images)
    for i in range(size):
        image = cv2.cvtColor((images[i]*255).astype('uint8'), cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image)
        if i == size-1:
            file_name = 'log/' + str(num) + '_predict_scene.png'
            pil_img.save(file_name)
        elif i == size-2:
            file_name = 'log/' + str(num) + '_answer_scene.png'
            pil_img.save(file_name)



def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('--root_dir',default="Data", type=str)
    parser.add_argument('--load',default="load", type=str)
    parser.add_argument('--seqlen',default=4, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    train_path = args.root_dir +"/*"
    seqlen = args.seqlen
    load_file = args.load
    dataset = load_images(train_path, seqlen=seqlen)
    
    video_num = dataset.shape[0]
    print("dataset_num:" + str(video_num))

    print(dataset.shape)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    #バッチサイズをi以外にするとエラーが出ます
    batch_size = 1
    data_gen = image_generator(dataset,seqlen=seqlen, batchsize=batch_size,num_frames = dataset.shape[1])

    # Construct the input layer with size (batch_size=None, num_frames=3,width=64,height=64,channels=1).
    inp = layers.Input(shape=(seqlen-1, *dataset.shape[2:]))

    # We will construct 3 ConvLSTM2D layers with batch normalization,
    # followed by a Conv3D layer for output.
    x = ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(
        filters=64,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=False,
        #return_sequences=True,
        activation="relu",
    )(x)


    #add
    x = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(x)

    out = Conv3D(
        filters=3, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
    )(x)


    # Next, we will build the complete model and compile it.
    model = Model(inp, out)

    model.compile(
        loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(0.001), metrics=[tf.keras.metrics.MeanSquaredError()]
    )

    model.summary()

    model.load_weights(load_file)

    #Model is evaluated using the test image generator
    evaluation = model.evaluate(
        data_gen,
        verbose=1,
        steps= dataset.shape[0] // batch_size
    )

    mse = evaluation[model.metrics_names.index('mean_squared_error')]
    print('Mse of the model is ' + str(mse))

    shutil.rmtree('log')
    os.mkdir('log')

    mse_list=[]

    for k in range(dataset.shape[0]):
        test_x, test_y = next(data_gen)
        y_predicted = model.predict(test_x)
        all = [test_x[0,i,:,:,:] for i in range(seqlen-1)]+[test_y[0,0,:,:,:]]+[y_predicted[0,-1,:,:,:]]
        save_list(all,k)
        mse, se = cv2.quality.QualityMSE_compute(test_y[0,0,:,:,:], y_predicted[0,-1,:,:,:])
        print("mse:")
        mse_mean = (mse[0] + mse[1] + mse[2]) /3
        print(mse_mean)
        mse_list.append(mse_mean)

    print("max:" + str(max(mse_list)))
    print("max_index:" + str(mse_list.index(max(mse_list))))
    print("min:" + str(min(mse_list)))
    print("ave:" + str(np.mean(mse_list)))

    x = list(range(len(mse_list)))
    plt.plot(x, mse_list)
    plt.savefig('mse.png')
