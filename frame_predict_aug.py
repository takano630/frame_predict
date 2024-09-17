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
def load_images(path, seqlen=4):

    video_path_list = glob.glob(path)

    return_images = []

    min_frames = 10000
    frame_num = seqlen

    for path in video_path_list:
        images_path = glob.glob(path + "/*")
        if min_frames > len(images_path):
            min_frames = len(images_path)
        del images_path
        gc.collect()

    print(str(min_frames))

    for path in video_path_list:
        images_path = glob.glob(path + "/*")
        image_list = []
        for i in range(1,min_frames):
            image_path = images_path[i]
            image = cv2.imread(image_path)
            resized_image = cv2.resize(image, (224, 224))
            image_list.append(resized_image)
            del image
            gc.collect()
            if i % frame_num == 0 and i !=0:
                return_images.append(np.array(image_list))
                print(np.array(image_list).shape)
                image_list = []

        del image_list
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
    parser.add_argument('--seqlen',default=4, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    train_path = args.root_dir +"/*"
    seqlen = args.seqlen
    dataset = load_images(train_path, seqlen=seqlen)
    
    video_num = dataset.shape[0]
    print("dataset_num:" + str(video_num))
    
    #pythonのintは切り捨て！
    train_num = int(video_num*6/10)
    val_test_num = int(video_num*2/10)
    trainset = dataset[:train_num] 
    valset = dataset[train_num:train_num+val_test_num] 
    testset = dataset[train_num+val_test_num:] 

    print("trainset")
    print(trainset.shape)
    print("valset")
    print(valset.shape)
    print("testset")
    print(testset.shape)

    del dataset
    gc.collect()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    #バッチサイズをi以外にするとエラーが出ます
    batch_size = 1
    train_gen = image_generator(trainset,seqlen=seqlen, batchsize=batch_size,num_frames = trainset.shape[1] )
    val_gen = image_generator(valset,seqlen=seqlen, batchsize=batch_size,num_frames = valset.shape[1])
    test_gen = image_generator(testset,seqlen=seqlen, batchsize=batch_size,num_frames = testset.shape[1])

    # Construct the input layer with size (batch_size=None, num_frames=3,width=64,height=64,channels=1).
    inp = layers.Input(shape=(seqlen-1, *trainset.shape[2:]))

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

    # Define some callbacks to improve training.
    '''
    #EarlyStopping helps to stop training to avoid overfitting
    early_stopping = EarlyStopping(monitor="val_mean_squared_error", patience=10, min_delta=0.001)
    #ReduceLROnPlateau reduces learning rate in case val_mean_squared_error is not improving
    reduce_lr = ReduceLROnPlateau(monitor='val_mean_squared_error', factor=0.5,
                                patience=3, min_lr=0.0001)
    '''
    
    #EarlyStopping helps to stop training to avoid overfitting
    early_stopping = EarlyStopping(monitor="val_mean_squared_error", patience=10, min_delta=0.001)
    #ReduceLROnPlateau reduces learning rate in case val_mean_squared_error is not improving
    reduce_lr = ReduceLROnPlateau(monitor='val_mean_squared_error', factor=0.5,
                                patience=3, min_lr=0.0001)
    

    # Define modifiable training hyperparameters.
    epochs = 50
    steps_per_epoch = trainset.shape[0] // batch_size
    validation_steps = valset.shape[0] // batch_size

    
    checkpoint_filepath = './model1_weights1.h5'

    #Save best model weights
    model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    #monitor='val_loss',
    monitor='val_mean_squared_error',
    save_best_only=True,
    verbose=1)



    # Fit the model to the training data.
    model.fit(
        train_gen,
        steps_per_epoch= steps_per_epoch,
        epochs=epochs,
        validation_data= val_gen,
        validation_steps= validation_steps,
        #callbacks=[early_stopping, reduce_lr],
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )

    model.load_weights('model1_weights1.h5')

    #Model is evaluated using the test image generator
    evaluation = model.evaluate(
        test_gen,
        verbose=1,
        steps= testset.shape[0] // batch_size
    )

    mse = evaluation[model.metrics_names.index('mean_squared_error')]
    print('Mse of the model is ' + str(mse))

    shutil.rmtree('log')
    os.mkdir('log')

    for k in range(testset.shape[0]):
        test_x, test_y = next(test_gen)
        y_predicted = model.predict(test_x)
        all = [test_x[0,i,:,:,:] for i in range(seqlen-1)]+[test_y[0,0,:,:,:]]+[y_predicted[0,-1,:,:,:]]
        save_list(all,k)


