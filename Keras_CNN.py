import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout


# CNN :
classifier = Sequential()

classifier.add(Conv2D(32, (3,3), input_shape = (64,64,1), activation='relu'))

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Conv2D(64, (3,3), input_shape = (64,64,1), activation='relu'))

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Dropout(0.2))

classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = "relu"))

classifier.add(Dense(units = 64, activation = "relu"))

classifier.add(Dense(units = 5, activation = "softmax"))

classifier.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])


# Fitting Images :
from keras.preprocessing.image import ImageDataGenerator

train_data = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

test_data = ImageDataGenerator(rescale = 1./255)

path_train = r"C:\Users\Ronith\Desktop\Data Science 2020\Projects\Hand Sign\Final_Project\Hand Signs\train"

path_test = r"C:\Users\Ronith\Desktop\Data Science 2020\Projects\Hand Sign\Final_Project\Hand Signs\test"

train_set = train_data.flow_from_directory(path_train, target_size = (64,64), color_mode = "grayscale", 
                                              class_mode = "categorical", batch_size = 16)

test_set = test_data.flow_from_directory(path_test, target_size = (64,64), color_mode = "grayscale", 
                                              class_mode = "categorical", batch_size = 16)

classifier.fit_generator(train_set, samples_per_epoch = 1900, nb_epoch = 20, validation_data = test_set, validation_steps = 100)

classifier.save("cnn_adv_01.h5")