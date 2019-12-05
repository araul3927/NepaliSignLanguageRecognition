# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam

# Step 1 - Building the CNN

# Initializing the CNN
classifier = Sequential()

# First convolution layer and pooling
classifier.add(Convolution2D(filters=32,kernel_size=(3, 3), padding="same",input_shape=(200, 200,1), activation='relu'))
# classifier.add(Convolution2D(filters=32, kernel_size=(3, 3),padding="same",activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# classifier.add(Dropout(0.25))

# Second convolution layer and pooling
classifier.add(Convolution2D(filters=64, kernel_size=(3, 3), padding="same",activation='relu'))
# classifier.add(Convolution2D(filters=64, kernel_size=(3, 3), padding="same",activation='relu'))
# classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# classifier.add(Dropout(0.25))

#Adding 3rd Concolution Layer
classifier.add(Convolution2D(filters=128, kernel_size=(3,3), padding="same",activation = 'relu'))
# classifier.add(Convolution2D(filters=128, kernel_size=(3,3), padding="same",activation = 'relu'))
# classifier.add(Convolution2D(filters=128, kernel_size=(3,3), padding="same",activation = 'relu'))
# classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size =(2,2), strides=(2, 2)))
# classifier.add(Dropout(0.25))

#Adding 4rd Concolution Layer
classifier.add(Convolution2D(filters=256, kernel_size=(3,3), padding="same",activation = 'relu'))
# classifier.add(Convolution2D(filters=256, kernel_size=(3,3), padding="same",activation = 'relu'))
# classifier.add(Convolution2D(filters=256, kernel_size=(3,3), padding="same",activation = 'relu'))
# classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size =(2,2), strides=(2, 2)))
# classifier.add(Dropout(0.25))

#Adding 5th layer
# classifier.add(Convolution2D(filters=256, kernel_size=(3,3), padding="same",activation = 'relu'))
# classifier.add(Convolution2D(filters=256, kernel_size=(3,3), padding="same",activation = 'relu'))
# classifier.add(Convolution2D(filters=256, kernel_size=(3,3), padding="same",activation = 'relu'))
# classifier.add(BatchNormalization())
# classifier.add(MaxPooling2D(pool_size =(2,2), strides=(2, 2)))
# classifier.add(Dropout(0.25))

# Flattening the layers
classifier.add(Flatten())

# Adding a fully connected layer
classifier.add(Dense(units=1024, activation='relu'))
# classifier.add(Dense(units=512, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=36, activation='softmax')) # softmax for more than 2

# Compiling the CNN
classifier.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2


# Step 2 - Preparing the train/test data and training the model

# Code copied from - https://keras.io/preprocessing/image/
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/train',
                                                 target_size=(200, 200),
                                                 batch_size=32,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('dataset/test',
                                            target_size=(200, 200),
                                            batch_size=32,
                                            color_mode='grayscale',
                                            class_mode='categorical')
model=classifier.fit_generator(
        training_set,
        steps_per_epoch=len(training_set)/32,
        epochs=10,
        validation_data=test_set,
        validation_steps=len(test_set)/32)


# Saving the model
import h5py
classifier.save('model.h5')


print(model.history.keys())
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss

plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
