import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.keras.layers import Dropout, Input, Add, Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model

path = Path("part1/")
filenames = list(map(lambda x: x.name, path.glob('*.jpg')))

np.random.seed(10)
np.random.shuffle(filenames)

age_labels, gender_labels, image_path = [], [], []

for filename in filenames:
    image_path.append(filename)
    temp = filename.split('_')
    age_labels.append(temp[0])
    gender_labels.append(temp[1])


df = pd.DataFrame()
df['image'], df['age'], df['gender'] = image_path, age_labels, gender_labels

gender_dict = {0:"Male",1:"Female"}
df = df.astype({'age':'float32', 'gender': 'int32'})



train, test = train_test_split(df,
                               test_size=0.85,
                               random_state=42)

#converting Image to numpy array
x_train = []
for file in train.image:
    img = load_img("part1/"+file,
                   color_mode="grayscale" )
    img = img.resize((128,128), Image.Resampling.LANCZOS)
    img = np.array(img)
    x_train.append(img)

x_train = np.array(x_train)

x_train = x_train.reshape(len(x_train), 128,128,1)

x_train = x_train/255

y_gender = np.array(train.gender)
y_age = np.array(train.age)

input_size = (128,128,1)

#Crreating model
inputs = Input((input_size))
X = Conv2D(64,
           (3, 3),
           activation='relu',
           kernel_initializer = glorot_uniform(seed=0))(inputs)
X = BatchNormalization(axis = 3)(X)
X = MaxPooling2D((3, 3))(X)

X = Conv2D(128,
           (3, 3),
           activation='relu')(X)
X = MaxPooling2D((2, 2), strides=(2, 2))(X)

X = Conv2D(256,
           (3, 3),
           activation='relu')(X)
X = MaxPooling2D((2, 2))(X)

X = Flatten()(X)

dense_1 = Dense(256, activation='relu')(X)
dense_2 = Dense(256, activation='relu' )(X)
dense_3 = Dense(128, activation='relu' )(dense_2)

dropout_1 = Dropout(0.4)(dense_1)
dropout_2 = Dropout(0.4)(dense_3)

output_1 = Dense(1,
                 activation='sigmoid',
                 name='gender_output')(dropout_1)
output_2 = Dense(1,
                 activation='relu',
                 name='age_output')(dropout_2)

model = Model(inputs=[inputs],
              outputs=[output_1,output_2])

model.compile(loss=['binary_crossentropy','mae'],
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model_history = model.fit(x=x_train,
                          y=[y_gender, y_age],
                          batch_size = 10,
                          epochs= 20,
                          validation_split = 0.1)

num_samples = 12

plt.figure(figsize=(12, 8))
for i in range(num_samples):
    plt.subplot(3, 4, i + 1)

    index = np.random.randint(len(x_train))

    pred = model.predict(x_train[index].reshape(1, 128, 128, 1))
    pred_gender = gender_dict[round(pred[0][0][0])]
    pred_age = round(pred[1][0][0])

    plt.xlabel(f"Pred: Gender={pred_gender}, Age={pred_age}")
    plt.xticks([]), plt.yticks([])

    plt.imshow(x_train[index].reshape(128, 128), cmap='gray')
    plt.title(f"Real: Gender={gender_dict[y_gender[index]]}, Age={y_age[index]}")

plt.show()
plt.plot(model_history.history['gender_output_loss'])
plt.plot(model_history.history['val_gender_output_loss'])
plt.title('Gender loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(model_history.history['age_output_loss'])
plt.plot(model_history.history['val_age_output_loss'])
plt.title('Age loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()