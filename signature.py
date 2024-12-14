import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Corrected file paths using raw strings
gen_sign = [glob.glob(r"C:\Users\boyin\Documents\sign proj\dataset1\real\*"),
            glob.glob(r"C:\Users\boyin\Documents\sign proj\dataset2\real\*"),
            glob.glob(r"C:\Users\boyin\Documents\sign proj\dataset3\real\*"),
            glob.glob(r"C:\Users\boyin\Documents\sign proj\dataset4\real\*")]

forg_sign = [glob.glob(r"C:\Users\boyin\Documents\sign proj\dataset1\forge\*"),
             glob.glob(r"C:\Users\boyin\Documents\sign proj\dataset2\forge\*"),
             glob.glob(r"C:\Users\boyin\Documents\sign proj\dataset3\forge\*"),
             glob.glob(r"C:\Users\boyin\Documents\sign proj\dataset4\forge\*")]


 
 
gen_sign
 
 
forg_sign
 
 
def load_images(file_list):

    images = []

    for files in file_list:

        for file in files:

            img = cv2.imread(file)

            images.append(img)

    return images
 
 
gen_imgs = load_images(gen_sign)

forg_imgs = load_images(forg_sign)
 
 
print(gen_imgs[0])
 
plt.imshow(gen_imgs[0])

plt.show()
 
gen_imgs[0].shape
 
num_gen = sum([len(x) for x in gen_sign])

num_forg = sum([len(x) for x in forg_sign])

print(f'Number of real signatures: {num_gen}')

print(f'Number of forged signatures: {num_forg}')
 
 
gen_shape = [x.shape for x in gen_imgs]

forg_shape = [x.shape for x in forg_imgs]

print('Shape of real images: ', gen_shape[:5])

print('Shape of forged images: ', forg_shape[:5])
 
gen_color = [x.shape[2] for x in gen_imgs]

forg_color = [x.shape[2] for x in forg_imgs]

print('Color channels of real images: ', gen_color[:5])

print('Color channels of forged images: ',forg_color[:5])
 
gen_pixels = [x.mean() for x in gen_imgs]

forg_pixels = [x.mean() for x in forg_imgs]

print('Mean pixels values of real images: ',gen_pixels[:5])

print('Mean pixels value of forged images: ',forg_pixels[:5])
 
 
fig,ax = plt.subplots(2,5,figsize =(15,6))
 
for i in range(5):

    ax[0,i].imshow(cv2.cvtColor(gen_imgs[i],cv2.COLOR_BGR2RGB))

    ax[0,i].set_title('Real Signature')

    ax[0,i].axis('off')

    ax[1,i].imshow(cv2.cvtColor(forg_imgs[i],cv2.COLOR_BGR2RGB))

    ax[1,i].set_title('Forged Signature')

    ax[1,i].axis('off')

plt.show()
 
 
image_data = pd.DataFrame(columns=['Type','Shape','Color Channels','Mean Pixel Value'])
 
 
data_list = []
 
for i in range(len(gen_shape)):

    data_list.append({'Type': 'Real',

                      'Shape': gen_shape[i],

                      'Color Channels': gen_color[i],

                      'Mean Pixel Value': gen_pixels[i]})
 
# Convert the list of dictionaries to a DataFrame
 
 
for i in range(len(forg_shape)):

    data_list.append({'Type': 'Forged',

                      'Shape': forg_shape[i],

                      'Color Channels': forg_color[i],

                      'Mean Pixel Value': forg_pixels[i]})
 
# Convert the list of dictionaries to a DataFrame

image_data = pd.concat([image_data, pd.DataFrame(data_list)], ignore_index=True)
 
 
 
image_data.head()
 
image_data.tail()
 
image_data.shape
 
 
print(image_data.describe())
 
 
fig, ax = plt.subplots(2,2,figsize=(10,6))

image_data[image_data['Type']=='Real']['Color Channels'].hist(ax=ax[0,0])

image_data[image_data['Type']=='Forged']['Color Channels'].hist(ax=ax[1,0])

image_data[image_data['Type']=='Real']['Mean Pixel Value'].hist(ax=ax[0,1])

image_data[image_data['Type']=='Forged']['Mean Pixel Value'].hist(ax=ax[1,1])
 
ax[0,0].set_title('Color Channels of Real Signatures')

ax[0,1].set_title('Mean Pixel Value of Real Signatures')

ax[1,0].set_title('Color Channels of Forged Signatures')

ax[1,1].set_title('Mean Pixel Value of Forged Signatures')
 
plt.tight_layout()

plt.show()


 
import tensorflow
from tensorflow import keras
from keras import layers
 
#load the data
(train_data, train_labels),(test_data,test_labels) = keras.datasets.mnist.load_data()
 
train_labels
 
# preprocess the data
train_data = train_data.reshape((train_data.shape[0], train_data.shape[1],train_data.shape[2]* 1))
test_data = test_data.reshape((test_data.shape[0],test_data.shape[1],test_data.shape[2]*1))
train_data = train_data.astype('float32')/255
test_data = test_data.astype('float32') / 255
 
 
 
 
# define the model
model = keras.Sequential([layers.LSTM(64,input_shape=(train_data.shape[1], train_data.shape[2])),
                          layers.Dense(10, activation='softmax')])
 
 
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
 
model.summary()
 
 
 
 