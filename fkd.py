#
from __future__ import print_function
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import h5py
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import math

#### TRAINING DATA
dataframe = read_csv("training.csv")
dataframe_wo_na = dataframe.dropna() # I'm wasting a LOT of training data!!

## 1. I'm training a model only on images with full features specifications.
image_data = dataframe_wo_na["Image"].apply(lambda im: np.fromstring(im, sep = ' '))
X_train = np.vstack(image_data.values) / 255.
X_train = X_train.astype(np.float32)
y_train = (dataframe_wo_na[dataframe_wo_na.columns[:-1]] - 48) / 48
X_train = X_train.reshape(-1,1,96,96) # to get into conv layer

#### Note: HOW to show a pic
#image = image_data[0,:].reshape((96,96))
#x = data[0,:30:2]
#y = data[1,:30:2]
#plt.imshow(image, cmap='gray')
#plt.plot(x, y, 'ro')
#plt.show()

# --> model = load_model('fkd-1.h5')
model = Sequential()
model.add(Convolution2D(16, 2,2, subsample=(1,1), border_mode='same',input_shape = (1,96,96)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), (1,1), border_mode='same'))
model.add(Convolution2D(16, 3,3, subsample=(1,1),border_mode='same'))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(30))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
model.fit(X_train, y_train.values, batch_size=64, nb_epoch=200, shuffle=True) # › 1.7306

model.save("fkd-full.h5".format(iteration))

## 2. now that I have a working model (albeit not really precise), I'm using it to fill the voids
## in the data as I run batches. I'm hoping that the only available data will pull the coeffs the right way anyway.

batch_run = 64

# selecting only rows with some null value (don't want to re-train on previous samples ... we'll see about that)
dataframe_w_na = pd.DataFrame(dataframe, index = pd.isnull(dataframe).any(1).nonzero()[0])
for i in range(100): # 100 epochs
	print("epoch {0} - ".format(i))
	for j in range(math.floor(dataframe_w_na.shape[0] / batch_run)): # one batch at a time
		img = dataframe_w_na.iloc[j*batch_run:(j + 1)*batch_run]["Image"].apply(lambda im: np.fromstring(im, sep = ' '))
		x = np.vstack(img.values) / 255.
		x = x.astype(np.float32)
		y = (dataframe_w_na.iloc[j*batch_run:(j + 1)*batch_run][dataframe_w_na.columns[:-1]] - 48) / 48
		x = x.reshape(-1,1,96,96) # to get into conv layer
		y_tmp = pd.DataFrame(model.predict(x, batch_size = batch_run)) # predict
		y_tmp.columns = y.columns
		y = y.reset_index(drop = True).fillna(y_tmp).values
		a = model.fit(x, y, batch_size = batch_run, nb_epoch = 1,verbose = 0)



## ALRIGHT: Now that I have ma model, Imma predict and save submission for y'all.
test = read_csv('test.csv')
# test.columns => Index(['ImageId', 'Image'], dtype='object')
id_lookup_table = read_csv('IdLookupTable.csv')
# id_lookup_table.columns => Index(['RowId', 'ImageId', 'FeatureName', 'Location'], dtype='object')
submission = read_csv('SampleSubmission.csv')
submission['Location'] = submission['Location'].astype(np.float32)
# submission.columns => Index(['RowId', 'Location'], dtype='object')

cols = ['left_eye_center_x','left_eye_center_y','right_eye_center_x','right_eye_center_y','left_eye_inner_corner_x','left_eye_inner_corner_y','left_eye_outer_corner_x','left_eye_outer_corner_y','right_eye_inner_corner_x','right_eye_inner_corner_y','right_eye_outer_corner_x','right_eye_outer_corner_y','left_eyebrow_inner_end_x','left_eyebrow_inner_end_y','left_eyebrow_outer_end_x','left_eyebrow_outer_end_y','right_eyebrow_inner_end_x','right_eyebrow_inner_end_y','right_eyebrow_outer_end_x','right_eyebrow_outer_end_y','nose_tip_x','nose_tip_y','mouth_left_corner_x','mouth_left_corner_y','mouth_right_corner_x','mouth_right_corner_y','mouth_center_top_lip_x','mouth_center_top_lip_y','mouth_center_bottom_lip_x','mouth_center_bottom_lip_y']
features = pd.DataFrame(columns=cols, index=[0])

for imageId, imageData in test.values:
	print("# {0}".format(imageId))
	image = np.array(imageData.split()).astype(np.float32) / 255.0
	y = model.predict(image.reshape(1,1,96,96), verbose=0)
	y = y*48 + 48
	features.iloc[0] = y
	q = "ImageId == {0}".format(imageId)
	rows = id_lookup_table.query(q)
	for rowid, image_id, feature_name, location in rows.values:
		a = submission.set_value(rowid - 1, 'Location', features[feature_name])

submission.dropna()
out = submission['Location']
out = pd.DataFrame(out)
out.to_csv('submission-2.csv', index='RowId')



