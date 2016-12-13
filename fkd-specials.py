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

## 1. BASE MODEL
image_data = dataframe_wo_na["Image"].apply(lambda im: np.fromstring(im, sep = ' '))
X_train = np.vstack(image_data.values) / 255.
X_train = X_train.astype(np.float32)
y_train = (dataframe_wo_na[dataframe_wo_na.columns[:-1]] - 48) / 48
X_train = X_train.reshape(-1,1,96,96) # to get into conv layer
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
history = model.fit(X_train, y_train.values, batch_size = 64, nb_epoch = 200, shuffle = True, validation_split = 0.3) # › 1.7306

model.save('fkd-base.h5')

## then i run the "special" detectors
history = []
for i in range(0,30,2): # for every feature
	model = Sequential()
	model.add(Convolution2D(16, 2,2, subsample=(1,1), border_mode='same',input_shape = (1,96,96)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D((2,2), (1,1), border_mode='same'))
	model.add(Dropout(0.3))
	model.add(Convolution2D(16, 3,3, subsample=(1,1),border_mode='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D((2,2), (1,1), border_mode='same'))
	model.add(Dropout(0.3))
	model.add(Convolution2D(32, 2,2, subsample=(2,2),border_mode='same'))
	model.add(Activation('relu'))
	model.add(Dropout(0.3))
	model.add(Flatten())
	model.add(Dense(80))
	model.add(Activation('relu'))
	model.add(Dropout(0.3))
	model.add(Dense(2))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
	df = dataframe.iloc[:,i:i+2].join(dataframe["Image"]).dropna()
	image_data = df["Image"].apply(lambda im: np.fromstring(im, sep = ' '))
	X_train = np.vstack(image_data.values) / 255.
	X_train = X_train.astype(np.float32)
	y_train = (df[df.columns[:-1]] - 48) / 48
	X_train = X_train.reshape(-1,1,96,96) # to get into conv layer
	history.append(model.fit(X_train, y_train.values, batch_size = 64, nb_epoch = 200, shuffle = True))
	model.save("fkd--{0}.h5".format(i))

## ALRIGHT: Now that I have ma model, Imma predict and save submission for y'all.
test = read_csv('test.csv')
# test.columns => Index(['ImageId', 'Image'], dtype='object')
id_lookup_table = read_csv('IdLookupTable.csv')
# id_lookup_table.columns => Index(['RowId', 'ImageId', 'FeatureName', 'Location'], dtype='object')
submission = read_csv('SampleSubmission.csv')
submission['Location'] = submission['Location'].astype(np.float32)

cols = ['ImgId', 'left_eye_center_x','left_eye_center_y','right_eye_center_x','right_eye_center_y','left_eye_inner_corner_x','left_eye_inner_corner_y','left_eye_outer_corner_x','left_eye_outer_corner_y','right_eye_inner_corner_x','right_eye_inner_corner_y','right_eye_outer_corner_x','right_eye_outer_corner_y','left_eyebrow_inner_end_x','left_eyebrow_inner_end_y','left_eyebrow_outer_end_x','left_eyebrow_outer_end_y','right_eyebrow_inner_end_x','right_eyebrow_inner_end_y','right_eyebrow_outer_end_x','right_eyebrow_outer_end_y','nose_tip_x','nose_tip_y','mouth_left_corner_x','mouth_left_corner_y','mouth_right_corner_x','mouth_right_corner_y','mouth_center_top_lip_x','mouth_center_top_lip_y','mouth_center_bottom_lip_x','mouth_center_bottom_lip_y']
features = pd.DataFrame(columns=cols, index=range(test.shape[0]))

for j in range(0,30,2):
	print("model {0}".format(j))
	model = load_model("fkd--{0}.h5".format(j))
	h = 0
	for imageId, imageData in test.values:
		print("# {0}".format(imageId))
		image = np.array(imageData.split()).astype(np.float32) / 255.0
		y = model.predict(image.reshape(1,1,96,96), verbose=0)
		y = y*48 + 48
		if features.loc[lambda df: df.ImgId == imageId].shape[0] == 0:
			features.iloc[h]['ImgId'] = imageId
			h += 1
		features.loc[lambda df: df.ImgId == imageId, cols[j+1]:cols[j+2]] = y
		q = "ImageId == {0}".format(imageId)
		rows = id_lookup_table.query(q)
		for rowid, image_id, feature_name, location in rows.values:
			a = submission.set_value(rowid - 1, 'Location', features.loc[lambda df: df.ImgId == image_id, feature_name])


submission.dropna()
out = submission['Location']
out = pd.DataFrame(out)
out.to_csv('submission-4.csv', index='RowId')


