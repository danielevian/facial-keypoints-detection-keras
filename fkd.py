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

#### TRAINING DATA
dataframe = read_csv("training.csv")
dataframe = dataframe.dropna() # I'm wasting a LOT of training data!!

## ALL IMAGES
## Staying in Dataframe domain because of some nice
image_data = dataframe["Image"].apply(lambda im: np.fromstring(im, sep = ' '))
X_train = np.vstack(image_data.values) / 255.
X_train = X_train.astype(np.float32)
y_train = (dataframe[dataframe.columns[:-1]] - 48) / 48
X_train = X_train.reshape(-1,1,96,96) # to get into conv layer

#### Want to show a pic?
#image = image_data[0,:].reshape((96,96))
#x = data[0,:30:2]
#y = data[1,:30:2]
#plt.imshow(image, cmap='gray')
#plt.plot(x, y, 'ro')
#plt.show()

#1 --> simple dense layer
# model = Sequential()
# model.add(Dense(100, input_dim=96*96))
# model.add(Activation('relu'))
# model.add(Dense(30))
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])


#2 --> using conv
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
model.fit(X_train, y_train, batch_size=64, nb_epoch=200, shuffle=True) # › 1.7306

model.save("fkd-1.h5".format(iteration))





test = read_csv('test.csv')
# test.columns => Index(['ImageId', 'Image'], dtype='object')
id_lookup_table = read_csv('IdLookupTable.csv')
# id_lookup_table.columns => Index(['RowId', 'ImageId', 'FeatureName', 'Location'], dtype='object')
submission = read_csv('SampleSubmission.csv')
submission['Location'] = submission['Location'].astype(np.float32)
# >>> submission.columns
# Index(['RowId', 'Location'], dtype='object')

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
out.to_csv('submission-1.csv', index='RowId')



