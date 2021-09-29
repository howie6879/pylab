from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(256, (7, 7), activation="relu", input_shape=(1014, 69, 1)))
model.add(layers.MaxPooling2D(pool_size=(3, 1)))
model.add(layers.Conv2D(256, (7, 7), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(3, 1)))
model.add(layers.Conv2D(256, (3, 3), activation="relu"))
model.add(layers.Conv2D(256, (3, 3), activation="relu"))
model.add(layers.Conv2D(256, (3, 3), activation="relu"))
model.add(layers.Conv2D(256, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(3, 1)))

# model.add(layers.Flatten())
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(2, activation='sigmoid'))
model.summary()
