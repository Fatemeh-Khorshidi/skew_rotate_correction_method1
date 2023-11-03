import keras
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from preprocess import prepare_dataset



def train_model(train_path, val_path, num_epochs, batch_size ,image_shape, num_classes):
    # Prepare the training and validation datasets
    Xtrain, ytrain = prepare_dataset(train_path)
    Xval, yval = prepare_dataset(val_path)
    

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='LeakyReLU', input_shape=(image_shape)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='LeakyReLU'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', activation='LeakyReLU'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='LeakyReLU'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same', activation='LeakyReLU'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='LeakyReLU'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(num_classes, activation="softmax"))

    checkpoint = ModelCheckpoint('rotate_model.h5',
                                 monitor="val_loss",
                                 mode="min",
                                 save_best_only=True,
                                 verbose=1)

    earlystop = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=10,
                              verbose=1,
                              restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.2,
                                  # factor by which the learning rate will be reduced. new_lr = lr * factor
                                  patience=10,
                                  verbose=1,
                                  min_delta=0.0001)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    history = model.fit(Xtrain, ytrain, validation_data = (Xval, yval), epochs=num_epochs, callbacks = [earlystop, checkpoint, reduce_lr], batch_size=batch_size)



    return model, history


if __name__ == "__main__":
    train_path = "/content/train"
    val_path = "/content/val"
    num_epochs = 20
    batch_size = 4
    num_classes = 4
    image_shape = (668,668,3)
    model, history = train_model(train_path, val_path, num_epochs, batch_size, image_shape, num_classes)
    model.save('rotate_model.h5')
