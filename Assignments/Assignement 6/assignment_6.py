import keras


def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize the input data - MNIST data is pixel arrays, so divide by max pixel value 255
    x_train = x_train/255.0
    x_test = x_test/255.0

    # Output is categorical - map from digit target to vector (e.g. 2 -> [0,0,1,0,0,0,0,0,0,0])
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test


def build_model(cnn=True):

    model = keras.Sequential()

    # Input is 28x28 image, single channel (grayscale)
    model.add(keras.Input(shape=(28, 28, 1)))

    if not cnn:

        ###  Fully connected neural network ###

        # Input is multidimensional, flattened to single dimension
        model.add(keras.layers.Flatten())
        # Add a hidden layer - units is number of neurons/layer width
        model.add(keras.layers.Dense(units=128, activation="relu"))
        
        
        


    else:

        ###  Convolutional neural network  ###

        # Add convolutional layer - filters is depth of layer output and kernel_size the convolution window
        model.add(keras.layers.Conv2D(filters=64, kernel_size=(4, 4), activation="relu", padding="same"))
        # Add pooling layer to downscale (MaxPooling downscales by returning the maximum value in each input window)
       
        model.add(keras.layers.MaxPooling2D(pool_size=(8, 8)))
    

        # Flatten internal dimensions before output - additional dense layers could also be included after this line
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation="relu"))
        model.add(keras.layers.Dropout(0.2))


    # Final model layer - the same for all model architectures
    # Activation is softmax
    model.add(keras.layers.Dense(units=10, activation="softmax"))

    return model



if __name__ == "__main__":

    epochs = 8
    learning_rate = 1.0

    x_train, y_train, x_test, y_test = load_mnist()
    model = build_model(cnn=False)  # set cnn=True for convolutional network, false for MLP

    # Compile model - Stochastic gradient descent is chosen for the optimizer and categorical cross entropy for the
    # loss calculation
    #optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    optimizer = keras.optimizers.Adam(learning_rate=5e-4)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

    # Show model architecture details and compare parameter counts
    model.summary()

    # Train the model on training data
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=128, verbose=1, validation_split=0.1)

    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
