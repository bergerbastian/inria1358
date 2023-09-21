# imports
import tensorflow as tf

def initialize_model_unet(X):
    """
    Initialize the UNET Neural Network with random weights and return model
    """
    # $CODE_BEGIN
    input_shape = X[0].shape
    inputs = tf.keras.layers.Input(input_shape)

    c1 = tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same")(inputs)
    p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

    c2 = tf.keras.layers.Conv2D(128, (3,3), activation="relu", padding="same")(p1)
    p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

    c3 = tf.keras.layers.Conv2D(256, (3,3), activation="relu", padding="same")(p2)
    p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

    b = tf.keras.layers.Conv2D(512, (3,3), activation="relu", padding="same")(p3)

    u1 = tf.keras.layers.UpSampling2D((2,2))(b)
    concat1 = tf.keras.layers.Concatenate()([u1, c3])
    c4= tf.keras.layers.Conv2D(256, (3,3), activation="relu", padding="same")(concat1)

    u2 = tf.keras.layers.UpSampling2D((2,2))(c4)
    concat2 = tf.keras.layers.Concatenate()([u2, c2])
    c5= tf.keras.layers.Conv2D(128, (3,3), activation="relu", padding="same")(concat2)

    u3 = tf.keras.layers.UpSampling2D((2,2))(c5)
    concat3 = tf.keras.layers.Concatenate()([u3, c1])
    c6= tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same")(concat3)

    outputs = tf.keras.layers.Conv2D(1, (1,1), activation="sigmoid")(c6)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # $CODE_END

    print("✅ UNET Model initialized")

    return model

def initialize_model_segnet(X):
    """
    Initialize the SEGNET Neural Network with random weights and return model
    """
    # $CODE_BEGIN
    input_shape = X[0].shape
    def encoder_block(x, filters):
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        return x

    def decoder_block(x, filters):
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def build_segnet(input_shape):
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Encoder
        enc1 = encoder_block(inputs, 64)
        enc2 = encoder_block(enc1, 128)
        enc3 = encoder_block(enc2, 256)
        
        # Decoder
        dec3 = decoder_block(enc3, 256)
        dec2 = decoder_block(dec3, 128)
        dec1 = decoder_block(dec2, 64)
        
        # Pixel-wise classification
        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(dec1)
        
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        return model

    # Create the SegNet model
    segnet_model = build_segnet(input_shape)
    # $CODE_END

    print("✅ SEGNET Model initialized")

    return segnet_model


def compile_model(model: tf.keras.Model, learning_rate=0.0005):
    """
    Compile the Neural Network and return model
    """
    # $CODE_BEGIN
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    # $CODE_END

    print("✅ Model compiled")

    return model

def train_model(
    model: tf.keras.Model,
    X,
    y,
    validation_data,
    batch_size=256,
    patience=2
):
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    # $CODE_BEGIN
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )
    # $CODE_END

    print(f"✅ Model trained on {len(X)} patches.")

    return model, history

def make_model(
    X,
    y,
    validation_data,
    model_name='segnet',
    learning_rate=0.0005,
    batch_size=32,
    epochs=2,
    patience=2,
):
    '''
    Initialize, compile, and train model specified by model_name "segnet", "unet". Return model, history
    '''    
    # $CODE_BEGIN
    if model_name == 'segnet':
        model = initialize_model_segnet(X)
    elif model_name == 'unet':
        model = initialize_model_unet(X)
    else:
        print(f'No {model_name} model defined')
        return 1
    
    model = compile_model(model=model, learning_rate=learning_rate)

    model, history = train_model(
        model=model,
        X=X,
        y=y,
        validation_data=validation_data,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience
    )
    # $CODE_END

    return model, history   