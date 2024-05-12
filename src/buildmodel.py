import efficientnet.tfkeras as efn
import tensorflow as tf


def build_model(USE_KAGGLE_SPECTROGRAMS, USE_EEG_SPECTROGRAMS, version="b0"):
    inp = tf.keras.Input(shape=(128, 256, 8))
    # Select the EfficientNet model based on the provided version
    if version == "b0":
        base_model = efn.EfficientNetB0(
            include_top=False, weights=None, input_shape=None
        )
    elif version == "b1":
        base_model = efn.EfficientNetB1(
            include_top=False, weights=None, input_shape=None
        )
    elif version == "b2":
        base_model = efn.EfficientNetB2(
            include_top=False, weights=None, input_shape=None
        )
    elif version == "b3":
        base_model = efn.EfficientNetB3(
            include_top=False, weights=None, input_shape=None
        )
    elif version == "b4":
        base_model = efn.EfficientNetB4(
            include_top=False, weights=None, input_shape=None
        )
    elif version == "b5":
        base_model = efn.EfficientNetB5(
            include_top=False, weights=None, input_shape=None
        )

    # Reshape and concatenate the input to adapt to the EfficientNet input requirements
    x1 = [inp[:, :, :, i : i + 1] for i in range(4)]
    x1 = tf.keras.layers.Concatenate(axis=1)(x1)
    x2 = [inp[:, :, :, i + 4 : i + 5] for i in range(4)]
    x2 = tf.keras.layers.Concatenate(axis=1)(x2)

    if USE_KAGGLE_SPECTROGRAMS & USE_EEG_SPECTROGRAMS:
        x = tf.keras.layers.Concatenate(axis=2)([x1, x2])
    elif USE_EEG_SPECTROGRAMS:
        x = x2
    else:
        x = x1
    x = tf.keras.layers.Concatenate(axis=3)([x, x, x])

    # Feed into the base model
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(x)

    # Compile the model
    model = tf.keras.Model(inputs=inp, outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.KLDivergence()
    model.compile(loss=loss, optimizer=opt)

    return model
