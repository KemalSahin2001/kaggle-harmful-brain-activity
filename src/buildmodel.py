import efficientnet.tfkeras as efn
import tensorflow as tf


def build_model(USE_KAGGLE_SPECTROGRAMS, USE_EEG_SPECTROGRAMS):
    inp = tf.keras.Input(shape=(128, 256, 8))
    base_model = efn.EfficientNetB0(include_top=False, weights=None, input_shape=None)
    base_model.load_weights(
        "..\models\TF EfficientNet ImageNet Weights\efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
    )

    # RESHAPE INPUT 128x256x8 => 512x512x3 MONOTONE IMAGE
    # KAGGLE SPECTROGRAMS
    x1 = [inp[:, :, :, i : i + 1] for i in range(4)]
    x1 = tf.keras.layers.Concatenate(axis=1)(x1)
    # EEG SPECTROGRAMS
    x2 = [inp[:, :, :, i + 4 : i + 5] for i in range(4)]
    x2 = tf.keras.layers.Concatenate(axis=1)(x2)
    # MAKE 512X512X3
    if USE_KAGGLE_SPECTROGRAMS & USE_EEG_SPECTROGRAMS:
        x = tf.keras.layers.Concatenate(axis=2)([x1, x2])
    elif USE_EEG_SPECTROGRAMS:
        x = x2
    else:
        x = x1
    x = tf.keras.layers.Concatenate(axis=3)([x, x, x])

    # OUTPUT
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(x)

    # COMPILE MODEL
    model = tf.keras.Model(inputs=inp, outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.KLDivergence()

    model.compile(loss=loss, optimizer=opt)

    return model
