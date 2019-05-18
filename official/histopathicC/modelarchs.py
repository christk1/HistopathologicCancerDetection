import tensorflow as tf

def Alexlike(data_format):
    """Alexlike model.

    Args:
      data_format: Either 'channels_first' or 'channels_last'. 'channels_first' is
        typically faster on GPUs while 'channels_last' is typically faster on
        CPUs. See
        https://www.tensorflow.org/performance/performance_guide#data_formats

    Returns:
      A tf.keras.Model.
    """
    if data_format == 'channels_first':
        input_shape = [3, 48, 48]
    else:
        assert data_format == 'channels_last'
        input_shape = [48, 48, 3]

    l = tf.keras.layers
    r = tf.keras.regularizers
    ini = tf.keras.initializers
    # The model consists of a sequential chain of layers, so tf.keras.Sequential
    # (a subclass of tf.keras.Model) makes for a compact description.
    return tf.keras.Sequential(
        [
            l.Reshape(
                target_shape=input_shape,
                input_shape=(3 * 48 * 48,)),

            l.Conv2D(data_format=data_format,
                     filters=6,
                     kernel_size=4,
                     strides=4,
                     padding="valid",
                     activation=tf.nn.relu,
                     kernel_initializer=ini.he_normal(),
                     kernel_regularizer=r.l2(0.1)),

            l.MaxPooling2D(data_format=data_format, pool_size=3, strides=2, padding="valid"),

            l.Conv2D(data_format=data_format,
                     filters=16,
                     kernel_size=5,
                     strides=1,
                     padding="same",
                     activation=tf.nn.relu,
                     kernel_initializer=ini.he_normal(),
                     kernel_regularizer=r.l2(0.1)),

            l.MaxPooling2D(data_format=data_format, pool_size=3, strides=2, padding="valid"),


            l.Conv2D(data_format=data_format,
                     filters=120,
                     kernel_size=3,
                     strides=1,
                     padding="same",
                     activation=tf.nn.relu,
                     kernel_initializer=ini.he_normal(),
                     kernel_regularizer=r.l2(0.1)),

            # Flatten tensor into a batch of vectors
            l.Flatten(),

            l.Dense(units=84, activation=tf.nn.relu, kernel_initializer=ini.he_normal()),
            l.Dropout(0.5),
            l.Dense(units=1, kernel_initializer=ini.he_normal())
        ])

def VGGlike(data_format):
    """VGGlike model.

    Args:
      data_format: Either 'channels_first' or 'channels_last'. 'channels_first' is
        typically faster on GPUs while 'channels_last' is typically faster on
        CPUs. See
        https://www.tensorflow.org/performance/performance_guide#data_formats

    Returns:
      A tf.keras.Model.
    """
    if data_format == 'channels_first':
        input_shape = [3, 48, 48]
    else:
        assert data_format == 'channels_last'
        input_shape = [48, 48, 3]

    l = tf.keras.layers
    r = tf.keras.regularizers
    ini = tf.keras.initializers
    # The model consists of a sequential chain of layers, so tf.keras.Sequential
    # (a subclass of tf.keras.Model) makes for a compact description.
    return tf.keras.Sequential(
        [
            l.Reshape(
                target_shape=input_shape,
                input_shape=(3 * 48 * 48,)),

            # Block 1
            l.Conv2D(data_format=data_format,
                     filters=64,
                     kernel_size=3,
                     strides=1,
                     padding="same",
                     activation=tf.nn.relu),

            l.Conv2D(data_format=data_format,
                     filters=64,
                     kernel_size=3,
                     strides=1,
                     padding="same",
                     activation=tf.nn.relu),

            l.MaxPooling2D(data_format=data_format, pool_size=2, strides=2, padding="valid"),

            # Block 2
            l.Conv2D(data_format=data_format,
                     filters=128,
                     kernel_size=3,
                     strides=1,
                     padding="same",
                     activation=tf.nn.relu),

            l.Conv2D(data_format=data_format,
                     filters=128,
                     kernel_size=3,
                     strides=1,
                     padding="same",
                     activation=tf.nn.relu),

            l.MaxPooling2D(data_format=data_format, pool_size=2, strides=2, padding="valid"),

            # Block 3
            l.Conv2D(data_format=data_format,
                     filters=256,
                     kernel_size=3,
                     strides=1,
                     padding="same",
                     activation=tf.nn.relu),

            l.Conv2D(data_format=data_format,
                     filters=256,
                     kernel_size=3,
                     strides=1,
                     padding="same",
                     activation=tf.nn.relu),

            l.Conv2D(data_format=data_format,
                     filters=256,
                     kernel_size=3,
                     strides=1,
                     padding="same",
                     activation=tf.nn.relu),

            l.MaxPooling2D(data_format=data_format, pool_size=2, strides=2, padding="valid"),

            # Flatten tensor into a batch of vectors
            l.Flatten(),

            l.Dense(units=2048, activation=tf.nn.relu),
            l.Dropout(0.5),
            l.Dense(units=2048, activation=tf.nn.relu),
            l.Dropout(0.2),
            l.Dense(units=1)
        ])
