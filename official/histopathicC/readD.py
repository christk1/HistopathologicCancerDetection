import tensorflow as tf
import matplotlib.pyplot as plt
import random
import math

tf.enable_eager_execution()

raw_image_dataset = tf.data.TFRecordDataset('dataset/train_images.tfrecords')


def rotatedRectWithMaxArea(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr


def crop_centr(img):
    image_size = 96
    image_cropped = 48
    cnt_c = (image_size - image_cropped) // 2
    patched_img = img[cnt_c:cnt_c + image_cropped, cnt_c:cnt_c + image_cropped]
    return patched_img

def _parse_image_function(example_proto):
    # Create a dictionary describing the features.
    image_feature_description = {
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
    }

    # Parse the input tf.Example proto using the dictionary above.
    parsed = tf.parse_single_example(example_proto, image_feature_description)

    image = tf.image.decode_jpeg(parsed["image_raw"])
    image = tf.reshape(image, [48, 48, 3])
    #image = tf.image.random_flip_left_right(image) # flip image randomly with a 50% chance
    #image = tf.image.random_brightness(image, max_delta=0.3) # random brightness
    image = tf.image.rot90(image, k=1)
    #print("ankle:", r_num_rotate)
    #image = tf.image.random_saturation(image, 0, 2)
    #image = tf.image.random_hue(image, 0.1)
    label = tf.cast(parsed["label"], tf.int32)
    return {"image": image}


parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

for image_features in parsed_image_dataset:
    plt.imshow(image_features["image"])
    plt.show()
