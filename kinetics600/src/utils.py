import os
import tensorflow as tf
from functools import partial

def parse_tfrecord_fn(example, shape=(128, 128, 3)):
    feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'video': tf.io.VarLenFeature(tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)

    # Function to decode each frame and potentially alter the number of channels
    def decode_frame(frame):
        return tf.io.decode_png(frame, channels=shape[2])
    
    # Decode each frame of the video
    video = tf.map_fn(decode_frame, tf.sparse.to_dense(example['video']), 
                      fn_output_signature=tf.TensorSpec(shape=shape, dtype=tf.uint8))

    label = example['label']
    
    return video, label


def load_dataset(tfrec_path, train=True, shape=(128, 128, 3)):
    # Set up the parse function
    parse_fn = partial(parse_tfrecord_fn, shape=shape)

    if train:
        tfrec_list = tf.io.gfile.glob(os.path.join(tfrec_path, "train_*.tfrec"))
    else:
        tfrec_list = tf.io.gfile.glob(os.path.join(tfrec_path, "val_*.tfrec"))

    # Create a dataset from the files
    dataset = tf.data.TFRecordDataset(tfrec_list, num_parallel_reads=tf.data.AUTOTUNE)
    # Map the parse function over the dataset
    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
    return dataset
