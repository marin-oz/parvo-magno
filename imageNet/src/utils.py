import numpy as np
import pandas as pd
import tensorflow as tf
import os, math
import scipy.ndimage as ndimage
from PIL import Image
from tensorflow_addons.image import gaussian_filter2d

AUTO = tf.data.AUTOTUNE # used in tf.data.Dataset API

def load_images(list_images, img_size, color=True, texture=False):
    '''
    Load images from a list of file paths and return the images and labels.

    args:
    list_images(list): the list of images to load
    img_size(tuple): the size of the image
    color(bool): if True, the image is in color; otherwise, grayscale
    texture(bool): set to True for texture images labels
    '''
    # Arg check
    assert isinstance(list_images, list), "list_images must be a list"
    assert isinstance(img_size, tuple), "img_size must be a tuple"
    assert isinstance(color, bool), "color must be a boolean"
    assert isinstance(texture, bool), "texture must be a boolean"
    assert len(img_size) == 3, "img_size must be a tuple of length 3"

    input_images = np.zeros(shape=(len(list_images),*img_size),dtype=np.uint8)
    if texture:
        class_labels = []
    else:
        class_labels = np.zeros(shape=(len(list_images),),dtype=np.int32)

    for i, file in enumerate(list_images): 
        newimg = Image.open(file)
        if not color:
            newimg = newimg.convert('L')
        if newimg.mode != 'RGB':
            newimg = newimg.convert('RGB')
        newimg = np.asarray(newimg).astype(np.uint8)
        if texture:
            class_labels.append((os.path.basename(file).split('.')[0]))
        else:
            class_labels[i] = int(os.path.basename(file).split('_')[0])
        input_images[i] = newimg
    
    return input_images, class_labels

def read_tfrecord(example, img_size, color, saturation=None):
    """
    Function to read a single TFRecord example and return the image and label.

    Args:
    example (tf.train.Example): A single TFRecord example.
    color (bool): If True, the image is read in color; 
                  otherwise, it is read in grayscale.

    Returns:
    image (tf.Tensor): A decoded and reshaped image tensor.
    label (tf.Tensor): The corresponding label tensor.
    """

    # Define the features and their types within the TFRecord example
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "class": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, features)
    
    # Decode the image based on the color flag and reshape it to the desired size
    if color:
        image = tf.image.decode_jpeg(example['image'], channels=3)
    else:
        image = tf.image.decode_jpeg(example['image'], channels=1)
        image = tf.image.grayscale_to_rgb(image)
    if saturation is not None:
        image = tf.image.adjust_saturation(image, saturation)
    image = tf.reshape(image, img_size)

    # Extract the label from the example
    label = example['class']
    
    return image, label

@tf.function
def gaussianBlur(image, sigma):
    # kernel_size according to a MATLAB function (imgaussfilt)
    kernel_size = int(2 * np.ceil(2 * sigma) + 1)
    with tf.device("/gpu:0"):
        filtered = gaussian_filter2d(image=image,
                             filter_shape=kernel_size,
                             sigma=float(sigma))
    return filtered

def load_dataset(tfrec_path, img_size, color, blur_sigma, blur_batch_size=256, saturation=None):
    '''
    Load a dataset from TFRecord files.

    args:
    tfrec_path(str): the path to the TFRecord files
    img_size(tuple): the size of the image
    color(bool): if True, the image is in color; otherwise, it is in grayscale
    blur_sigma(float): the sigma of the Gaussian blur
    blur_batch_size(int): the batch size for applying the Gaussian blur
    saturation(float): the saturation of the image
    '''
    # Arg check
    assert isinstance(tfrec_path, str), "tfrec_path must be a string"
    assert isinstance(img_size, tuple), "img_size must be a tuple"
    assert isinstance(color, bool), "color must be a boolean"
    assert isinstance(blur_sigma, (int, float)), "blur_sigma must be a number"
    assert isinstance(blur_batch_size, int), "blur_batch_size must be an integer"
    assert isinstance(saturation, (int, float, type(None))), "saturation must be a number or None"
    assert len(img_size) == 3, "img_size must be a tuple of length 3"
    assert blur_sigma >= 0, "blur_sigma must be greater than or equal to 0"
    assert blur_batch_size > 0, "blur_batch_size must be greater than 0"
    assert saturation is None or saturation >= 0, "saturation must be greater than or equal to 0"

    tfrec_list = tf.io.gfile.glob(os.path.join(tfrec_path, "*.tfrec"))
    dataset = tf.data.TFRecordDataset(tfrec_list, num_parallel_reads=AUTO)
    dataset = dataset.map(lambda x: read_tfrecord(x, img_size, color, saturation), num_parallel_calls=AUTO)
    if blur_sigma > 0:
        dataset = dataset.batch(blur_batch_size).map(lambda x, y: (gaussianBlur(x, blur_sigma), y), num_parallel_calls=AUTO).unbatch()
        
    return dataset

def saveDataFrame(dataframe, path, overwrite=False):
    """
    Save a pandas dataframe to a csv file.
    If overwrite is False, the file will be saved as 
    a concatenation with the existing file if any.
    """
    # Arg check
    assert isinstance(dataframe, pd.DataFrame), "dataframe must be a pandas dataframe"
    assert isinstance(path, str), "path must be a string"
    assert isinstance(overwrite, bool), "overwrite must be a boolean"

    if os.path.exists(path) and (not overwrite):
        dataframe_prev = pd.read_csv(path, index_col=0)
        dataframe = pd.concat([dataframe_prev, dataframe], axis=1)
    dataframe.to_csv(path)

def normalize(x, x_min=None, x_max=None):
    '''
    Normalize a value between a minimum and maximum value.

    args:
    x(np.array, float): the values to normalize
    x_min(float): the minimum value. If None, the minimum value of x is used.
    x_max(float): the maximum value. If None, the maximum value of x is used.
    '''
    if x_min is None:
        x_min = np.min(x)
    if x_max is None:
        x_max = np.max(x)
    return (x - x_min) / (x_max - x_min)

def calc_color_index(img, n=0):
    '''
    Calculate the color index of an image.

    args:
    img(np.array): the image to calculate the color index
    n(int): the number of top colorful pixels to consider. 
            If n=0, the mean of all pixels is returned.
    '''
    # Arg check
    assert isinstance(img, np.ndarray), "Image must be a numpy array"
    assert isinstance(n, int), "n must be an integer"
    assert n >= 0, "n must be greater than or equal to 0"

    x = img[:, :, 0] * math.cos(0) + img[:, :, 1] * math.cos(math.radians(120)) + img[:, :, 2] * math.cos(math.radians(-120))
    y = img[:, :, 0] * math.sin(0) + img[:, :, 1] * math.sin(math.radians(120)) + img[:, :, 2] * math.sin(math.radians(-120))
    col_vec_length = np.sqrt(x * x + y * y)

    if n == 0:
        return np.mean(col_vec_length, axis=None)
    else:
        return np.mean(np.flip(np.sort(col_vec_length, axis=None))[0:n])

def azimuthal_average(ps, angle_interval, r_min, r_max):
    '''
    Calculate the azimuthal average of an image. 
    Due to symmetry, 180 to 360 degrees is the same as 0 to 180 degrees.

    args:
    ps(np.array): 2d power spectrum to calculate the azimuthal average
    angle_interval(int): the angle interval for azimuthal averaging
    r_min(int): the minimum radius to consider
    r_max(int): the maximum radius to consider
    '''
    h = ps.shape[0]
    w = ps.shape[1]
    h_mid = h // 2
    w_mid = w // 2

    Y, X = np.ogrid[0:h, 0:w]
    r = np.hypot(-(Y - h_mid), (X - w_mid))
    mask = np.logical_and(r>r_min, r<r_max)

    theta = np.rad2deg(np.arctan2(-(Y - h_mid), (X - w_mid)))
    theta = np.mod(theta + angle_interval / 2 + 360, 360)
    theta = (angle_interval * (theta // angle_interval)).astype(int)
    dia = np.multiply(np.ones(theta.shape) * 180, (theta >= 180).astype(int))
    theta = theta - dia + 1
    theta = np.multiply(mask, theta)
    theta = theta - 1

    angle_index = np.arange(0, 180, int(angle_interval))
    means = ndimage.mean(ps, theta, index=angle_index)

    return angle_index, means

def radial_average(ps):
    '''
    Calculate the radial average of an image.

    args:
    ps(np.array): 2d power spectrum to calculate the radial average
    '''
    # Arg check
    assert isinstance(ps, np.ndarray), "ps must be a numpy array"

    h = ps.shape[0]
    w = ps.shape[1]
    h_mid = h // 2
    w_mid = w // 2
    r_max = np.min((w_mid, h_mid))

    Y, X = np.ogrid[0:h, 0:w]
    r = np.hypot(X - w_mid, Y - h_mid).astype(int)
    r_index = np.arange(0, r_max)
    means = ndimage.mean(ps, r, index=r_index)

    return r_index, means

def make_larger(original, n_times):
    '''
    Make an image larger by a factor of n_times.

    args:
    original(np.array): the original image
    n_times(int): upsampling factor
    '''
    # Arg check
    assert isinstance(original, np.ndarray), "Original image must be a numpy array"
    assert isinstance(n_times, int), "n_times must be an integer"
    assert n_times > 0, "n_times must be greater than 0"

    large = np.zeros((original.shape[0]*n_times,original.shape[1]*n_times))
    for i in range(original.shape[0]):
        for j in range(original.shape[1]):
            large[i * n_times:(i + 1)*n_times, j * n_times:(j + 1) * n_times] = original[i,j]

    return large

def fft_average(abs_spectrum, n_times, angle_interval):
    '''
    Calculate the radial and azimuthal averages of the FFT of an image.

    args:
    abs_spectrum(np.array): the absolute spectrum of the image
    n_times(int): upsampling factor for averaging
    angle_interval(int): the angle interval for azimuthal averaging
    '''
    # Arg check
    assert isinstance(abs_spectrum, np.ndarray), "abs_spectrum must be a numpy array"
    assert isinstance(n_times, int), "n_times must be an integer"
    assert isinstance(angle_interval, int), "angle_interval must be an integer"
    assert n_times > 0, "n_times must be greater than 0"
    assert angle_interval > 0, "angle_interval must be greater than 0"

    as_large = make_larger(abs_spectrum, n_times)
    r_index, r_means = radial_average(as_large)
    a_index, a_means = azimuthal_average(as_large, angle_interval, 0, abs_spectrum.shape[0] * n_times / 4)
 
    return r_index, r_means, a_index, a_means

def mean_resultant_length(a_index, a_mean):
    '''
    Calculate the mean resultant length of the azimuthal average of the FFT of an image.

    args:
    a_index(np.array): the angle indices of the azimuthal average
    a_mean(np.array): the azimuthal average of the FFT
    '''
    #arg check
    assert isinstance(a_index, np.ndarray), "a_index must be a numpy array"
    assert isinstance(a_mean, np.ndarray), "a_mean must be a numpy array"
    assert a_index.shape == a_mean.shape, "a_index and a_mean must have the same shape"

    return np.abs(np.sum(a_mean * np.exp(2 * math.pi * 1j * a_index / 180))) / np.sum(a_mean)

def rgb2gray(rgb):
    '''
    Convert an RGB image to grayscale.

    args:
    rgb(np.array): the RGB image to convert to grayscale
    '''
    # Arg check
    assert rgb.shape[-1] == 3, "Input image must have 3 color channels"

    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]) # coefs according to MATLAB function

def calc_fft_index(img, n_times, angle_interval, return_raw=False):
    '''
    Calculate the frequency index of the FFT of an image.

    args:
    img(np.array): the image to calculate the FFT index
    n_times(int): upsampling factor for averaging
    angle_interval(int): the angle interval for azimuthal averaging
    return_raw(bool): if True, return the raw indices of the FFT as well
    '''
    # Arg check
    assert isinstance(img, np.ndarray), "Image must be a numpy array"
    assert img.shape[0] == img.shape[1], "Image must be square"
    assert n_times > 0, "n_times must be greater than 0"
    assert angle_interval > 0, "angle_interval must be greater than 0"
    assert isinstance(return_raw, bool), "return_raw must be a boolean"

    # Convert to grayscale if necessary
    if len(img.shape) == 3:
        img = rgb2gray(img)
    gray_fft = np.fft.fftshift(np.fft.fft2(img))
    abs_spectrum = np.abs(gray_fft)
    half_size = img.shape[0] // 2
    if img.shape[0] % 2 == 0:
        abs_spectrum[half_size - 1:half_size + 1, half_size - 1:half_size + 1] = 0
    else:
        abs_spectrum[half_size, half_size] = 0
    r_index, r_means, a_index, a_means = fft_average(abs_spectrum, n_times, angle_interval)
    
    # weighted average of frequency
    r_means = r_means / np.sum(r_means)
    peak_freq = np.sum(r_means*r_index)/np.sum(r_means)
    peak_freq = peak_freq / n_times

    # mean resultant length
    mrl = mean_resultant_length(a_index, a_means)
    
    if return_raw:
        return peak_freq, mrl, r_index, r_means, a_index, a_means
    
    return peak_freq, mrl

def calc_rf_indices(weights, n_top_col_pixel=48, n_times=100, angle_interval=1, return_rank=False, color_only=False):
    '''
    Calculate the indices of the receptor fields (first conv layer weights) of the network.

    args:
    weights(np.array): the weights of the first conv layer
    n_top_col_pixel(int): the number of top color pixels to calculate the color index.
                          if n_top_col_pixel=0, the mean of all pixels is returned
    n_times(int): upsampling factor for averaging
    angle_interval(int): the angle interval for azimuthal averaging
    return_rank(bool): if True, return the rank of the receptor fields (argsort, from smallest to largest)
    color_only(bool): if True, only the color index is returned
    '''
    # Arg check
    assert n_top_col_pixel >= 0, "n_top_col_pixel must be greater than or equal to 0"
    assert n_times > 0, "n_times must be greater than 0"
    assert angle_interval > 0, "angle_interval must be greater than 0"
    assert isinstance(return_rank, bool), "return_rank must be a boolean"
    assert len(weights.shape) == 4, "Weights must be 4D"
    assert weights.shape[0] == weights.shape[1], "Weights must be square"

    weights_norm = np.zeros(weights.shape)
    num_rf = weights.shape[3]
    for i in range(num_rf):
        weights_norm[:,:,:,i] = normalize(weights[:,:,:,i])

    # Color index
    color_index = np.array([calc_color_index(weights_norm[:,:,:,i], n_top_col_pixel) for i in range(num_rf)])

    if color_only:
        if return_rank:
            return np.argsort(color_index)
        else:
            return color_index

    # FFT index
    fft_freq_index = []
    fft_az_index = []
    for i in range(num_rf):
        peak_freq, mrl = calc_fft_index(weights_norm[:,:,:,i], n_times, angle_interval)
        fft_freq_index.append(peak_freq)
        fft_az_index.append(mrl)

    fft_freq_index = np.array(fft_freq_index)
    fft_az_index = np.array(fft_az_index)

    if return_rank:
        color_rank = np.argsort(color_index)
        fft_freq_rank = np.argsort(fft_freq_index)
        fft_az_rank = np.argsort(fft_az_index)
        return color_rank, fft_freq_rank, fft_az_rank
    
    return color_index, fft_freq_index, fft_az_index
