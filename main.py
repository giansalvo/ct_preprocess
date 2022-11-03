# adapted from Vicente Rodríguez https://vincentblog.xyz/posts/medical-images-in-python-computed-tomography
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

import pydicom
import cv2
import os
from skimage import morphology
from scipy import ndimage


def plot_samples_matplotlib(display_list, labels_list=None, figsize=None, fname=None):
    NIMG_PER_COLS = 6
    if figsize is None:
        PX = 1/plt.rcParams['figure.dpi']  # pixel in inches
        figsize = (600*PX, 300*PX)
    ntot = len(display_list)
    if ntot <= NIMG_PER_COLS:
        nrows = 1
        ncols = ntot
    elif ntot % NIMG_PER_COLS == 0:
        nrows = ntot // NIMG_PER_COLS
        ncols = NIMG_PER_COLS
    else:
        nrows = ntot // NIMG_PER_COLS + 1
        ncols = NIMG_PER_COLS
    _, axes = plt.subplots(nrows, ncols, figsize=figsize)
    for i_img in range(ntot):
        i = i_img // NIMG_PER_COLS
        j = i_img % NIMG_PER_COLS
        if display_list[i_img].shape[-1] == 3:
                img = tf.keras.preprocessing.image.array_to_img(display_list[i_img])
                if nrows > 1:
                    if labels_list is not None:
                        axes[i, j].set_title(labels_list[i_img])
                    axes[i, j].imshow(img, cmap='Greys_r')
                else:
                    if labels_list is not None:
                        axes[i_img].set_title(labels_list[i_img])
                    axes[i_img].imshow(img, cmap='Greys_r')
        else:
                img = display_list[i_img]
                if nrows > 1:
                    if labels_list is not None:
                        axes[i, j].set_title(labels_list[i_img])
                    axes[i, j].imshow(img, cmap='Greys_r')
                else:
                    if labels_list is not None:
                        axes[i_img].set_title(labels_list[i_img])
                    axes[i_img].imshow(img, cmap='Greys_r')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for axes in axes.flat:
        axes.label_outer()

    if fname is None:
        plt.show()
    else:
        #logger.debug("Saving prediction to file {}...".format(fname))
        plt.savefig(fname)
        plt.close()


def display_views(file_path):
    image_axis = 2
    medical_image = nib.load(file_path)
    image = medical_image.get_data()

    sagital_image = image[110, :, :] # Axis 0
    axial_image = image[:, :, 144] # Axis 2
    coronal_image = image[:, 144, :] # Axis 1

    plt.figure(figsize=(20, 10))
    plt.style.use('grayscale')
    plt.subplot(141)
    plt.imshow(np.rot90(sagital_image))
    plt.title('Sagital Plane')
    plt.axis('off')
    plt.subplot(142)
    plt.imshow(np.rot90(axial_image))
    plt.title('Axial Plane')
    plt.axis('off')
    plt.subplot(143)
    plt.imshow(np.rot90(coronal_image))
    plt.title('Coronal Plane')
    plt.axis('off')


def display_views2(file_path):
    medical_image = nib.load(file_path)
    image = medical_image.get_fdata()

    plt.figure(figsize=(5, 5))
    plt.style.use('grayscale')
    plt.imshow(np.rot90(image, 3))
    plt.title('Plane')
    plt.axis('off')
    plt.show()


def display_image(img):
    plt.figure(figsize=(5, 5))
    plt.style.use('grayscale')
    plt.title('Image')
    plt.axis('off')
    plt.imshow(img)
    plt.show()


def transform_to_hu(medical_image, image):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept
    return hu_image


def window_image(image, img_min, img_max):
    #img_min = window_center - window_width // 2
    #img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    return window_image


# GIANS: this is specialized for brain tissue but it can be adapted
def remove_noise(file_path):
    medical_image = pydicom.read_file(file_path)
    image = medical_image.pixel_array
    hu_image = transform_to_hu(medical_image, image)
    lung_image = window_image(hu_image, 0, 80)

    # morphology.dilation creates a segmentation of the image
    # If one pixel is between the origin and the edge of a square of size
    # 5x5, the pixel belongs to the same class

    # We can instead use a circule using: morphology.disk(2)
    # In this case the pixel belongs to the same class if it's between the origin
    # and the radius

    segmentation = morphology.dilation(lung_image, np.ones((5, 5)))
    labels, label_nb = ndimage.label(segmentation)
    label_count = np.bincount(labels.ravel().astype(int))

    # The size of label_count is the number of classes/segmentations found
    # We don't use the first class since it's the background
    label_count[0] = 0

    # We create a mask with the class with more pixels
    # In this case should be the brain
    mask = labels == label_count.argmax()

    # Improve the brain mask
    mask = morphology.dilation(mask, np.ones((5, 5)))
    mask = ndimage.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3)))

    # Since the the pixels in the mask are zero's and one's
    # We can multiple the original image to only keep the brain region
    masked_image = mask * lung_image

    return masked_image, mask


def crop_image(image, display=False):
    # Create a mask with the background pixels
    mask = image == 0
    # Find the brain area
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)

    # Remove the background
    croped_image = image[top_left[0]:bottom_right[0],
                top_left[1]:bottom_right[1]]
    return croped_image


def add_pad(image, new_height=512, new_width=512):
    height, width = image.shape
    final_image = np.zeros((new_height, new_width))
    pad_left = int((new_width - width) / 2)
    pad_top = int((new_height - height) / 2)

    # Replace the pixels with the image's pixels
    final_image[pad_top:pad_top + height, pad_left:pad_left + width] = image
    return final_image


def resample(image, image_thickness, pixel_spacing):
    new_size = [1, 1, 1]
    x_pixel = float(pixel_spacing[0])
    y_pixel = float(pixel_spacing[1])
    size = np.array([x_pixel, y_pixel, float(image_thickness)])
    image_shape = np.array([image.shape[0], image.shape[1], 1])
    new_shape = image_shape * size
    new_shape = np.round(new_shape)
    resize_factor = new_shape / image_shape
    resampled_image = ndimage.zoom(np.expand_dims(image, axis=2), resize_factor)
    return resampled_image


def main():
    DCM_FILE_PATH = "images/1-040.dcm"
    DCM_SLICE1 = "images/1-040.dcm"
    DCM_SLICE2 = "images/1-041.dcm"

    # THIS IS OK
    # display_views2("images/slice001.nii.gz") # doesn't work with .dcm

    medical_image = pydicom.read_file(DCM_FILE_PATH)
    #print(medical_image)

    image = medical_image.pixel_array
    print("image.shape=" + str(image.shape))

    hu_image = transform_to_hu(medical_image, image)
    lung_image = window_image(hu_image, 0, 80)
    bone_image = window_image(hu_image, -100, 900)

    plot_samples_matplotlib([lung_image, bone_image], ["lung_image", "bone_image"])

    denoised_img, mask = remove_noise(DCM_FILE_PATH)
    plot_samples_matplotlib([image, mask, denoised_img], ["image", "mask", "denoised_img"])

    cropped_img = crop_image(denoised_img)
    padded_img = add_pad(cropped_img)
    plot_samples_matplotlib([denoised_img, cropped_img, padded_img], ["denoised_img", "cropped_img", "padded_img"])

    # RESAMPLING
    first_medical_image = pydicom.read_file(DCM_SLICE1)
    second_medical_image = pydicom.read_file(DCM_SLICE2)

    # we use the last axis
    image_thickness = np.abs(first_medical_image.ImagePositionPatient[2] - second_medical_image.ImagePositionPatient[2])
    print("image_thickness="+str(image_thickness))
    pixel_spacing = first_medical_image.PixelSpacing
    print("pixel_spacing="+str(pixel_spacing))
    first_image = first_medical_image.pixel_array
    resampled_image = resample(first_image, image_thickness, pixel_spacing)
    t = np.squeeze(resampled_image)
    print("resampled_image.shape="+str(t.shape))

    #plt.imshow(resampled_image)

    return

if __name__ == '__main__':
    main()