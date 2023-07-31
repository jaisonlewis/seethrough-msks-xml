import pandas as pd
import tensorflow as tf
import cv2
import os
import ast
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET

# Define directories and directory name for images
image_dir = 'nor_img'
mask_dir = 'norm_img_mask'
dirname = 'images2'

# Read the CSV file into a DataFrame
df = pd.read_csv('nor_img.csv')

# Extract filenames and annotations from the DataFrame
filenames = df['id'].values
annotations = df['annotations'].apply(ast.literal_eval).values

# Initialize a counter for naming the saved files
start_no = 0

# Loop through each image and its corresponding annotations
for filename, annotation in tqdm(zip(filenames, annotations)):

    # Load the large image using TensorFlow and convert it into a tensor
    large_image = tf.keras.preprocessing.image.load_img(os.path.join(dirname, filename + '.tif'))
    large_image = tf.keras.preprocessing.image.img_to_array(large_image)
    large_image = tf.reshape(large_image, [1, *large_image.shape])

    # Extract patches from the large image using TensorFlow's 'extract_patches' function
    patches = tf.image.extract_patches(
        images=large_image,
        sizes=[1, 512, 512, 1],
        strides=[1, 256, 256, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    patches = tf.reshape(patches, [-1, 512, 512, 3])

    # Loop through each patch and create masks based on the corresponding annotations
    for i in range(patches.shape[0]):
        im = np.asarray(patches[i, :, :, :]).astype('uint8')
        imname = os.path.join(image_dir, "{}{}.tif".format(filename, start_no + i))

        # Get the coordinates from the annotation
        coordinates = annotation[i]['coordinates']

        # Create a transparent mask with a black border
        mask = np.zeros_like(im, dtype=np.uint8)
        for coord_set in coordinates:
            points = np.array(coord_set, dtype=np.int32)
            cv2.fillPoly(mask, [points], (255, 255, 255))
            cv2.polylines(mask, [points], isClosed=True, color=(0, 0, 0), thickness=2)

        # Combine the mask with the original image
        masked_image = cv2.addWeighted(im, 1.0, mask, 0.7, 0)

        # Save the masked image
        im_masked_name = os.path.join(mask_dir, "{}{}.png".format(filename, start_no + i))
        im_masked = Image.fromarray(masked_image.astype(np.uint8))
        im_masked.save(im_masked_name)

        # Save the corresponding annotations as an XML file
        xml_name = os.path.join(mask_dir, "{}{}.xml".format(filename, start_no + i))
        root = ET.Element("annotation")
        object_node = ET.SubElement(root, "object")
        type_node = ET.SubElement(object_node, "type")
        type_node.text = annotation[i]['type']
        for coord_set in coordinates:
            points = np.array(coord_set, dtype=np.int32)
            bndbox_node = ET.SubElement(object_node, "bndbox")
            xmin_node = ET.SubElement(bndbox_node, "xmin")
            ymin_node = ET.SubElement(bndbox_node, "ymin")
            xmax_node = ET.SubElement(bndbox_node, "xmax")
            ymax_node = ET.SubElement(bndbox_node, "ymax")
            xmin_node.text = str(np.min(points[:, 0]))
            ymin_node.text = str(np.min(points[:, 1]))
            xmax_node.text = str(np.max(points[:, 0]))
            ymax_node.text = str(np.max(points[:, 1]))
        tree = ET.ElementTree(root)
        tree.write(xml_name)

    # Update the counter for naming the saved files
    start_no = start_no + patches.shape[0]
