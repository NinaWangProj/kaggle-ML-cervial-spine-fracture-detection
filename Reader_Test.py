import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pylab as plt
import math
import os
import TFRecord_Reader


tfRecord_folder_path = r'D:\MachineLearning\Projects\Kaggle\RSNA_Cervical_Spine_Fracture_Detection\Voxels_TFRecords'
debug = False

tFRecordReader = TFRecord_Reader.TFRecord_Reader(tfRecord_folder_path)
decoded_voxels_dataset = tFRecordReader.read()

#visualize loaded images
if (debug):
    for voxels, label in decoded_voxels_dataset:
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        for i in range(50):
            first_slice = voxels[:,:,i,0]
            test = first_slice.numpy()
            slice_num = i
            plt.title('interpolated' + str(slice_num))
            plt.imshow(test)
            plt.show()
            
            image_file_folder = r'D:\\MachineLearning\\Projects\\Kaggle\\RSNA_Cervical_Spine_Fracture_Detection\\Zip\\rsna-2022-cervical-spine-fracture-detection\\train_images\\1.2.826.0.1.3680043.22517\\'
            file_name = str(math.floor(i*5.53061224+1))+'.dcm'
            file_path = os.path.join(image_file_folder, file_name)
            byte_string = tf.io.read_file(file_path)
            
            image = tfio.image.decode_dicom_image(byte_string, dtype=tf.uint32)
            if (tf.equal(tf.size(image), 0)):
                image = tfio.image.decode_dicom_image(byte_string, dtype=tf.uint32)
        
            image = image[0,:,:,0]
            plt.title('original' + file_name)
            plt.imshow(image)
            plt.show()