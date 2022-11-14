import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pylab as plt
import math
import os

class TFRecord_Reader:
  def __init__(self, tfRecord_folder_path):
    self.tfRecord_folder_path = tfRecord_folder_path
    file_paths = os.listdir(tfRecord_folder_path)
    self.file_paths = file_paths
    self.featrue_description = {
        'image': tf.io.FixedLenFeature([], dtype = tf.string),
        'label': tf.io.FixedLenFeature([], tf.string)
    }

    def _parse_image_function(patient_voxels_bytes):
      return tf.io.parse_single_example(patient_voxels_bytes, self.image_feature_description)
    
    def decode_bytes_to_ndarray_float_tensor(voxels_dict):
        decoded_label = tf.io.parse_tensor(voxels_dict.get("label"), out_type=tf.float32)
        decoded_voxel = tf.io.parse_tensor(voxels_dict.get("image"), out_type=tf.float32)
        return decoded_voxel, decoded_label
    
    def read():
        raw_image_dataset = tf.data.TFRecordDataset(self.file_paths)
        parsed_voxels_dataset = raw_image_dataset.map(_parse_image_function)
        decoded_voxels_dataset = parsed_voxels_dataset.map(decode_bytes_to_ndarray_float_tensor)
        return decoded_voxels_dataset


tfRecord_folder_path = r'D:\MachineLearning\Projects\Kaggle\RSNA_Cervical_Spine_Fracture_Detection\Voxels_TFRecords'
debug = True

tFRecordReader = TFRecord_Reader(tfRecord_folder_path)
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