import tensorflow as tf
import os

class TFRecord_Reader:
    def __init__(self, tfRecord_folder_path, batch_size):
      self.tfRecord_folder_path = tfRecord_folder_path
      file_paths = os.listdir(tfRecord_folder_path)
      file_paths = list(map(lambda x: os.path.join(tfRecord_folder_path, x), file_paths))
      self.file_paths = file_paths
      self.batch_size = batch_size
      self.featrue_description = {
          'image': tf.io.FixedLenFeature([], dtype = tf.string),
          'label': tf.io.FixedLenFeature([], tf.string)
      }

    def _parse_image_function(self, patient_voxels_bytes):
      return tf.io.parse_single_example(patient_voxels_bytes, self.featrue_description)
    
    def decode_bytes_to_ndarray_float_tensor(self, voxels_dict):
        decoded_label = tf.io.parse_tensor(voxels_dict.get("label"), out_type=tf.float32)
        decoded_voxel = tf.io.parse_tensor(voxels_dict.get("image"), out_type=tf.float32)
        return decoded_voxel, decoded_label
    
    def set_shape(self, voxels, target):
        voxels.set_shape((self.batch_size,512,512,50,1))

        target.set_shape((self.batch_size,8))
        return voxels,target
    
    def read(self):
        raw_image_dataset = tf.data.TFRecordDataset(self.file_paths)
        parsed_voxels_dataset = raw_image_dataset.map(self._parse_image_function)
        decoded_voxels_dataset = parsed_voxels_dataset.map(self.decode_bytes_to_ndarray_float_tensor)
        dataset = decoded_voxels_dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.map(self.set_shape)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


