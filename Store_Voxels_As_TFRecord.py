##Prepare patient voxels and store in TFRecord
import Data_Pipeline
import Prepare
import tensorflow as tf


##Input files and folder names
#tf.data.experimental.enable_debug_mode()
target_path = r'D:\MachineLearning\Projects\Kaggle\RSNA_Cervical_Spine_Fracture_Detection\Zip\rsna-2022-cervical-spine-fracture-detection\train.csv'
train_folder_dir = r'D:\MachineLearning\Projects\Kaggle\RSNA_Cervical_Spine_Fracture_Detection\Zip\rsna-2022-cervical-spine-fracture-detection\train_images'
voxel_shape = (512, 512, 50)
batch_size = 2
input_shape_batch = (512, 512, 50, 1)
debug = False
visualize_slices = False
tfRecord_folder_path = r'D:\MachineLearning\Projects\Kaggle\RSNA_Cervical_Spine_Fracture_Detection\Voxels_TFRecords\Test'

##Split target value for train and test & prepare image full paths for patients
uid_train, uid_test, y_train, y_test = Prepare.target_train_test_split(target_path, 0.8, 1)
patients_images_full_path_train = Prepare.prepare_patient_images_path(uid_train, train_folder_dir)
patients_images_full_path_test = Prepare.prepare_patient_images_path(uid_test, train_folder_dir)

##Prepare input data
pipeline = Data_Pipeline.Data_Pipeline(voxel_shape, batch_size, debug, visualize_slices)
#train_dataset = pipeline.get_dataset(patients_images_full_path_train, y_train)
test_dataset = pipeline.get_dataset(patients_images_full_path_test, y_test)


##Write out dataset as TFRecord
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy()]))

def patient_voxels(image, label):
  feature = {
      'image': _bytes_feature(image),
      'label': _bytes_feature(label)
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))

record_file = 'images.tfrecords'
record_file= tf.io.gfile.join(tfRecord_folder_path, record_file)
count = 0
with tf.io.TFRecordWriter(record_file) as writer:
  for voxels, label in test_dataset:
      voxels_byte_string = tf.io.serialize_tensor(voxels)
      label_byte_string = tf.io.serialize_tensor(label)
      tf_example = patient_voxels(voxels_byte_string, label_byte_string)
      writer.write(tf_example.SerializeToString())
      count +=1 
      tf.print('finished writing ' + str(count) + ' files.')
tf.print('finished serializing all records.')








