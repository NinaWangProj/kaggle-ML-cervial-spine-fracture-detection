##Datasets approach when preparing input data
import tensorflow as tf
import tensorflow_io as tfio
from scipy.ndimage.interpolation import zoom
import tensorflow_probability as tfp
import matplotlib.pylab as plt
import math
import os


##Prepare training data
class Data_Pipeline():
    def __init__(self, voxel_shape, batch_size, debug, visualize_slices):
        self.voxel_shape = voxel_shape
        self.batch_size = batch_size
        self.debug = debug
        self.visualize_slices = visualize_slices
        
    def _load_prepare_voxels(self, patient_uid, y_train):
        dataset_per_patient = tf.data.Dataset.from_tensor_slices(patient_uid)
        dataset_per_patient = dataset_per_patient.map(self.read_file)
        dataset_per_patient = dataset_per_patient.reduce(tf.zeros([1,512,512,1],dtype=tf.dtypes.float32),lambda x, y: tf.concat([x, y], axis=-1))
        tf.print('finished concatinating :')
        tf.print(dataset_per_patient)
        tf.print(dataset_per_patient.shape)
        tf.print(patient_uid)
        dataset_per_patient = dataset_per_patient[0,:,:,:]
        patient_voxels = dataset_per_patient[:,:,1:]
        return patient_voxels, y_train
    
    def read_file(self, file_path):
        byte_string = tf.io.read_file(file_path)
        byte_flat = tf.strings.length(byte_string, unit='BYTE', name=None)
        if(self.debug): 
            tf.print(file_path)
            tf.print('number of bytes read:')
            tf.print(byte_flat)

        image = tfio.image.decode_dicom_image(byte_string, dtype=tf.uint16)
        image = tf.cast(image, tf.float32, name=None)
        
        if (tf.equal(tf.size(image), 0)):
            if(self.debug):
                tf.print('entering into 32 bit block!!!')
            image = tfio.image.decode_dicom_image(byte_string, dtype=tf.uint32)
            image = tf.cast(image, tf.float32, name=None)
        if(self.debug):
            tf.print('file read completed with image size ' + str(image.shape))
        return image
    
    def interpolate(self, patient_voxels, y_train):
        x_ref_min = [0,0,0]
        x_ref_max = [tf.math.subtract(tf.shape(patient_voxels)[0],1), tf.math.subtract(tf.shape(patient_voxels)[1],1),
                     tf.math.subtract(tf.shape(patient_voxels)[2],1)]
        y_ref = patient_voxels
        
        x1, x2, x3 = tf.meshgrid(tf.linspace(x_ref_min[0], x_ref_max[0], num=self.voxel_shape[0]),
                                tf.linspace(x_ref_min[1], x_ref_max[1], num=self.voxel_shape[1]),
                                tf.linspace(x_ref_min[2], x_ref_max[2], num=self.voxel_shape[2]),indexing='ij')
        
        x1_reshaped = tf.reshape(x1, [-1])
        x2_reshaped = tf.reshape(x2, [-1])
        x3_reshaped = tf.reshape(x3, [-1])
        
        x = tf.stack([x1_reshaped, x2_reshaped, x3_reshaped], axis = -1)
        x = tf.cast(x, tf.float32, name=None)
        x_ref_max = tf.cast(x_ref_max, tf.float32, name=None)
    
        y_interpolated = tfp.math.batch_interp_regular_nd_grid(
            # x.shape = [3, 1], x_ref_min/max.shape = [1].  Trailing `1` for `1-D`.
            x, x_ref_min, x_ref_max, y_ref=y_ref,
            axis=0)
        
        y_reshaped = tf.reshape(y_interpolated, self.voxel_shape)
        patient_resized_voxels = tf.expand_dims(y_reshaped, -1)
        
        if(self.visualize_slices):
            ## visualize a slice 1.dcm
            os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
            for i in range(50):
                first_slice = patient_resized_voxels[:,:,i,0]
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
        
        tf.print('Finished interpolating for current patient.')
        return patient_resized_voxels, y_train
        
    def interpolate_using_zoom(self, patient_voxels, y_train, voxel_shape):
        x_dimension = tf.math.divide(voxel_shape[0], patient_voxels.shape[0])
        y_dimension = tf.math.divide(voxel_shape[1], patient_voxels.shape[1])
        z_dimension = tf.math.divide(voxel_shape[2], patient_voxels.numpy().shape[2])
        interpolated_voxels = tf.py_function(func=self.zoom, inp=[patient_voxels, x_dimension, y_dimension, z_dimension,], Tout=[tf.uint16])
        tf.print(interpolated_voxels[0])
        tf.print(interpolated_voxels[0].shape)
        #interpolated_voxels = self.zoom(patient_voxels, x_dimension, y_dimension, z_dimension)
        return interpolated_voxels[0], y_train
        
    def zoom(self, patient_voxels, x_dimension, y_dimension, z_dimension):
        return zoom(input=patient_voxels.numpy(), zoom=[x_dimension.numpy(),y_dimension.numpy(),z_dimension.numpy(), 1])
    
    def test(self, patient_voxels, y_train, voxel_shape):
        x_dimension = tf.math.divide(voxel_shape[0], patient_voxels.shape[1])
        y_dimension = tf.math.divide(voxel_shape[1], patient_voxels.shape[2])
        z_dimension = tf.math.divide(voxel_shape[2], patient_voxels.numpy().shape[3])
        return x_dimension, y_dimension, z_dimension
    
    def set_shape(self, patient_voxels, y_train):
        patient_voxels.set_shape((512,512,50,1))
        y_train.set_shape((8,))
        y_train = tf.cast(y_train, tf.float32, name=None)
        return patient_voxels, y_train

    def get_dataset(self, patient_uid_array, y_train_array):
        dataset = tf.data.Dataset.from_tensor_slices((patient_uid_array, y_train_array))
        dataset = dataset.map(self._load_prepare_voxels)
        #dataset = dataset.map(lambda x,y : tf.py_function(self.interpolate, inp=[x,y,voxel_shape], Tout=[tf.uint16, tf.int64]))
        dataset = dataset.map(self.interpolate)
        dataset = dataset.map(self.set_shape)
        #dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
