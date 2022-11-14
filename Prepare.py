##train test split
import pandas as pd
import os
import tensorflow as tf

def target_train_test_split(target_path, train_factor, test_factor):
    target_df = pd.read_csv(target_path)
    train_target_df= target_df.sample(frac = train_factor, random_state = 100)
    #train_target_df = target_df[target_df['StudyInstanceUID']=='1.2.826.0.1.3680043.19882']
    test_target_df = target_df.drop(train_target_df.index).sample(frac = test_factor, random_state = 100)
    train_target_df.reset_index()
    test_target_df.reset_index()
    y_train_df = train_target_df.drop(columns="StudyInstanceUID")
    y_train = y_train_df.to_numpy()
    y_test_df = test_target_df.drop(columns = "StudyInstanceUID")
    y_test = y_test_df.to_numpy()
    uid_train = train_target_df['StudyInstanceUID'].to_numpy()
    uid_test = test_target_df['StudyInstanceUID'].to_numpy()
    return uid_train, uid_test, y_train, y_test


def prepare_patient_images_path(uid_train, train_folder_dir):
    patients_images_full_path=[]
    for x in uid_train:
        patient_directory = os.path.join(train_folder_dir,x)
        patient_file_names = os.listdir(patient_directory)
        patient_file_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        patient_full_paths = []
        for path in patient_file_names:
            patient_full_paths.append(os.path.join(patient_directory,path))
        patients_images_full_path.append(patient_full_paths)
    patients_images_full_path = tf.ragged.constant(patients_images_full_path) 
    return patients_images_full_path