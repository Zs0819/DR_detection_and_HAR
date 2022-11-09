import pandas as pd
import os
from glob import glob
import numpy as np
import gin

# basepath = '/content/drive/My Drive/HAPT_dataset'  # location on local memory
# basepath = '/misc/home/data/HAPT_dataset'  # location on dl-lab server
basepath = './datasets/HAPT_dataset'  # location on final version code


def file_to_dataframe(filepath, names):
    """read a file and return table"""
    dataframe = pd.read_table(filepath, names=names, delim_whitespace=True)
    return dataframe


label_path = os.path.join(basepath, "RawData", "labels.txt")
label_df = file_to_dataframe(filepath=label_path, names=["experiment", "userid", "activity", "start_pos", "end_pos"])


# print(label_df)

def collect_accfilelist_per_user(userid):
    """ return a list of all files according to a certain user id"""
    if userid < 10:
        userid = '0' + str(userid)
    else:
        userid = str(userid)
    acc_path = "acc*user" + userid + ".txt"
    acc_files = glob(os.path.join(basepath, "RawData", acc_path))
    acc_files.sort()
    # print(acc_files)
    return acc_files


def create_df(acc_filepath):
    """input is acc_file, return a dataframe includes a_x, a_y, a_z ,w_x, w_y, w_z and label"""
    basep = os.path.basename(acc_filepath)
    # print(basep)
    experiment = int(basep[7:9])
    userid = int(basep[14:16])
    acc_file = acc_filepath
    gyro_file = acc_filepath.replace("acc", "gyro")
    acc_df = file_to_dataframe(filepath=acc_file, names=["a_x", "a_y", "a_z"])
    gyro_df = file_to_dataframe(filepath=gyro_file, names=["w_x", "w_y", "w_z"])

    label_list = pd.DataFrame(0, index=np.arange(acc_df.shape[0]), columns=["label"])
    label_info = label_df[
        (label_df["experiment"] == int(experiment)) & (label_df["userid"] == int(userid))]

    for index in label_info.index:
        start = label_info["start_pos"][index]
        end = label_info["end_pos"][index]
        label_list.loc[start - 1:end - 1, "label"] = label_info["activity"][index]
    df_perExp_perUser = pd.concat([acc_df, gyro_df, label_list], axis=1)

    return df_perExp_perUser


def z_score(df):
    """Z-score normalization"""
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()

    return df_std


# def create_dataset(start_user, end_user, shift_window_size=125, window_size=250):
#   ds_total = pd.DataFrame()
#   for user in range(start_user,end_user+1):
#     for accfile in collect_accfilelist_per_user(user):
#       # print(accfile)
#       df = create_df(accfile)
#       length = len(df)
#       del_len = (length-window_size)%shift_window_size
#       df = df[:-del_len]
#       ds_total = pd.concat([ds_total,df] ,axis=0, ignore_index=True)
#   ds_total = ds_total.drop(ds_total[ds_total.label == 0].index)
#   ds_x = ds_total[["a_x", "a_y", "a_z", "w_x", "w_y", "w_z"]]
#   ds_y = ds_total[["label"]]
#   ds_x = z_score(ds_x)
#   ds = tf.data.Dataset.from_tensor_slices((ds_x, ds_y)).window(size=window_size, shift=shift_window_size,drop_remainder=True)
#   ds = ds.flat_map(lambda data, label: tf.data.Dataset.zip((data, label))).batch(window_size,drop_remainder=True)

#   return ds


# train_ds = create_dataset(start_user=1, end_user=21)
# val_ds =create_dataset(start_user=28, end_user=30)
# test_ds = create_dataset(start_user=22, end_user=27)

# for x,y in train_ds.take(1):
#   print(x,y)

@gin.configurable
def create_dataset(shift_window_ratio, window_size, start_user=1, end_user=21, return_mode='all'):
    shift_window_size = int(shift_window_ratio*window_size)
    if return_mode == 'params':
        return window_size, shift_window_size
    else:
        ds_total = pd.DataFrame()
        for user in range(start_user, end_user + 1):
            for accfile in collect_accfilelist_per_user(user):
                # print(accfile)
                df = create_df(accfile)
                length = len(df)
                del_len = (length - window_size) % shift_window_size
                df = df[:-del_len]
                ds_total = pd.concat([ds_total, df], axis=0, ignore_index=True)
        ds_total = ds_total.drop(ds_total[ds_total.label == 0].index)
        ds_x = ds_total[["a_x", "a_y", "a_z", "w_x", "w_y", "w_z"]]
        ds_y = ds_total[["label"]]
        ds_x = z_score(ds_x)

        return ds_x, ds_y, window_size, shift_window_size





# def get_window_size():
#     return WindowSize


# for i in range(5):
#   data = train_x.iloc[i].values
#   data = tf.convert_to_tensor(data, dtype=tf.float32)
#   print(data)
#   label = train_y.iloc[i].values
#   label = tf.convert_to_tensor(label, dtype=tf.int64)
#   print(label)
# print(train_x.index)


