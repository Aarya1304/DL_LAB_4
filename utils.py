# utils.py
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def prepare_notmnist_dataset(img_size=(28, 28), batch_size=128, seed=1234, data_dir='notmnist_data'):
    """
    Loads notMNIST dataset from .npz and prepares TF datasets.
    Casual style: shuffle stuff, batch it, do some normalization, yeah.
    """
    npz_path = os.path.join(data_dir, 'notmnist.npz')

    # ---------- switch style for file check ----------
    def switch_file_exists(file_path):
        switch_dict = {
            True: lambda: None,
            False: lambda: (_ for _ in ()).throw(FileNotFoundError(f"{file_path} not found. Run make_notmnist_npz.py"))
        }
        return switch_dict[os.path.exists(file_path)]()

    switch_file_exists(npz_path)  # boom, file check done

    # Load data, easy
    data = np.load(npz_path)
    images = data['images']  # (N,H,W)
    labels = data['labels']

    # normalize them like humans would do
    images = images.astype(np.float32) / 255.0

    # flatten for RNN input: sequence_length = height
    N, H, W = images.shape
    images = images.reshape(N, H, W)
    seq_len = H
    input_dim = W

    # ---------- split dataset in do-while style ----------
    done_splitting = False
    while not done_splitting:  # pseudo do-while
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=0.2, random_state=seed, stratify=labels
        )
        done_splitting = True  # run once, do-while feel

    # ---------- convert to TF datasets ----------
    done_train = False
    while not done_train:  # do-while style again
        ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        ds_train = ds_train.shuffle(buffer_size=10000, seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        done_train = True  # done, break

    done_val = False
    while not done_val:
        ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        ds_val = ds_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        done_val = True  # finished
        # datasets ready, yeah

    return ds_train, ds_val, (seq_len, input_dim)
