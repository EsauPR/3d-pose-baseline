import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

from top_vae_3d_pose.args_def import ENVIRON as ENV

class Dataset:
    def __init__(self, n=5):
        self.i = 0
        self.n = n
        self.data = np.zeros(3)

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.n:
            i = self.i
            self.i += 1
            return self.data + self.i, i
        else:
            raise StopIteration()


@tf.function
def main():
    checkpoint_path = "experiments/All/dropout_0.5/epochs_200/lr_0.001/residual/depth_2/linear_size1024/batch_size_64/no_procrustes/maxnorm/batch_normalization/use_stacked_hourglass/predict_17/checkpoint-4874200"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    print(checkpoint_dir)

    dataset = tf.data.Dataset.from_generator(Dataset, (tf.float32, tf.int16))

    epochs = 10

    #dataset = Dataset(5)
    for e in range(3):
        for x in dataset:
            print(e, x)



if __name__ == "__main__":
    ENV.setup()
    main()
