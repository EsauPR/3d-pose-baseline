""" Functions to convert joints to bones and viceversa """

import yaml

import tensorflow as tf
import numpy as np

from top_vae_3d_pose.args_def import ENVIRON as ENV



def get_bones_mapping():
    """ Return the mapping for the bones arquitecture """
    with open(ENV.FLAGS.bones_mapping_dir) as fyml:
        bones_mapping = yaml.load(fyml, Loader=yaml.FullLoader)
    return bones_mapping


def get_bones_joints(bones_mapping):
    """ Returns two arrays that contains the joint that belongs to the start and end of the bone """
    bones = []

    def bone_walk(mapping):
        for _, infop in mapping.items():
            next_joints = infop['next']
            if next_joints is None:
                continue

            c_id = infop['joint_id_17']
            for bmapn in next_joints:
                for _, infoc in bmapn.items():
                    n_id = infoc['joint_id_17']
                    bones.append((c_id, n_id))
                    bone_walk(bmapn)

    bone_walk(bones_mapping['bones_path'][0])

    bja, bjb = zip(*bones)
    # Sumamos 1, porque agregaremos al inicio la articulación
    # de la pelvis o cadera [0, 0, 0] para facilitar las operaciones
    bja = np.array(bja) + 1
    bjb = np.array(bjb) + 1

    return bja, bjb


def get_bones_vectors_tf(map_i, map_j, joints):
    """ Return the matrix with the vectors of each bone
        This functions is necessary because tf does not support
        array indexing like numpy
    """
    # Reshape to [num samples, joint, coordinates[x,y,z]]
    joints = tf.reshape(joints, (joints.shape[0], -1, 3))

    # Compute the direction matrix for each sample by the map_i and map_j bones mapping
    # [num sample, direction[x,y,z], bone]
    all_b_v = []
    for k in range(joints.shape[0]):
        bones_vectors = []
        for idx in range(map_i.shape[0]):
            i, j = map_i[idx], map_j[idx]
            if i == 0: # Hip, this joint is the origin and this is not predicted
                bones_vectors.append(joints[k][j-1])
                continue
            bones_vectors.append(joints[k][j-1] - joints[k][i-1])
        all_b_v.append(tf.stack(bones_vectors))

    return tf.stack(all_b_v)


def convert_to_bones_tf(joints, bones_joints):
    """ Transform the joints to direction cosines and magnitudes """
    joints = tf.reshape(joints, (joints.shape[0], -1, 3))
    map_i = np.array([ 0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  9, 14, 15,  9, 11, 12])
    map_j = np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 14, 15, 16, 11, 12, 13])

    # Compute the direction matrix for each sample by the map_i and map_j bones mapping
    # [num sample, direction[x,y,z], bone]
    all_b_v = []
    for k in range(joints.shape[0]):
        bv = []
        for idx in range(map_i.shape[0]):
            i, j = map_i[idx], map_j[idx]
            if i == 0: # Hip, this joint is the origin and this is not predicted
                bv.append(joints[k][j-1])
                continue
            bv.append(joints[k][j-1] - joints[k][i-1])
        all_b_v.append(tf.stack(bv))

    # bones vector with origin in (0, 0, 0)
    bones_vectors = tf.stack(all_b_v)

    bones_magnitudes = tf.sqrt(tf.reduce_sum(tf.square(bones_vectors), axis=2))

    # derection cosines
    bones_dir_cos = bones_vectors / tf.stack([bones_magnitudes, bones_magnitudes, bones_magnitudes], axis=2)

    return bones_magnitudes, tf.reshape(bones_dir_cos, (bones_dir_cos.shape[0], -1))


def convert_to_bones(joints, bones_joints):
    """ Transform the joints to direction cosines and magnitudes """
    bja, bjb = bones_joints
    # Add hip as zeros array because this is the center
    hip_joints = np.zeros((joints.shape[0], 3))
    joints = np.concatenate([hip_joints, joints], axis=1)
    joints = joints.reshape(joints.shape[0], -1, 3)
    # joints.shape

    # bones vector with origin in (0, 0, 0)
    bones_vectors = joints[:, bjb] - joints[:, bja]
    # bones_vectors.shape

    bones_magnitudes = np.sqrt(np.sum(bones_vectors **2, axis=2))
    # bones_magnitudes.shape

    # derection cosines
    bones_dir_cos = bones_vectors / np.stack([bones_magnitudes, bones_magnitudes, bones_magnitudes], axis=2)
    # bones_dir_cos.shape

    return bones_magnitudes.astype('float32'), bones_dir_cos.reshape(bones_dir_cos.shape[0], -1).astype('float32')


def convert_to_joints(bones_mapping, magnitudes, dir_cos):
    """ Transform the magnitudes and direction cosines to points joints """
    joints = np.zeros(dir_cos.shape)

    start = np.zeros((magnitudes.shape[0], 3))

    # magnitudes = np.stack([magnitudes, magnitudes, magnitudes], axis=2)

    def bone_walk(mapping, start, index=0):
        for _, infop in mapping.items():
            next_joints = infop['next']
            if next_joints is None:
                continue

            for bmapn in next_joints:
                for _, infoc in bmapn.items():
                    n_id = infoc['joint_id_17']

                    mgs = magnitudes[:, index]
                    dcos = dir_cos[:, index]

                    ssum = start + dcos * np.stack([mgs, mgs, mgs], axis=1)
                    joints[:, n_id] = ssum

                    index = bone_walk(bmapn, ssum, index=index+1)
        return index

    bone_walk(bones_mapping['bones_path'][0], start)

    return joints.reshape(joints.shape[0], -1)
