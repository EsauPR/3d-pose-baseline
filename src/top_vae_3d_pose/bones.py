""" Functions to convert joints to bones and viceversa """

import yaml

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

    return bones_magnitudes, bones_dir_cos


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

    return joints
