#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os.path
import re
from multiprocessing import Pool

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.paths import nnUNet_raw


def get_identifiers_from_splitted_dataset_folder(folder: str, file_ending: str):
    files = subfiles(folder, suffix=file_ending, join=False)
    # all files have a 4 digit channel index (_XXXX)
    crop = len(file_ending) + 5
    files = [i[:-crop] for i in files]
    # only unique image ids
    files = np.unique(files)
    return files


def create_paths_fn(folder, files, file_ending, f):
    p = re.compile(re.escape(f) + r"_\d\d\d\d" + re.escape(file_ending))            
    return [join(folder, i) for i in files if p.fullmatch(i)]


def create_lists_from_splitted_dataset_folder(folder: str, file_ending: str, identifiers: List[str] = None, num_processes: int = 12) -> List[
    List[str]]:
    """
    does not rely on dataset.json
    """
    if identifiers is None:
        identifiers = get_identifiers_from_splitted_dataset_folder(folder, file_ending)
    files = subfiles(folder, suffix=file_ending, join=False, sort=True)
    list_of_lists = []

    params_list = [(folder, files, file_ending, f) for f in identifiers]
    with Pool(processes=num_processes) as pool:
        list_of_lists = pool.starmap(create_paths_fn, params_list)
        
    return list_of_lists


def get_filenames_of_train_images_and_targets(raw_dataset_folder: str, dataset_json: dict = None):
    if dataset_json is None:
        dataset_json = load_json(join(raw_dataset_folder, 'dataset.json'))

    if 'dataset' in dataset_json.keys():
        dataset = dataset_json['dataset']
        for k in dataset.keys():
            expanded_label_file = os.path.expandvars(dataset[k]['label'])
            dataset[k]['label'] = os.path.abspath(join(raw_dataset_folder, expanded_label_file)) if not os.path.isabs(expanded_label_file) else expanded_label_file
            dataset[k]['images'] = [os.path.abspath(join(raw_dataset_folder, os.path.expandvars(i))) if not os.path.isabs(os.path.expandvars(i)) else os.path.expandvars(i) for i in dataset[k]['images']]
    else:
        identifiers = get_identifiers_from_splitted_dataset_folder(join(raw_dataset_folder, 'imagesTr'), dataset_json['file_ending'])
        images = create_lists_from_splitted_dataset_folder(join(raw_dataset_folder, 'imagesTr'), dataset_json['file_ending'], identifiers)
        segs = [join(raw_dataset_folder, 'labelsTr', i + dataset_json['file_ending']) for i in identifiers]
        dataset = {i: {'images': im, 'label': se} for i, im, se in zip(identifiers, images, segs)}
    return dataset


if __name__ == '__main__':
    print(get_filenames_of_train_images_and_targets(join(nnUNet_raw, 'Dataset002_Heart')))
