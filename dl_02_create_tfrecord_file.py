#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compile the data into TFRecord files, which can be used for training
the deep learning models.
"""

import mneflow
import numpy as np

from config import analysis_root, data_ids
from filenames import FileNames


fn = FileNames(analysis_root)
output_folder = fn.dl_dir
labels = np.load(fn.ages_all_subjects)

for data_id in data_ids:
    print("Making TFRecord file for data_id:", data_id)
    sensor_space = True if data_id.startswith("sensor") else False
    scale = True if "minmax_scaled" in data_id else False
    if "wpli_wpli" in data_id:
        lower_triangle = "wpli"
        upper_triangle = "wpli"
    elif "aec_wpli" in data_id:
        lower_triangle = "aec"
        upper_triangle = "wpli"
    elif "aec_aec" in data_id:
        lower_triangle = "aec"
        upper_triangle = "aec"
    else:
        raise ValueError(f"Invalid data_id: {data_id}")

    data = np.load(
        fn.fc_data(
            lower_triangle,
            upper_triangle,
            minmax=scale,
            sensor_space=sensor_space)
    )
    # Create or load tfrecord file.
    import_opt = dict(
        path=fn.tfr_dir,  # path where TFR files will be saved
        data_id=data_id,  # name of TFRecords files
        fs=1,
        input_type='fconn',
        target_type='float',
        scale=False,  # apply baseline_scaling
        crop_baseline=False,  # remove baseline interval after scaling
        n_folds=12,  # validation set size set to 20% of all data
        overwrite=False,
        segment=False,
        test_set=None)

    if data_id == data_ids[0]:
        meta = mneflow.produce_tfrecords((data, labels), **import_opt)
    else:
        # Use the same split for all datasets
        import_opt_0 = import_opt.copy()
        import_opt_0['data_id'] = data_ids[0]
        meta = mneflow.produce_tfrecords(None, **import_opt_0)
        meta = mneflow.produce_tfrecords((data, labels),
                                         predefined_split=meta.data['folds'],
                                         **import_opt)
    print("TFRecord file created for data_id:", data_id)
