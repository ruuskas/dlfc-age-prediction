"""Run the DL models for a given set of hyperparameters and data."""

import argparse
import json
import time

import mneflow
from mneflow.fc_models import Conv3DModel, WeightedSum3dModel, SymmetricModel
import tensorflow as tf
from numpy import random

from config import analysis_root, models, data_ids, subjects_dir
from filenames import FileNames

parser = argparse.ArgumentParser(description="Run DL models for a given set of "
                                 "hyperparameters and data.")
parser.add_argument('model_name', type=str,
                    help="Name of the model to run (e.g., 'm3_conv3d', "
                    "m2_sym3d', 'm1_depthwise).")
parser.add_argument('dataset_idx', type=int,
                    help="Index of the dataset to use")
parser.add_argument('run_idx', type=int,
                    help="Index of the run to execute (for different random "
                         "seeds)")
parser.add_argument('--no-train', dest='train', action='store_false',
                    default=True)
parser.add_argument('--ablation', dest='ablation', action='store_true',
                    default=False)
parser.add_argument('--visualize', dest='visualize', action='store_true',
                    default=False)

args = parser.parse_args()

time.sleep(random.uniform(0, 10))  # add random delay to avoid race conditions when multiple runs start at the same time
model_name = args.model_name
dataset_idx = args.dataset_idx
run_idx = args.run_idx
train = args.train
if model_name not in models:
    raise ValueError(f"Invalid model name: {model_name}. Must be one of: "
                     f"{models}")

# Load hyperparameters from JSON file
fn = FileNames(analysis_root)
hyperparam_file = fn.hyperparam_file(model_name)
with open(hyperparam_file, 'r') as f:
    hyperparameters = json.load(f)

# Select the hyperparameters for the specified run
hyperparameters = hyperparameters[run_idx]
if hyperparameters['model_id'] != model_name:
    raise ValueError(f"Model ID in hyperparameters "
                     f"({hyperparameters['model_id']}) "
                     f"does not match the specified model name ({model_name}).")
params = dict(
    nonlin=tf.nn.relu,
    stride=1,
    l1_scope=[],
    l2_scope=[],
    l2_lambda=0.,
    model_id=model_name,
    trainable_pointwise_kernel=False
)
hyperparameters.update(params)

# The following hyperparameters are specific to sensor space data
if "sensor" in data_ids[dataset_idx]:
    hyperparameters.update(dict(
        stddev=0.15,
        dropout=0.2
    ))

print(f"Training with hyperparameters: {hyperparameters}")

# Import data and labels from TFRecord files
import_opt = dict(
    path=fn.tfr_dir,
    data_id=data_ids[dataset_idx],  # name of TFRecords files
    fs=1,
    input_type='fconn',
    target_type='float',
    scale=False,
    crop_baseline=False,
    scale_interval=None,
    n_folds=12,
    overwrite=False,
    segment=False,
    test_set=None
)
meta = mneflow.produce_tfrecords(None, **import_opt)
meta.update(model_specs=hyperparameters)
dataset = mneflow.Dataset(meta, train_batch=hyperparameters['batch_size'])

# Build model
if model_name == "m1_depthwise":
    model = WeightedSum3dModel(meta=meta, dataset=dataset, specs_prefix=True)
elif model_name == "m2_sym3d":
    model = SymmetricModel(meta=meta, dataset=dataset, specs_prefix=True)
elif model_name == "m3_conv3d":
    model = Conv3DModel(meta=meta, dataset=dataset, specs_prefix=True)
else:
    raise ValueError(f"Invalid model name: {model_name}. Must be one of: "
                     f"{list(models.keys())}")
if hyperparameters['loss'] == 'MAE':
    model.build(
        learn_rate=hyperparameters['learn_rate'],
        loss='MAE',
        metrics='RootMeanSquaredError',
    )
elif hyperparameters['loss'] == 'MSE':
    model.build(
        learn_rate=hyperparameters['learn_rate'],
        loss='MSE',
        metrics='MAE',
    )
else:
    raise ValueError(f"Invalid loss function: {hyperparameters['loss']}. "
                     f"Must be one of: ['MAE', 'MSE']")

if train:
    model.train(
        n_epochs=500,
        eval_step=hyperparameters['iterations'],
        early_stopping=hyperparameters['early_stop'],
        mode='cv',
        collect_patterns=True
    )
else:
    meta.model_name = "_".join([meta.model_specs['scope'],
                                meta.data['data_id'] + '.h5'])
    model.km.load_weights(os.path.join(meta.model_specs['model_path'],
                                       meta.model_name))
    model.cv_patterns = meta.patterns
    model.meta = meta
    roi_labels = model.get_roi_labels(
        methods=['row', 'col'],
        percentile=80,
        selection_method='activation')
    _ = roi_labels.pop('row')
    _ = roi_labels.pop('col')

if args.ablation:
    for name, label_inds in roi_labels.items():
        meta_abl, model_abl, dataset = model.ablation_analysis(
            name,
            label_inds,
            hyperparameters
        )
        model_abl.build(learn_rate=hyperparameters['learn_rate'],
                        loss='MAE',
                        metrics='RootMeanSquaredError')

        model_abl.train(n_epochs=args['n_train'], eval_step=hyperparameters['iterations'],
                        early_stopping=hyperparameters['early_stop'], mode='cv',
                        collect_patterns=False)

if args.visualize:
    from viz_utils import visualize_source_estimate, numpy_to_stc, set_colormap_alpha
    for name, label_inds in roi_labels.items():
        if name == 'row-col':
            r = model.extract_patterns()['spatial_patterns']['row']
            c = model.extract_patterns()['spatial_patterns']['col']
            pattern = np.concatenate([r, c], axis=-1)
        else:
            pattern = model.extract_patterns()['spatial_patterns'][name]
        avg_pattern = ((pattern - pattern.mean(0)) / pattern.std(0, keepdims=True)).mean(1)
        mask = label_inds
        plot_pattern = np.zeros(avg_pattern.shape)
        plot_pattern[mask] = np.ones(mask.shape)
        cmap = plt.get_cmap('gnuplot')
        cmap = set_colormap_alpha(cmap, alpha=0.6)

        print(':'.join([inp, name, str(len(mask)), str(len(mask) ** 2 / 90 ** 2)]))
        cm = 1 / 2.54
        stc = numpy_to_stc(plot_pattern, subjects_dir=subjects_dir)
        fig = visualize_source_estimate(
            stc,
            '_'.join([fn.figure_dir + inp,
                      name,
                      str(args['percentile']),
                      'run',
                      str(run),
                      args['selection_method'] + ".png"]),
            subjects_dir,
            'fsaverage',
            colorbar_label='Node weight',
            backend='pyvistaqt',
            colormap=cmap,
            # clim={'kind':'percent', 'lims':(50., 75., 90.)},
            title=" Method {} : {} ROIs".format(name, len(label_inds)),
            transparent_overlay=False,
            transparent_background=False,
            title_fontsize=12,
            colorbar_fontsize=8,
            figsize=(10 * cm, 10 * cm),
            alpha=0.8,
        )
        stc.save('_'.join([figure_path + inp,
                           name,
                           str(args['percentile']),
                           'run',
                           str(run),
                           'stc',
                           args['selection_method']]))
