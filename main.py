from __future__ import print_function

import os
import sys
import types
from timeit import default_timer as timer

import numpy as np
import pandas as pd

### Internal Imports
from datasets.dataset_survival import Generic_WSI_Survival_Dataset, Generic_MIL_Survival_Dataset
from utils.file_utils import save_pkl, load_pkl
from utils.core_utils import train
from utils.utils import get_custom_exp_code

### PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Combinaisons à parcourir --------------------------------------------------
list_encoder = ["openmidnight", "musk", "virchow2", "gpfm", "hibou_l"]
marker_list  = ["BCL2", "BCL6", "CD10", "HE", "MUM1", "MYC"]
label_list = ["PFS", "OS"]

# --- Chemins (ancrés sur l'emplacement du script, peu importe le cwd du job) --
data_root      = os.path.join(BASE_DIR, 'data_224_reborn')  # racine des données, structure {data_root}/{encoder}/{marker}/graphs
label_csv_name = os.path.join(BASE_DIR, 'csv', 'multi_label_patient_id.csv')

results_dir   = os.path.join(BASE_DIR, 'results')   # dossier de sauvegarde des résultats
splits_root   = os.path.join(BASE_DIR, 'splits')     # racine des ./splits/{which_splits}/{marker}
which_splits  = '5foldcv'                            # sous-dossier dans splits_root
global_summary_path = os.path.join(results_dir, 'summary_all.csv')  # agrège summary_latest.csv de tous les runs

# --- Reproductibilité ---------------------------------------------------------
seed = 42

# --- Cross-validation ---------------------------------------------------------
k       = 5    # nombre total de folds
k_start = -1   # fold de départ (-1 = 0)
k_end   = -1   # fold de fin    (-1 = k)

# --- Logs / debug -------------------------------------------------------------
log_data  = True   # activer TensorBoard
overwrite = False  # écraser un run existant
testing   = False  # mode debug (sous-ensemble de données)

# --- Modèle -------------------------------------------------------------------
model_type     = 'patchgcn'  # 'deepset' | 'amil' | 'mifcn' | 'dgc' | 'patchgcn'
mode           = 'graph'     # 'path' | 'cluster' | 'graph'
num_gcn_layers = 4           # nombre de couches GCN (PatchGCN uniquement)
edge_agg       = 'spatial'   # type d'arêtes : 'spatial' | 'latent'
resample       = 0.0         # taux de dropout des patches (0 = désactivé)
drop_out       = True        # dropout interne au modèle (p=0.25)
encoding_size  = 1024        # dimension des embeddings en entrée

# --- Optimiseur ---------------------------------------------------------------
opt        = 'adam'  # 'adam' | 'sgd'
batch_size = 1       # taille de batch (1 recommandé : graphes de tailles variables)
gc         = 32      # gradient accumulation (simule batch_size * gc)
max_epochs = 80      # nombre d'époques d'entraînement
lr         = 1e-4    # taux d'apprentissage

# --- Fonction de perte survie -------------------------------------------------
bag_loss        = 'nll_surv'  # 'ce_surv' | 'nll_surv' | 'cox_surv'
alpha_surv      = 0.0         # poids des patients non censurés dans la loss
bag_weight      = 0.7         # poids de la loss bag-level
label_frac      = 1.0         # fraction des labels utilisés pour l'entraînement
reg             = 1e-5        # L2 weight decay
reg_type        = 'None'      # régularisation L1 : 'None' | 'omic' | 'pathomic'
lambda_reg      = 1e-4        # force de la régularisation L1
weighted_sample = True        # rééquilibrage par classe au sampling
early_stopping  = False       # arrêt anticipé si la loss ne s'améliore plus


ENCODER_CFG = {
    "prism":   dict(in_shape=2560, tiles_subdir="features_virchow",   slide_subdir="slide_features_prism",  slide_csv="prism_encoder.csv"),
    "titan":   dict(in_shape=768,  tiles_subdir="features_conch_v15", slide_subdir="slide_features_titan",  slide_csv="titan_encoder.csv"),
    "feather": dict(in_shape=768,  tiles_subdir="features_conch_v15", slide_subdir="slide_features_feather", slide_csv="feather_encoder.csv"),
    "gpfm": dict(in_shape=1024,  tiles_subdir="features_gpfm", slide_subdir="", slide_csv=""),
    "musk": dict(in_shape=1024,  tiles_subdir="features_musk", slide_subdir="", slide_csv=""),
    "openmidnight": dict(in_shape=1536,  tiles_subdir="features_openmidnight", slide_subdir="", slide_csv=""),
    "hibou_l": dict(in_shape=1024,  tiles_subdir="features_hibou_l", slide_subdir="", slide_csv=""),
    "virchow2": dict(in_shape=2560,  tiles_subdir="features_virchow2", slide_subdir="", slide_csv=""),
}

# =============================================================================
# NE PAS MODIFIER EN DESSOUS
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cleaning_csv(df, marker, element_time): #element_time = 'PFS_time' or 'OS_time'
    df = df[df['stain'] == marker].copy()
    df['case_id'] = df['patient_id']
    mask = df[element_time] > 5.0
    df.loc[mask, "status"] = 0
    df.loc[mask, element_time] = 5.0
    df = df.rename(columns={"status": "censorship", "patient_id": "slide_id"})
    df = df[['case_id', 'slide_id', 'censorship', element_time]]
    return df


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_args(encoder, marker, label):
    args = types.SimpleNamespace(
        data_root_dir=os.path.join(data_root, encoder, marker),
        label_col = label,
        results_dir=results_dir,
        which_splits=which_splits,
        split_dir=marker,
        seed=seed,
        k=k, k_start=k_start, k_end=k_end,
        log_data=log_data, overwrite=overwrite, testing=testing,
        model_type=model_type, mode=mode, num_gcn_layers=num_gcn_layers,
        edge_agg=edge_agg, resample=resample, drop_out=drop_out,
        encoding_size=ENCODER_CFG[encoder]['in_shape'],
        opt=opt, batch_size=batch_size, gc=gc, max_epochs=max_epochs, lr=lr,
        bag_loss=bag_loss, alpha_surv=alpha_surv, bag_weight=bag_weight,
        label_frac=label_frac, reg=reg, reg_type=reg_type, lambda_reg=lambda_reg,
        weighted_sample=weighted_sample, early_stopping=early_stopping,
    )
    args = get_custom_exp_code(args)
    args.task = '%s_survival' % marker
    return args


def load_dataset(args, marker):
    print('\nLoad Dataset')
    args.n_classes = 4
    dataset = Generic_MIL_Survival_Dataset(
        csv_path  = cleaning_csv(pd.read_csv(label_csv_name), marker, args.label_col),
        mode      = args.mode,
        data_dir  = os.path.join(args.data_root_dir, 'graphs'),
        shuffle   = False,
        seed      = args.seed,
        print_info= True,
        patient_strat = False,
        n_bins    = 4,
        label_col = args.label_col,
        ignore    = [],
    )
    if not isinstance(dataset, Generic_MIL_Survival_Dataset):
        raise NotImplementedError
    args.task_type = 'survival'
    return dataset


def main(args, dataset):
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    fold_start = 0        if args.k_start == -1 else args.k_start
    fold_end   = args.k   if args.k_end   == -1 else args.k_end

    latest_val_cindex = []
    folds = np.arange(fold_start, fold_end)

    for i in folds:
        t_start = timer()
        seed_torch(args.seed)
        results_pkl_path = os.path.join(
            args.results_dir, 'split_latest_val_{}_results.pkl'.format(i)
        )
        if os.path.isfile(results_pkl_path):
            print("Skipping Split %d" % i)
            continue

        train_dataset, val_dataset = dataset.return_splits(
            from_id=False,
            csv_path='{}/splits_{}.csv'.format(args.split_dir, i)
        )
        print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
        datasets = (train_dataset, val_dataset)

        if args.task_type == 'survival':
            val_latest, cindex_latest = train(datasets, i, args)
            latest_val_cindex.append(cindex_latest)

        save_pkl(results_pkl_path, val_latest)
        print('Fold %d Time: %f seconds' % (i, timer() - t_start))

    if args.task_type == 'survival':
        results_latest_df = pd.DataFrame({'folds': folds, 'val_cindex': latest_val_cindex})

    results_latest_df.to_csv(os.path.join(args.results_dir, 'summary_latest.csv'))
    return results_latest_df


def append_to_global_summary(results_df, encoder, marker, label, args):
    tagged_df = results_df.copy()
    tagged_df.insert(0, 'encoder', encoder)
    tagged_df.insert(1, 'marker', marker)
    tagged_df.insert(2, 'label', label)
    tagged_df.insert(3, 'exp_code', args.exp_code)

    write_header = not os.path.isfile(global_summary_path)
    tagged_df.to_csv(global_summary_path, mode='a', header=write_header, index=False)


def run_experiment(encoder, marker, label):
    print("\n" + "=" * 80)
    print("Encoder: %s | Marker: %s | Label: %s" % (encoder, marker, label))
    print("=" * 80)

    args = build_args(encoder, marker, label)
    print("Experiment Name:", args.exp_code)
    seed_torch(args.seed)

    settings = {
        'num_splits':    args.k,
        'k_start':       args.k_start,
        'k_end':         args.k_end,
        'task':          args.task,
        'max_epochs':    args.max_epochs,
        'results_dir':   args.results_dir,
        'lr':            args.lr,
        'experiment':    args.exp_code,
        'reg':           args.reg,
        'label_frac':    args.label_frac,
        'bag_loss':      args.bag_loss,
        'bag_weight':    args.bag_weight,
        'seed':          args.seed,
        'model_type':    args.model_type,
        'weighted_sample': args.weighted_sample,
        'gc':            args.gc,
        'opt':           args.opt,
    }

    dataset = load_dataset(args, marker)

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    args.results_dir = os.path.join(
        args.results_dir, encoder, args.which_splits, args.param_code,
        str(args.exp_code) + '_s{}'.format(args.seed)
    )
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
        print("Exp Code <%s> already exists! Skipping." % args.exp_code)
        results_latest_df = pd.read_csv(os.path.join(args.results_dir, 'summary_latest.csv'), index_col=0)
        append_to_global_summary(results_latest_df, encoder, marker, label, args)
        return

    args.split_dir = os.path.join(splits_root, args.which_splits, args.split_dir)
    print("split_dir", args.split_dir)
    assert os.path.isdir(args.split_dir)
    settings.update({'split_dir': args.split_dir})

    with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
        print(settings, file=f)

    print("################# Settings ###################")
    for key, val in settings.items():
        print("{}:  {}".format(key, val))

    results_latest_df = main(args, dataset)
    append_to_global_summary(results_latest_df, encoder, marker, label, args)


for label in label_list:
    for encoder in list_encoder:
        for marker in marker_list:
            run_experiment(encoder, marker, label)

