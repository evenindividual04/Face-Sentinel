"""
Video Face Manipulation Detection Through Ensemble of CNNs

Image and Sound Processing Lab - Politecnico di Milano

Nicolò Bonettini
Edoardo Daniele Cannas
Sara Mandelli
Luca Bondi
Paolo Bestagini
"""
import torch.onnx
if not hasattr(torch.onnx, 'set_training'):
    torch.onnx.set_training = lambda model, mode: model.train(mode)
import argparse
import os
import time
import signal
import shutil
import warnings

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
from torchvision.transforms import ToPILImage, ToTensor
import torch.nn.functional as F
from isplutils import utils, split

torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import ImageChops, Image

from architectures import fornet
from isplutils.data import FrameFaceIterableDataset, load_face
from architectures.EfficientNetB4AttSECBAMFusion import EfficientNetB4AttSECBAMFusion
  


def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, help='Net model class', required=True)
    parser.add_argument('--traindb', type=str, help='Training datasets', nargs='+', choices=split.available_datasets,
                        required=True)
    parser.add_argument('--valdb', type=str, help='Validation datasets', nargs='+', choices=split.available_datasets,
                        required=True)
    parser.add_argument('--celebdf_faces_df_path', type=str, required=True,
                        help='Path to Celeb-DF faces DataFrame')
    parser.add_argument('--celebdf_faces_dir', type=str, required=True,
                        help='Path to Celeb-DF faces directory')
    parser.add_argument('--dfdc_faces_df_path', type=str, action='store',
                        help='Path to the Pandas Dataframe obtained from extract_faces.py on the DFDC dataset. '
                             'Required for training/validating on the DFDC dataset.')
    parser.add_argument('--dfdc_faces_dir', type=str, action='store',
                        help='Path to the directory containing the faces extracted from the DFDC dataset. '
                             'Required for training/validating on the DFDC dataset.')
    parser.add_argument('--ffpp_faces_df_path', type=str, action='store',
                        help='Path to the Pandas Dataframe obtained from extract_faces.py on the FF++ dataset. '
                             'Required for training/validating on the FF++ dataset.')
    parser.add_argument('--ffpp_faces_dir', type=str, action='store',
                        help='Path to the directory containing the faces extracted from the FF++ dataset. '
                             'Required for training/validating on the FF++ dataset.')
    parser.add_argument('--face', type=str, help='Face crop or scale', required=True,
                        choices=['scale', 'tight'])
    parser.add_argument('--size', type=int, help='Train patch size', required=True)

    parser.add_argument('--batch', type=int, help='Batch size to fit in GPU memory', default=32)
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--valint', type=int, help='Validation interval (iterations)', default=500)
    parser.add_argument('--patience', type=int, help='Patience before dropping the LR [validation intervals]',
                        default=10)
    parser.add_argument('--maxiter', type=int, help='Maximum number of iterations', default=20000)
    parser.add_argument('--init', type=str, help='Weight initialization file')
    parser.add_argument('--scratch', action='store_true', help='Train from scratch')

    parser.add_argument('--trainsamples', type=int, help='Limit the number of train samples per epoch', default=-1)
    parser.add_argument('--valsamples', type=int, help='Limit the number of validation samples per epoch',
                        default=6000)

    parser.add_argument('--logint', type=int, help='Training log interval (iterations)', default=100)
    parser.add_argument('--workers', type=int, help='Num workers for data loaders', default=6)
    parser.add_argument('--device', type=str, default='0',help='Device: mps, cpu, or GPU index (int)')
    parser.add_argument('--seed', type=int, help='Random seed', default=0)

    parser.add_argument('--debug', action='store_true', help='Activate debug')
    parser.add_argument('--suffix', type=str, help='Suffix to default tag')

    parser.add_argument('--attention', action='store_true',
                        help='Enable Tensorboard log of attention masks')
    parser.add_argument('--log_dir', type=str, help='Directory for saving the training logs',
                        default='runs/binclass/')
    parser.add_argument('--models_dir', type=str, help='Directory for saving the models weights',
                        default='weights/binclass/')

    # Distillation arguments
    parser.add_argument('--distillation', action='store_true', help='Enable distillation')
    parser.add_argument('--teacher_model', type=str, help='Teacher model class for distillation')
    parser.add_argument('--teacher_weights', type=str, help='Path to teacher model weights')
    parser.add_argument('--alpha', type=float, default=0.5, help='Distillation alpha parameter')
    parser.add_argument('--temperature', type=float, default=2.0, help='Distillation temperature parameter')

    args = parser.parse_args()

    # Parse arguments
    net_class = getattr(fornet, args.net)
    train_datasets = args.traindb
    val_datasets = args.valdb
    dfdc_df_path = args.dfdc_faces_df_path
    ffpp_df_path = args.ffpp_faces_df_path
    dfdc_faces_dir = args.dfdc_faces_dir
    ffpp_faces_dir = args.ffpp_faces_dir
    celebdf_df_path = args.celebdf_faces_df_path
    celebdf_faces_dir = args.celebdf_faces_dir
    face_policy = args.face
    face_size = args.size

    batch_size = args.batch
    initial_lr = args.lr
    validation_interval = args.valint
    patience = args.patience
    max_num_iterations = args.maxiter
    initial_model = args.init
    train_from_scratch = args.scratch

    max_train_samples = args.trainsamples
    max_val_samples = args.valsamples

    log_interval = args.logint
    num_workers = args.workers
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device.isdigit() and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    seed = args.seed

    debug = args.debug
    suffix = args.suffix

    enable_attention = args.attention

    weights_folder = args.models_dir
    logs_folder = args.log_dir

    # Distillation parameters
    distillation = args.distillation
    if distillation:
        teacher_class = getattr(fornet, args.teacher_model)
        teacher_weights = args.teacher_weights
        alpha = args.alpha
        temperature = args.temperature


    # Random initialization
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # Load net
    net: nn.Module = net_class().to(device)

    # Load teacher model if distillation is enabled
    if distillation:
        teacher_net: nn.Module = teacher_class().to(device)
        print(f'Loading teacher model from: {teacher_weights}')
        teacher_state = torch.load(teacher_weights, map_location='cpu')
        teacher_net.load_state_dict(teacher_state['net'])
        teacher_net.eval()

    # Loss and optimizers
    criterion = nn.BCEWithLogitsLoss()

    min_lr = initial_lr * 1e-5
    optimizer = optim.Adam(net.get_trainable_parameters(), lr=initial_lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=0.1,
        patience=patience,
        cooldown=2 * patience,
        min_lr=min_lr,
    )

    tag = utils.make_train_tag(net_class=net_class,
                               traindb=train_datasets,
                               face_policy=face_policy,
                               patch_size=face_size,
                               seed=seed,
                               suffix=suffix,
                               debug=debug,
                               )

    # Model checkpoint paths
    bestval_path = os.path.join(weights_folder, tag, 'bestval.pth')
    last_path = os.path.join(weights_folder, tag, 'last.pth')
    periodic_path = os.path.join(weights_folder, tag, 'it{:06d}.pth')

    os.makedirs(os.path.join(weights_folder, tag), exist_ok=True)

    # Load model
    val_loss = min_val_loss = 10
    epoch = iteration = 0
    patience_counter = 0
    net_state = None
    opt_state = None
    if initial_model is not None:
        # If given load initial model
        print('Loading model form: {}'.format(initial_model))
        state = torch.load(initial_model, map_location='cpu')
        net_state = state['net']
        opt_state = state.get('opt')
        iteration = state.get('iteration', 0)
        epoch = state.get('epoch', 0)
    elif not train_from_scratch and os.path.exists(last_path):
        print('Loading model form: {}'.format(last_path))
        state = torch.load(last_path, map_location='cpu')
        net_state = state['net']
        opt_state = state['opt']
        iteration = state['iteration'] + 1
        epoch = state['epoch']
    if not train_from_scratch and os.path.exists(bestval_path):
        state = torch.load(bestval_path, map_location='cpu')
        min_val_loss = state['val_loss']
    if net_state is not None:
        incomp_keys = net.load_state_dict(net_state, strict=False)
        print(incomp_keys)
    if opt_state is not None:
        for param_group in opt_state['param_groups']:
            param_group['lr'] = initial_lr
        optimizer.load_state_dict(opt_state)

    # Initialize Tensorboard
    logdir = os.path.join(logs_folder, tag)
    if iteration == 0:
        # If training from scratch or initialization remove history if exists
        shutil.rmtree(logdir, ignore_errors=True)

    # TensorboardX instance
    tb = SummaryWriter(logdir=logdir)
    if iteration == 0:
        dummy = torch.randn((1, 3, face_size, face_size), device=device)
        dummy = dummy.to(device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # tb.add_graph(net, [dummy, ], verbose=False)

    transformer = utils.get_transformer(face_policy=face_policy, patch_size=face_size,
                                        net_normalizer=net.get_normalizer(), train=True)

    # Datasets and data loaders
    print('Loading data')
    # Check if paths for DFDC and FF++ extracted faces and DataFrames are provided
    for dataset in train_datasets:
        if dataset.split('-')[0] == 'dfdc' and (dfdc_df_path is None or dfdc_faces_dir is None):
            raise RuntimeError('Specify DataFrame and directory for DFDC faces for training!')
        elif dataset.split('-')[0] == 'ff' and (ffpp_df_path is None or ffpp_faces_dir is None):
            raise RuntimeError('Specify DataFrame and directory for FF++ faces for training!')
        elif dataset.split('-')[0] == 'celebdf' and (celebdf_df_path is None or celebdf_faces_dir is None):
            raise RuntimeError('Specify DataFrame and directory for Celeb-DF faces for training!')
    for dataset in val_datasets:
        if dataset.split('-')[0] == 'dfdc' and (dfdc_df_path is None or dfdc_faces_dir is None):
            raise RuntimeError('Specify DataFrame and directory for DFDC faces for validation!')
        elif dataset.split('-')[0] == 'ff' and (ffpp_df_path is None or ffpp_faces_dir is None):
            raise RuntimeError('Specify DataFrame and directory for FF++ faces for validation!')
        elif dataset.split('-')[0] == 'celebdf' and (celebdf_df_path is None or celebdf_faces_dir is None):
            raise RuntimeError('Specify DataFrame and directory for Celeb-DF faces for validation!')
    # Load splits with the make_splits function
    splits = split.make_splits(
    dfdc_df=dfdc_df_path,
    ffpp_df=ffpp_df_path,
    dfdc_dir=dfdc_faces_dir,
    ffpp_dir=ffpp_faces_dir,
    celebdf_df=args.celebdf_faces_df_path,
    celebdf_dir=args.celebdf_faces_dir,
    dbs={'train': train_datasets, 'val': val_datasets}
)
    train_dfs = [splits['train'][db][0] for db in splits['train']]
    train_roots = [splits['train'][db][1] for db in splits['train']]
    val_roots = [splits['val'][db][1] for db in splits['val']]
    val_dfs = [splits['val'][db][0] for db in splits['val']]
    
    print('Train DataFrame lengths:', [len(df) for df in train_dfs])
    print('Validation DataFrame lengths:', [len(df) for df in val_dfs])


    train_dataset = FrameFaceIterableDataset(roots=train_roots,
                                             dfs=train_dfs,
                                             scale=face_policy,
                                             num_samples=max_train_samples,
                                             transformer=transformer,
                                             size=face_size,
                                             )

    val_dataset = FrameFaceIterableDataset(roots=val_roots,
                                           dfs=val_dfs,
                                           scale=face_policy,
                                           num_samples=max_val_samples,
                                           transformer=transformer,
                                           size=face_size,
                                           )
    
    print('Training dataset length:', len(train_dataset))
    print('Validation dataset length:', len(val_dataset))


    train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, )

    val_loader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, )
    
    print('Batch size:', batch_size)
    print('Batches per epoch (train):', len(train_loader))
    print('Batches per epoch (val):', len(val_loader))


    print('Training samples: {}'.format(len(train_dataset)))
    print('Validation samples: {}'.format(len(val_dataset)))

    if len(train_dataset) == 0:
        print('No training samples. Halt.')
        return

    if len(val_dataset) == 0:
        print('No validation samples. Halt.')
        return
    
    """LOGGING
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup logging and checkpoint paths
    log_file = os.path.join(args.log_dir, 'training_log.txt')
    checkpoint_path = os.path.join(args.models_dir, 'checkpoint.pth')
    best_model_path = os.path.join(args.models_dir, 'bestval.pth')
    
    # Initialize logging
    with open(log_file, 'w') as f:
        f.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Arguments: {vars(args)}\n")
        """

    stop = False

    while not stop:


        # Training
        optimizer.zero_grad()

        train_loss = train_num = 0
        train_pred_list = []
        train_labels_list = []
        for train_batch in tqdm(train_loader, desc='Epoch {:03d}'.format(epoch), leave=False,
                                total=len(train_loader) // train_loader.batch_size):
            net.train()
            batch_data, batch_labels = train_batch

            train_batch_num = len(batch_labels)
            train_num += train_batch_num
            train_labels_list.append(batch_labels.numpy().flatten())

            if distillation:
                train_batch_loss, train_batch_pred = batch_forward_distillation(net, teacher_net, device, criterion, batch_data, batch_labels, alpha, temperature)
            else:
                train_batch_loss, train_batch_pred = batch_forward(net, device, criterion, batch_data, batch_labels)
            train_pred_list.append(train_batch_pred.flatten())

            if torch.isnan(train_batch_loss):
                raise ValueError('NaN loss')

            train_loss += train_batch_loss.item() * train_batch_num

            # Optimization
            train_batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Logging
            if iteration > 0 and (iteration % log_interval == 0):
                train_loss /= train_num
                tb.add_scalar('train/loss', train_loss, iteration)
                tb.add_scalar('lr', optimizer.param_groups[0]['lr'], iteration)
                tb.add_scalar('epoch', epoch, iteration)

                # Checkpoint
                save_model(net, optimizer, train_loss, val_loss, iteration, batch_size, epoch, last_path)
                train_loss = train_num = 0

            # Validation
            if iteration > 0 and (iteration % validation_interval == 0):

                # Model checkpoint
                save_model(net, optimizer, train_loss, val_loss, iteration, batch_size, epoch,
                           periodic_path.format(iteration))

                # Train cumulative stats
                train_labels = np.concatenate(train_labels_list)
                train_pred = np.concatenate(train_pred_list)
                train_labels_list = []
                train_pred_list = []

                train_roc_auc = roc_auc_score(train_labels, train_pred)
                tb.add_scalar('train/roc_auc', train_roc_auc, iteration)
                tb.add_pr_curve('train/pr', train_labels, train_pred, iteration)

                # Validation
                val_loss = validation_routine(net, device, val_loader, criterion, tb, iteration, 'val')
                tb.flush()

                # LR Scheduler
                lr_scheduler.step(val_loss)

                # Model checkpoint
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    save_model(net, optimizer, train_loss, val_loss, iteration, batch_size, epoch, bestval_path)
                    patience_counter = 0  # Reset patience counter on improvement
                else:
                    patience_counter += 1  # Increment patience counter on no improvement

                if patience_counter >= patience:  # Check if patience limit is reached
                    print(f'Early stopping triggered: Validation loss did not improve for {patience} intervals.')
                    stop = True
                    break

                # Attention
                if enable_attention and hasattr(net, 'get_attention'):
                    net.eval()
                    # For each dataframe show the attention for a real,fake couple of frames
                    for df, root, sample_idx, tag in [
                        (train_dfs[0], train_roots[0], train_dfs[0][train_dfs[0]['label'] == False].index[0],
                         'train/att/real'),
                        (train_dfs[0], train_roots[0], train_dfs[0][train_dfs[0]['label'] == True].index[0],
                         'train/att/fake'),
                    ]:
                        record = df.loc[sample_idx]
                        tb_attention(tb, tag, iteration, net, device, face_size, face_policy,
                                     transformer, root, record)

                if optimizer.param_groups[0]['lr'] == min_lr:
                    print('Reached minimum learning rate. Stopping.')
                    stop = True
                    break

            iteration += 1

            if iteration > max_num_iterations:
                print('Maximum number of iterations reached')
                stop = True
                break

            # End of iteration

        epoch += 1

    # Needed to flush out last events
    tb.close()

    print('Completed')


def tb_attention(tb: SummaryWriter,
                 tag: str,
                 iteration: int,
                 net: nn.Module,
                 device: torch.device,
                 patch_size_load: int,
                 face_crop_scale: str,
                 val_transformer: A.BasicTransform,
                 root: str,
                 record: pd.Series,
                 ):
    # Crop face
    sample_t = load_face(record=record, root=root, size=patch_size_load, scale=face_crop_scale,
                         transformer=val_transformer)
    sample_t_clean = load_face(record=record, root=root, size=patch_size_load, scale=face_crop_scale,
                               transformer=ToTensorV2())
    if torch.cuda.is_available():
        sample_t = sample_t.cuda(device)
    # Transform
    # Feed to net
    with torch.no_grad():
        att: torch.Tensor = net.get_attention(sample_t.unsqueeze(0))[0].cpu()
    att_img: Image.Image = ToPILImage()(att)
    sample_img = ToPILImage()(sample_t_clean)
    att_img = att_img.resize(sample_img.size, resample=Image.NEAREST).convert('RGB')
    sample_att_img = ImageChops.multiply(sample_img, att_img)
    sample_att = ToTensor()(sample_att_img)
    tb.add_image(tag=tag, img_tensor=sample_att, global_step=iteration)


def batch_forward(net: nn.Module, device: torch.device, criterion, data: torch.Tensor, labels: torch.Tensor) -> (
        torch.Tensor, float, int):
    data = data.to(device)
    labels = labels.float().to(device)
    out = net(data)
    pred = torch.sigmoid(out).detach().cpu().numpy()
    loss = criterion(out, labels)
    return loss, pred


def batch_forward_distillation(student_net: nn.Module, teacher_net: nn.Module, device: torch.device, criterion, data: torch.Tensor, labels: torch.Tensor, alpha: float, temperature: float) -> (
        torch.Tensor, float, int):
    data = data.to(device)
    labels = labels.float().to(device)

    # Teacher forward
    with torch.no_grad():
        teacher_out = teacher_net(data)

    # Student forward
    student_out = student_net(data)

    # Standard loss
    loss = criterion(student_out, labels)

    # Distillation loss
    distillation_loss = nn.KLDivLoss()(F.log_softmax(student_out / temperature, dim=1),
                                    F.softmax(teacher_out / temperature, dim=1)) * (temperature * temperature)

    # Total loss
    total_loss = alpha * loss + (1 - alpha) * distillation_loss

    pred = torch.sigmoid(student_out).detach().cpu().numpy()
    return total_loss, pred


def validation_routine(net, device, val_loader, criterion, tb, iteration, tag: str, loader_len_norm: int = None):
    net.eval()
    loader_len_norm = loader_len_norm if loader_len_norm is not None else val_loader.batch_size
    val_num = 0
    val_loss = 0.
    pred_list = list()
    labels_list = list()
    for val_data in tqdm(val_loader, desc='Validation', leave=False, total=len(val_loader) // loader_len_norm):
        batch_data, batch_labels = val_data

        val_batch_num = len(batch_labels)
        labels_list.append(batch_labels.flatten())
        with torch.no_grad():
            val_batch_loss, val_batch_pred = batch_forward(net, device, criterion, batch_data,
                                                           batch_labels)
        pred_list.append(val_batch_pred.flatten())
        val_num += val_batch_num
        val_loss += val_batch_loss.item() * val_batch_num

    # Logging
    val_loss /= val_num
    tb.add_scalar('{}/loss'.format(tag), val_loss, iteration)

    if isinstance(criterion, nn.BCEWithLogitsLoss):
        val_labels = np.concatenate(labels_list)
        val_pred = np.concatenate(pred_list)
        val_roc_auc = roc_auc_score(val_labels, val_pred)
        tb.add_scalar('{}/roc_auc'.format(tag), val_roc_auc, iteration)
        tb.add_pr_curve('{}/pr'.format(tag), val_labels, val_pred, iteration)

    return val_loss


def save_model(net: nn.Module, optimizer: optim.Optimizer,
               train_loss: float, val_loss: float,
               iteration: int, batch_size: int, epoch: int,
               path: str):
    path = str(path)
    state = dict(net=net.state_dict(),
                 opt=optimizer.state_dict(),
                 train_loss=train_loss,
                 val_loss=val_loss,
                 iteration=iteration,
                 batch_size=batch_size,
                 epoch=epoch)
    torch.save(state, path)


if __name__ == '__main__':
    main()
