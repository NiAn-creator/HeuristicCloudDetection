from tqdm import tqdm
import network
import os
import random
import argparse
import numpy as np
from torch.utils import data
from datasets import wscd_test_gf1,wscd_test_wdcd,wscd_trainval_gf1
from metrics import StreamSegMetrics
import torch
import torch.nn as nn
import time

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--dataset", type=str, default='gf1', help='Name of dataset')
    parser.add_argument("--data_root", type=str, default='/home/FAKEDATA/GF1_datasets/datasets_321/data/',
                        help="path to Dataset")

    parser.add_argument("--gpu_id", type=str, default='4',help="GPU ID")
    parser.add_argument("--batch_size", type=int, default=8,help='batch size (default: 4)')
    parser.add_argument("--num_classes", type=int, default=2,help="num classes (default: None)")

    # Model Options
    parser.add_argument("--model", type=str, default='mResNet34_PHA_DBRM_GF1', help='model name')

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=True)
    parser.add_argument("--ckpt",
                        default='The path of your best checkpoint file',
                        help="restore from checkpoint")

    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"predict_path\"")
    parser.add_argument("--predict_path", default='./output/',
                        help="save prediction results")


    parser.add_argument("--random_seed", type=int, default=1,help="random seed (default: 1)")
    return parser

def get_dataset(opts):
    if opts.dataset == 'gf1':
        val_dst = wscd_trainval(root=opts.data_root, image_set='test')

    elif opts.dataset == 'landsat':
        val_dst = wscd_test_landsat(root=opts.data_root_test, image_set='test')

    elif opts.dataset == 'WDCD':
        val_dst = wscd_test_wdcd(root=opts.data_root_test, image_set='test')

    return  val_dst


def validate(opts, model, loader, metrics, device, threshold):
    metrics.reset()

    index = 0
    with torch.no_grad():
        for i,(images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs, boundary= model(images)

            outputs = torch.squeeze(outputs).cpu().numpy()
            targets = labels.cpu().numpy()
            b, h, w = targets.shape[0], targets.shape[1], targets.shape[2]
            preds = np.zeros((b, h, w), dtype=int)
            preds[outputs >= thres] = 1

            metrics.update(targets, preds)

            if opts.save_val_results:
                sample_fname = loader.sampler.data_source.masks[:]
                os.makedirs(opts.predict_path, exist_ok=True)

                for batch in range(images.shape[0]):
                    content = sample_fname[index].replace('.npy', "").split("/")
                    np.save(opts.predict_path + content[-1] + '.npy', preds[batch, :, :])
                    index = index + 1

        score = metrics.get_results()
    return score,threshold


if __name__ == '__main__':
    opts = get_argparser().parse_args()

    # select the GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s,  CUDA_VISIBLE_DEVICES: %s\n" % (device, opts.gpu_id))

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    val_dst = get_dataset(opts)
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size, shuffle=False, num_workers=8, drop_last=False,
                                 pin_memory=False)

    print("Dataset: %s, val set: %d" % (opts.dataset, len(val_dst)))

    # Set up model
    model_map = {
        'mResNet34_PHA_DBRM_GF1': network.mResNet34_PHA_DBRM_GF1,
    }

    print('Model = %s, num_classes=%d' % (opts.model, opts.num_classes))
    model = model_map[opts.model](num_classes=opts.num_classes)

    # Restore
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Model restored from %s" % opts.ckpt)
    else:
        print("Error: Can not load best checkpoints_v1.")

    if opts.test_only:
        model.eval()

        for thres in [0.5]:
            print('************************************************************************')
            print(thres)
            time_before_val = time.time()

            val_score,threshold = validate(opts=opts, model=model, loader=val_loader,
                                           metrics=metrics, device=device,threshold=thres)

            time_after_val = time.time()
            print('Time_val = %f' % (time_after_val - time_before_val))
            print('Threshold = %f' % (threshold))
            print(metrics.to_str(val_score))

