import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import os
import torch


def getDST(txt_path):
    dst = []
    with open(txt_path, 'r') as gf:
        for lines in gf.readlines():
            name = lines.replace('\n', '.npy')
            dst.append(name)
    return dst


def cal_affinity(x, axis=1):
    row_max = x.max(axis=axis).reshape(-1, 1)
    norm = x - row_max
    norm_exp = np.exp(norm)
    s = norm_exp / np.sum(norm_exp, axis=axis, keepdims=True)
    return s


def RAPL_HOT(data, lower_index):
    row, col, cha = data.shape[0], data.shape[1], data.shape[2]

    mfc_02 = np.zeros([row, col], dtype=int)
    mfc_04 = np.zeros([row, col], dtype=int)
    mfc_06 = np.zeros([row, col], dtype=int)
    mfc_08 = np.zeros([row, col], dtype=int)
    mfc_10 = np.zeros([row, col], dtype=int)
    mfc_12 = np.zeros([row, col], dtype=int)
    mfc_13 = np.zeros([row, col], dtype=int)

    # HOT test
    Blue = data[:, :, 0]
    Red  = data[:, :, 2]
    HOT = Blue - 0.5 * Red

    # # VBR test
    # VBR = np.min(data[:, :, :3], axis=2) / (np.max(data[:, :, :3], axis=2) + 1e-10)

    # # Red test
    # Red = data[:, :, 2]


    mfc_02[HOT >= 0.02] = 1
    mfc_04[HOT >= 0.04] = 1
    mfc_06[HOT >= 0.06] = 1
    mfc_08[HOT >= 0.08] = 1
    mfc_10[HOT >= 0.10] = 1
    mfc_12[HOT >= 0.12] = 1
    mfc_13[HOT >= 0.13] = 1

    mfc_02_f = mfc_02.flatten()
    mfc_04_f = mfc_04.flatten()
    mfc_06_f = mfc_06.flatten()
    mfc_08_f = mfc_08.flatten()
    mfc_10_f = mfc_10.flatten()
    mfc_12_f = mfc_12.flatten()
    mfc_13_f = mfc_13.flatten()


    c13 = F.cosine_similarity(torch.unsqueeze(torch.tensor(mfc_13_f, dtype=torch.float32), dim=0),
                              torch.unsqueeze(torch.tensor(mfc_13_f, dtype=torch.float32), dim=0))
    c12 = F.cosine_similarity(torch.unsqueeze(torch.tensor(mfc_13_f, dtype=torch.float32), dim=0),
                              torch.unsqueeze(torch.tensor(mfc_12_f, dtype=torch.float32), dim=0))
    c10 = F.cosine_similarity(torch.unsqueeze(torch.tensor(mfc_13_f, dtype=torch.float32), dim=0),
                              torch.unsqueeze(torch.tensor(mfc_10_f, dtype=torch.float32), dim=0))
    c08 = F.cosine_similarity(torch.unsqueeze(torch.tensor(mfc_13_f, dtype=torch.float32), dim=0),
                              torch.unsqueeze(torch.tensor(mfc_08_f, dtype=torch.float32), dim=0))
    c06 = F.cosine_similarity(torch.unsqueeze(torch.tensor(mfc_13_f, dtype=torch.float32), dim=0),
                              torch.unsqueeze(torch.tensor(mfc_06_f, dtype=torch.float32), dim=0))
    c04 = F.cosine_similarity(torch.unsqueeze(torch.tensor(mfc_13_f, dtype=torch.float32), dim=0),
                              torch.unsqueeze(torch.tensor(mfc_04_f, dtype=torch.float32), dim=0))
    c02 = F.cosine_similarity(torch.unsqueeze(torch.tensor(mfc_13_f, dtype=torch.float32), dim=0),
                              torch.unsqueeze(torch.tensor(mfc_02_f, dtype=torch.float32), dim=0))

    D = []
    cosine_sim = np.asarray([c13, c12, c10, c08, c06, c04, c02])
    for i in range(len(cosine_sim)):
        if cosine_sim[i] in D:
            if lower_index > i:
                lower_index = i
            break
        D.append(cosine_sim[i])

    if np.max(D) == 0 or np.min(D) == 1:
        label = mfc_13
    else:
        ann_set = D[:-1]
        MFC = np.asarray([mfc_13, mfc_12, mfc_10, mfc_08, mfc_06, mfc_04, mfc_02])

        s = cal_affinity(np.expand_dims(ann_set, 0))
        if len(ann_set) == 1:
            att = np.expand_dims(np.squeeze(s), 0)
        else:
            att = np.squeeze(s)
        att_pad = np.pad(att, (0, 7 - len(ann_set)), 'constant', constant_values=(0, 0))
        att_extend = np.expand_dims(att_pad, (1, 2))
        label = np.sum(att_extend * MFC, axis=0)

    return label


if __name__ == '__main__':
    tiff_path = './GF1_datasets/JPEGImages_TOA/'

    txt_path = './GF1_datasets/ImageSets/train.txt'
    save_path = './GF1_datasets/pseudoMask/RAPL_HOT/'
    vis_path = './GF1_datasets/pseudoMask/RAPL_HOT_VIS/'
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(vis_path, exist_ok=True)

    for file in getDST(txt_path):
        data = np.load(tiff_path + file)
        pseudo = RAPL_HOT(data)

        np.save(save_path + file, pseudo)

        fig, ax = plt.subplots()
        ax.imshow(pseudo, cmap='gray', aspect='equal')
        plt.axis('off')
        height, width = pseudo.shape
        fig.set_size_inches(width / 100.0, height / 100.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(vis_path + file[:-4] + '.png')
        plt.close()

    print('='*20)
    print('ALL FINISHED！')
    print('=' * 20)

