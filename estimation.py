import numpy as np
import json
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
import time
import kornia
import kornia.feature as KF
import kornia.geometry.transform as KGT
from kornia_moons.viz import draw_LAF_matches

def vis_warp_img_tensor(img1, img2, H):
    def tensor_to_plot(tensor):
        tensor = tensor.squeeze()
        temp = np.swapaxes(tensor, 0, 2)
        temp = np.swapaxes(temp, 0, 1)
        return temp
    w, h = img1.size
    warped_img = KGT.warp_perspective(img1, H, (h, w)).squeeze(0)  # C H W
    plt.subplot(3, 1, 1)
    plt.imshow(tensor_to_plot(img1))
    plt.subplot(3, 1, 2)
    plt.imshow(tensor_to_plot(warped_img))
    plt.subplot(3, 1, 3)
    plt.imshow(tensor_to_plot(img2))

    plt.show()
    plt.cla()

def auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs

def convert_coordinates(im_A_coords, im_A_to_im_B, wq, hq, wsup, hsup):
    im_A_coords = (
        np.stack(
            (
                (wq-1) * (im_A_coords[..., 0] + 1) / 2,
                (hq-1) * (im_A_coords[..., 1] + 1) / 2,
            ),
            axis=-1,
        )
    )
    im_A_to_im_B = (
        np.stack(
            (
                (wsup-1) * (im_A_to_im_B[..., 0] + 1) / 2,
                (hsup-1) * (im_A_to_im_B[..., 1] + 1) / 2,
            ),
            axis=-1,
        )
    )
    return im_A_coords, im_A_to_im_B
def demo_estimation(model, img1_path, img2_path, H_s2t_path, if_print=False):
    im_1 = Image.open(img1_path)
    im_2 = Image.open(img2_path)
    if type(H_s2t_path) is str:
        with open(H_s2t_path, 'r') as json_file:
            data = json.load(json_file)
        H_s2t = torch.tensor(data['H']).float()
    else:
        H_s2t = H_s2t_path
    
    w1, h1 = im_1.size
    w2, h2 = im_2.size
    
    start = time.time()
    dense_matches, dense_certainty = model.match(
        im_1, im_2
    )
    good_matches, _ = model.sample(dense_matches, dense_certainty, 5000)
    good_matches = good_matches.cpu().numpy()
    pos_a, pos_b = convert_coordinates(
        good_matches[:, :2], good_matches[:, 2:], w1, h1, w2, h2
    )
    try:
        H_pred, inliers = cv2.findHomography(
            pos_a,
            pos_b,
            method = cv2.RANSAC,
            confidence = 0.99999,
            ransacReprojThreshold = 3
        )
    except:
        H_pred = None

    # vis_warp_img_tensor(im_1, im_2, H_pred)

    if H_pred is None:
        H_pred = np.zeros((3, 3))
        H_pred[2, 2] = 1.0
    runtime = time.time() - start
    corners = np.array(
        [[0, 0, 1], [0, h1 - 1, 1], [w1 - 1, 0, 1], [w1 - 1, h1 - 1, 1]]
    )
    real_warped_corners = np.dot(corners, np.transpose(H_s2t))
    real_warped_corners = (
        real_warped_corners[:, :2] / real_warped_corners[:, 2:]
    )
    warped_corners = np.dot(corners, np.transpose(H_pred))
    warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
    mean_dist = np.mean(
        np.linalg.norm(real_warped_corners - warped_corners, axis=1)
    )
    if mean_dist > 70:
        mean_dist = 70.0
    if if_print:
        print(f'ACE is {mean_dist}.')
        error = kornia.geometry.homography.oneway_transfer_error(torch.from_numpy(pos_a)[None], torch.from_numpy(pos_b)[None], H_s2t[None]).squeeze(0)
        inliers = np.random.permutation(np.arange(0, len(error)))[:50] ##error<14
        plt.clf()
        draw_LAF_matches(
            KF.laf_from_center_scale_ori(
                torch.from_numpy(pos_a[inliers]).view(1, -1, 2),
                torch.ones(pos_a[inliers].shape[0]).view(1, -1, 1, 1),
                torch.ones(pos_a[inliers].shape[0]).view(1, -1, 1),
            ),
            KF.laf_from_center_scale_ori(
                torch.from_numpy(pos_b[inliers]).view(1, -1, 2),
                torch.ones(pos_b[inliers].shape[0]).view(1, -1, 1, 1),
                torch.ones(pos_b[inliers].shape[0]).view(1, -1, 1),
            ),
            torch.arange(pos_a[inliers].shape[0]).view(-1, 1).repeat(1, 2),
            np.array(im_1),
            np.array(im_2),
            error[inliers]<3,
            draw_dict={"inlier_color": (0.2, 1, 0.4), "tentative_color": (1, 0, 0), "feature_color": (0.2, 0.5, 1), "vertical": False},
        )
        plt.axis('off')
        plt.savefig(f'match.png')    
        print('The matching result is saved to match.png.')
    return mean_dist, runtime