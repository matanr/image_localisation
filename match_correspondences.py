import argparse
import glob
import sys

import torch
from pathlib import Path
from extractor import ViTExtractor
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import List, Tuple
import cv2 as cv
import os
import shutil
import random

def find_new_out_dir(out_dir, drone_img_path, drone_scale):
    pair_nums = [int(str(x).split("_")[-1]) for x in root_dir.iterdir() if x.is_dir()]
    if len(pair_nums) == 0:
        pair_nums = [0]
    pair_nums.sort()
    new_out_dir = os.path.join(out_dir, f"pairs_{pair_nums[-1]+1}")
    os.makedirs(new_out_dir, exist_ok=True)
    if drone_img_path:
        drone_image = cv.imread(drone_img_path)
        scale_fac = drone_scale  # percent of original size
        width = int(drone_image.shape[1] * scale_fac)
        height = int(drone_image.shape[0] * scale_fac)
        dim = (width, height)
        drone_image = cv.resize(drone_image, dim)
        cv.imwrite(os.path.join(new_out_dir, os.path.basename(drone_img_path)), drone_image)
    return new_out_dir

def divide_to_squares(in_dir, in_img, out_dim, stride, out_dir, angle, drone_img_path, drone_scale):
    img_path = os.path.join(in_dir, in_img)
    img = cv.imread(img_path)
    original_img_shape = img.shape

    # scale_fac_str = str(scale_fac).replace(".", "_")
    angle_str = str(angle)

    # if scale_fac != 1:
    #     width = int(img.shape[1] * scale_fac)
    #     height = int(img.shape[0] * scale_fac)
    #     dim = (width, height)
    #     img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    if angle == 90:
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        img = cv.rotate(img, cv.ROTATE_180)
    elif angle == 270:
        img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)

    shape = img.shape

    imgheight, imgwidth  = shape[0], shape[1]

    for i in range(0, imgheight, stride):
        for j in range(0, imgwidth, stride):
            if i+out_dim > imgheight or j+out_dim > imgwidth:
                continue
            square = img[i:i+out_dim, j:j+out_dim]
            # # delete this 2 lines and uncomment all lines below
            # img_wo_ext = os.path.splitext(os.path.basename(in_img))[0]
            # cv.imwrite(os.path.join(out_dir, "{}-{}-{}.png".format(img_wo_ext, i, j)), square)
            new_out_dir = find_new_out_dir(out_dir, drone_img_path, drone_scale)
            cv.imwrite(os.path.join(new_out_dir, "{}-{}-{}.png".format(angle_str, i, j)), square)
    for i in range(0, imgheight, stride):
        if i+out_dim > imgheight:
                continue
        square = img[i:i+out_dim, imgwidth-out_dim:imgwidth]
        new_out_dir = find_new_out_dir(out_dir, drone_img_path, drone_scale)
        cv.imwrite(os.path.join(new_out_dir, "{}-{}-{}.png".format(angle_str, i, j)), square)
    for j in range(0, imgwidth, stride):
        if j+out_dim > imgwidth:
                continue
        square = img[imgheight-out_dim:imgheight, j:j+out_dim]
        new_out_dir = find_new_out_dir(out_dir, drone_img_path, drone_scale)
        cv.imwrite(os.path.join(new_out_dir, "{}-{}-{}.png".format(angle_str, i, j)), square)

    square = img[imgheight-out_dim:imgheight, imgwidth-out_dim:imgwidth]
    new_out_dir = find_new_out_dir(out_dir, drone_img_path, drone_scale)
    cv.imwrite(os.path.join(new_out_dir, "{}-{}-{}.png".format(angle_str, i, j)), square)

    return shape, original_img_shape


def divide_to_hierarchy_of_squares(in_dir, in_img, out_dim, stride, out_dir, angles, drone_img_path=None, drone_scale=0.125):
    shapes = []
    img_path = os.path.join(in_dir, in_img)
    img = cv.imread(img_path)
    shape = img.shape
    shapes.append(shape)

    for angle in angles:
        shape,_ = divide_to_squares(in_dir, in_img, out_dim, stride, out_dir, angle, drone_img_path, drone_scale)
        shapes.append(shape)
    return shapes


def union_from_squares(in_dir, original_img_shape, img_shape, square_dim, stride, out_dir, img_name, scale_fac, img_format="jpg", write_img=True):
    scale_fac_str = str(scale_fac).replace(".", "_")
    imgheight, imgwidth  = img_shape[0], img_shape[1]
    out_img = np.zeros(shape=img_shape, dtype="float32")
    squares_number_per_pixel = np.zeros(shape=img_shape, dtype="float32")
    for i in range(0, imgheight, stride):
        for j in range(0, imgwidth, stride):
            if i+square_dim > imgheight or j+square_dim > imgwidth:
                continue
            # # # delete these 9 lines and uncomment all lines below
            # img_wo_ext = os.path.splitext(os.path.basename(img_name))[0]
            # square_name = os.path.join(in_dir, "{}-{}-{}.{}".format(img_wo_ext, i, j, img_format))
            # if os.path.exists(square_name):
            #     print ("FOUND {}".format(square_name))
            #     img = cv.imread(square_name)
            #     out_img[i:i + square_dim, j:j + square_dim] = img[:, :]
            # else:
            #     print ("NOT FOUND {}".format(square_name))
            #     out_img[i:i + square_dim, j:j + square_dim] = 255

            img = cv.imread(os.path.join(in_dir, "{}-{}-{}.{}".format(scale_fac_str, i, j, img_format)))
            cur_sum = out_img[i:i+square_dim, j:j+square_dim] * squares_number_per_pixel[i:i+square_dim, j:j+square_dim]
            cur_sum[:, :] = cur_sum[:, :] + img[:, :]
            squares_number_per_pixel[i:i + square_dim, j:j + square_dim] += 1
            out_img[i:i + square_dim, j:j + square_dim] = cur_sum / squares_number_per_pixel[i:i + square_dim, j:j + square_dim]
    for i in range(0, imgheight, stride):
        if i+square_dim > imgheight:
                continue
        img = cv.imread(os.path.join(in_dir, "{}-{}-{}.{}".format(scale_fac_str, i, j, img_format)))
        cur_sum = out_img[i:i+square_dim, imgwidth-square_dim:imgwidth] * squares_number_per_pixel[i:i+square_dim, imgwidth-square_dim:imgwidth]
        cur_sum[:, :] = cur_sum[:, :] + img[:, :]
        squares_number_per_pixel[i:i+square_dim, imgwidth-square_dim:imgwidth] += 1
        out_img[i:i+square_dim, imgwidth-square_dim:imgwidth] = cur_sum / squares_number_per_pixel[i:i+square_dim, imgwidth-square_dim:imgwidth]
    for j in range(0, imgwidth, stride):
        if j+square_dim > imgwidth:
                continue
        img = cv.imread(os.path.join(in_dir, "{}-{}-{}.{}".format(scale_fac_str, i, j, img_format)))
        cur_sum = out_img[imgheight-square_dim:imgheight, j:j+square_dim] * squares_number_per_pixel[imgheight-square_dim:imgheight, j:j+square_dim]
        cur_sum[:, :] = cur_sum[:, :] + img[:, :]
        squares_number_per_pixel[imgheight-square_dim:imgheight, j:j+square_dim] += 1
        out_img[imgheight-square_dim:imgheight, j:j+square_dim] = cur_sum / squares_number_per_pixel[imgheight-square_dim:imgheight, j:j+square_dim]

    img = cv.imread(os.path.join(in_dir, "{}-{}-{}.{}".format(scale_fac_str, i, j, img_format)))
    cur_sum = out_img[imgheight-square_dim:imgheight, imgwidth-square_dim:imgwidth] * squares_number_per_pixel[imgheight-square_dim:imgheight, imgwidth-square_dim:imgwidth]
    cur_sum[:, :] = cur_sum[:, :] + img[:, :]
    squares_number_per_pixel[imgheight-square_dim:imgheight, imgwidth-square_dim:imgwidth] += 1
    out_img[imgheight-square_dim:imgheight, imgwidth-square_dim:imgwidth] = cur_sum / squares_number_per_pixel[imgheight-square_dim:imgheight, imgwidth-square_dim:imgwidth]

    if scale_fac != 1:
        # width = int(out_img.shape[1] / scale_fac)
        # height = int(out_img.shape[0] / scale_fac)
        width = original_img_shape[1]
        height = original_img_shape[0]
        dim = (width, height)
        out_img = cv.resize(out_img, dim, interpolation=cv.INTER_AREA)
    if write_img:
        cv.imwrite(os.path.join(out_dir, img_name), out_img)
    return out_img


def union_from_hierarchy_of_squares(in_dir, img_shapes, square_dim, stride, out_dir, img_name, scales, img_format="jpg", gray_scale=False):

    final_img = np.zeros(shape=img_shapes[0], dtype="float32")

    weights_sum = 0
    for img_shape, scale_fac in zip(img_shapes[1:], scales):
        out_img = union_from_squares(in_dir, img_shapes[0], img_shape, square_dim, stride, out_dir, img_name, scale_fac, img_format, write_img=False)
        cv.imwrite(os.path.join(out_dir, "{}_{}".format(scale_fac,img_name)), out_img)
        weight = 1/scale_fac
        # print("w: {}".format(weight))
        final_img += (out_img * weight)
        weights_sum += weight

    final_img /= weights_sum
    # print("ws: {}".format(weights_sum))
    cv.imwrite(os.path.join(out_dir, img_name), final_img)


def compute_homography_score(image1_pil, image2_pil, points1, points2):
    # pil_image1 = image1_pil.convert('RGB')
    # open_cv_image1 = np.array(pil_image1)
    # # Convert RGB to BGR
    # image1 = open_cv_image1[:, :, ::-1].copy()
    #
    # pil_image2 = image2_pil.convert('RGB')
    # open_cv_image2 = np.array(pil_image2)
    # # Convert RGB to BGR
    # image2 = open_cv_image2[:, :, ::-1].copy()
    if len(points1) < 4 or len(points2) < 4:
        return 0


    src_pts = np.float32(points1).reshape(-1, 1, 2)
    dst_pts = np.float32(points2).reshape(-1, 1, 2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 10.0)
    matches_mask = mask.ravel().tolist()
    return np.sum(matches_mask) / 10


def find_correspondences(image_path1: str, image_path2: str, num_pairs: int = 10, load_size: int = 224, layer: int = 9,
                         facet: str = 'key', bin: bool = True, thresh: float = 0.05, model_type: str = 'dino_vits8',
                         stride: int = 4) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]],
                                                                              Image.Image, Image.Image]:
    """
    finding point correspondences between two images.
    :param image_path1: path to the first image.
    :param image_path2: path to the second image.
    :param num_pairs: number of outputted corresponding pairs.
    :param load_size: size of the smaller edge of loaded images. If None, does not resize.
    :param layer: layer to extract descriptors from.
    :param facet: facet to extract descriptors from.
    :param bin: if True use a log-binning descriptor.
    :param thresh: threshold of saliency maps to distinguish fg and bg.
    :param model_type: type of model to extract descriptors from.
    :param stride: stride of the model.
    :return: list of points from image_path1, list of corresponding points from image_path2, the processed pil image of
    image_path1, and the processed pil image of image_path2.
    """
    # extracting descriptors for each image
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(model_type, stride, device=device)
    image1_batch, image1_pil = extractor.preprocess(image_path1, load_size)
    descriptors1 = extractor.extract_descriptors(image1_batch.to(device), layer, facet, bin)
    num_patches1, load_size1 = extractor.num_patches, extractor.load_size
    image2_batch, image2_pil = extractor.preprocess(image_path2, load_size)
    descriptors2 = extractor.extract_descriptors(image2_batch.to(device), layer, facet, bin)
    num_patches2, load_size2 = extractor.num_patches, extractor.load_size

    # extracting saliency maps for each image
    saliency_map1 = extractor.extract_saliency_maps(image1_batch.to(device))[0]
    saliency_map2 = extractor.extract_saliency_maps(image2_batch.to(device))[0]
    # threshold saliency maps to get fg / bg masks
    fg_mask1 = saliency_map1 > thresh
    fg_mask2 = saliency_map2 > thresh

    # calculate similarity between image1 and image2 descriptors
    similarities = chunk_cosine_sim(descriptors1, descriptors2)

    # calculate best buddies
    image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=device)
    sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
    sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
    sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
    sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]
    bbs_mask = nn_2[nn_1] == image_idxs

    # remove best buddies where at least one descriptor is marked bg by saliency mask.
    fg_mask2_new_coors = nn_2[fg_mask2]
    fg_mask2_mask_new_coors = torch.zeros(num_patches1[0] * num_patches1[1], dtype=torch.bool, device=device)
    fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
    bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
    bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)

    # applying k-means to extract k high quality well distributed correspondence pairs
    bb_descs1 = descriptors1[0, 0, bbs_mask, :].cpu().numpy()
    bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :].cpu().numpy()
    # apply k-means on a concatenation of a pairs descriptors.
    all_keys_together = np.concatenate((bb_descs1, bb_descs2), axis=1)
    n_clusters = min(num_pairs, len(all_keys_together))  # if not enough pairs, show all found pairs.
    length = np.sqrt((all_keys_together ** 2).sum(axis=1))[:, None]
    normalized = all_keys_together / length
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized)
    bb_topk_sims = np.full((n_clusters), -np.inf)
    bb_indices_to_show = np.full((n_clusters), -np.inf)

    # rank pairs by their mean saliency value
    bb_cls_attn1 = saliency_map1[bbs_mask]
    bb_cls_attn2 = saliency_map2[nn_1[bbs_mask]]
    bb_cls_attn = (bb_cls_attn1 + bb_cls_attn2) / 2
    ranks = bb_cls_attn



    for k in range(n_clusters):
        for i, (label, rank) in enumerate(zip(kmeans.labels_, ranks)):
            if rank > bb_topk_sims[label]:
                bb_topk_sims[label] = rank
                bb_indices_to_show[label] = i

    # get coordinates to show
    indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[
        bb_indices_to_show]  # close bbs
    img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device=device)[indices_to_show]
    img2_indices_to_show = nn_1[indices_to_show]
    # coordinates in descriptor map's dimensions
    img1_y_to_show = (img1_indices_to_show / num_patches1[1]).cpu().numpy()
    img1_x_to_show = (img1_indices_to_show % num_patches1[1]).cpu().numpy()
    img2_y_to_show = (img2_indices_to_show / num_patches2[1]).cpu().numpy()
    img2_x_to_show = (img2_indices_to_show % num_patches2[1]).cpu().numpy()
    points1, points2 = [], []
    for y1, x1, y2, x2 in zip(img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show):
        x1_show = (int(x1) - 1) * extractor.stride[1] + extractor.stride[1] + extractor.p // 2
        y1_show = (int(y1) - 1) * extractor.stride[0] + extractor.stride[0] + extractor.p // 2
        x2_show = (int(x2) - 1) * extractor.stride[1] + extractor.stride[1] + extractor.p // 2
        y2_show = (int(y2) - 1) * extractor.stride[0] + extractor.stride[0] + extractor.p // 2
        points1.append((y1_show, x1_show))
        points2.append((y2_show, x2_show))

    img1_matched_patches_descriptors = descriptors1[0, 0, img1_indices_to_show, :]
    img2_matched_patches_descriptors = descriptors2[0, 0, img2_indices_to_show, :]
    cos = torch.nn.CosineSimilarity()
    sim = cos(img1_matched_patches_descriptors, img2_matched_patches_descriptors)
    similarity_score = sim.mean().cpu().numpy()

    homography_score = compute_homography_score(image1_pil, image2_pil, points1, points2)
    return points1, points2, image1_pil, image2_pil, similarity_score, homography_score


def draw_correspondences(points1: List[Tuple[float, float]], points2: List[Tuple[float, float]],
                         image1: Image.Image, image2: Image.Image) -> Tuple[plt.Figure, plt.Figure]:
    """
    draw point correspondences on images.
    :param points1: a list of (y, x) coordinates of image1, corresponding to points2.
    :param points2: a list of (y, x) coordinates of image2, corresponding to points1.
    :param image1: a PIL image.
    :param image2: a PIL image.
    :return: two figures of images with marked points.
    """
    assert len(points1) == len(points2), f"points lengths are incompatible: {len(points1)} != {len(points2)}."
    num_points = len(points1)
    fig1, ax1 = plt.subplots()
    ax1.axis('off')
    fig2, ax2 = plt.subplots()
    ax2.axis('off')
    ax1.imshow(image1)
    ax2.imshow(image2)
    if num_points > 15:
        cmap = plt.get_cmap('tab10')
    else:
        cmap = ListedColormap(["red", "yellow", "blue", "lime", "magenta", "indigo", "orange", "cyan", "darkgreen",
                               "maroon", "black", "white", "chocolate", "gray", "blueviolet"])
    colors = np.array([cmap(x) for x in range(num_points)])
    radius1, radius2 = 8, 1
    for point1, point2, color in zip(points1, points2, colors):
        y1, x1 = point1
        circ1_1 = plt.Circle((x1, y1), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ1_2 = plt.Circle((x1, y1), radius2, facecolor=color, edgecolor='white')
        ax1.add_patch(circ1_1)
        ax1.add_patch(circ1_2)
        y2, x2 = point2
        circ2_1 = plt.Circle((x2, y2), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ2_2 = plt.Circle((x2, y2), radius2, facecolor=color, edgecolor='white')
        ax2.add_patch(circ2_1)
        ax2.add_patch(circ2_2)
    return fig1, fig2


def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)


""" taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def move_random_pairs(src_pairs_dir, dst_pairs_dir, amount_to_move, best_fit_patch_name=None):
    all_pairs = os.listdir(src_pairs_dir)
    if best_fit_patch_name:
        print(f"Moving {amount_to_move+1} patches for testing.")
        source = os.path.join(src_pairs_dir, best_fit_patch_name)
        shutil.move(source, dst_pairs_dir, copy_function=shutil.copytree)
        all_pairs.remove(best_fit_patch_name)
    else:
        print(f"Moving {amount_to_move} patches for testing.")
    pairs_to_move = random.sample(all_pairs, amount_to_move)
    for pair_to_move in pairs_to_move:
        source = os.path.join(src_pairs_dir, os.path.basename(pair_to_move))
        # destination = os.path.join(dst_pairs_dir, os.path.basename(pair_to_move))
        shutil.move(source, dst_pairs_dir, copy_function=shutil.copytree)



def copy_pairs_and_replace_drone_img(src_pairs_dir, dst_pairs_dir, new_drone_img, drone_scale):
    all_pairs = os.listdir(src_pairs_dir)
    for pair_to_move in all_pairs:
        source = os.path.join(src_pairs_dir, os.path.basename(pair_to_move))
        dest = os.path.join(dst_pairs_dir, os.path.basename(pair_to_move))
        shutil.move(source, dst_pairs_dir, copy_function=shutil.copytree)

        drone_image = cv.imread(new_drone_img)
        scale_fac = drone_scale  # percent of original size
        width = int(drone_image.shape[1] * scale_fac)
        height = int(drone_image.shape[0] * scale_fac)
        dim = (width, height)
        drone_image = cv.resize(drone_image, dim)
        cv.imwrite(os.path.join(dest, os.path.basename(new_drone_img)), drone_image)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facilitate ViT Descriptor point correspondences.')
    parser.add_argument('--full_pair', type=str, required=True, help='The dir of drone and ortho image pairs.')
    parser.add_argument('--all_patches_diff_drone', type=str, required=False, help='The dir of all backup patches pairs with an old drone image.')
    parser.add_argument('--all_patches', type=str, required=False, help='The dir of all backup patches pairs.')
    parser.add_argument('--root_dir', type=str, required=True, help='The root dir of image pairs.')
    parser.add_argument('--save_dir', type=str, required=True, help='The root save dir for image pairs results.')
    parser.add_argument('--load_size', default=224, type=int, help='load size of the input image.')
    parser.add_argument('--stride', default=4, type=int, help="""stride of first convolution layer. 
                                                                 small stride -> higher resolution.""")
    parser.add_argument('--model_type', default='dino_vits8', type=str,
                        help="""type of model to extract. 
                           Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                           vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""")
    parser.add_argument('--facet', default='key', type=str, help="""facet to create descriptors from. 
                                                                    options: ['key' | 'query' | 'value' | 'token']""")
    parser.add_argument('--layer', default=9, type=int, help="layer to create descriptors from.")
    parser.add_argument('--bin', default='True', type=str2bool, help="create a binned descriptor if True.")
    parser.add_argument('--thresh', default=0.05, type=float, help='saliency maps threshold to distinguish fg / bg.')
    parser.add_argument('--num_pairs', default=10, type=int, help='Final number of correspondences.')

    args = parser.parse_args()

    drone_scale = 0.125

    with torch.no_grad():

        full_pair = Path(args.full_pair)
        full_images = [x for x in full_pair.iterdir() if x.suffix.lower() in ['.jpg', '.png', '.jpeg']]
        ortho_img = ""
        drone_img = ""
        if os.path.basename(str(full_images[0])).startswith("ortho"):
            ortho_path = str(full_images[0])
            drone_img_path = str(full_images[1])
            drone_img = cv.imread(str(full_images[1]))
        elif os.path.basename(str(full_images[1])).startswith("ortho"):
            ortho_path = str(full_images[1])
            drone_img_path = str(full_images[0])
            drone_img = cv.imread(str(full_images[0]))
        out_dim = max(drone_img.shape[0], drone_img.shape[1]) // 8

        if args.all_patches_diff_drone:
            copy_pairs_and_replace_drone_img(args.all_patches_diff_drone, args.all_patches, drone_img_path, drone_scale)
            # sys.exit(-1)

        if args.all_patches:
            amount_to_move = 99
            # move_random_pairs(src_pairs_dir=args.all_patches, dst_pairs_dir=args.root_dir, amount_to_move=amount_to_move, best_fit_patch_name="pairs_2116")
            move_random_pairs(src_pairs_dir=args.all_patches, dst_pairs_dir=args.root_dir, amount_to_move=amount_to_move, best_fit_patch_name="pairs_2081")

        # divide_to_hierarchy_of_squares(full_pair, os.path.basename(ortho_path), out_dim=out_dim, stride=out_dim // 3, out_dir=os.path.join(root_dir), angles=[0, 90, 180, 270], drone_img_path=drone_img_path, drone_scale=drone_scale)
        # sys.exit(-1)

        # prepare directories
        root_dir = Path(args.root_dir)
        pair_dirs = [x for x in root_dir.iterdir() if x.is_dir()]
        save_dir = Path(args.save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        # lists that are used to find the best matching patch, based on the two scores.
        # index 0: point sim score, index 1: homography score, index 2: multiplied
        best_patch_score = [-np.inf, -np.inf, -np.inf]
        best_patch_name = [-np.inf, -np.inf, -np.inf]

        for pair_dir in tqdm(pair_dirs):
            curr_images = [x for x in pair_dir.iterdir() if x.suffix.lower() in ['.jpg', '.png', '.jpeg']]
            assert len(curr_images) == 2, f"{pair_dir} contains {len(curr_images)} images instead of 2."
            curr_save_dir = save_dir / pair_dir.name
            curr_save_dir.mkdir(parents=True, exist_ok=True)

            # compute point correspondences
            points1, points2, image1_pil, image2_pil, point_sim_score, homography_score = \
                find_correspondences(curr_images[0], curr_images[1], args.num_pairs, args.load_size, args.layer,
                                     args.facet, args.bin, args.thresh)
            mult_score = point_sim_score * homography_score
            # saving point correspondences
            file1 = open(curr_save_dir / "correspondence_A.txt", "w")
            file2 = open(curr_save_dir / "correspondence_Bt.txt", "w")
            for point1, point2 in zip(points1, points2):
                file1.write(f'{point1}\n')
                file2.write(f'{point2}\n')
            file1.write(f'mean matched points similarity score = {point_sim_score}\n')
            file2.write(f'mean matched points similarity score = {point_sim_score}\n')
            file1.write(f'homography inliers score = {homography_score}\n')
            file2.write(f'homography inliers score = {homography_score}\n')
            file1.write(f'multiplied scores = {mult_score}\n')
            file2.write(f'multiplied scores = {mult_score}\n')
            file1.close()
            file2.close()

            fig1, fig2 = draw_correspondences(points1, points2, image1_pil, image2_pil)
            fig1.savefig(curr_save_dir / f'{Path(curr_images[0]).stem}_corresp.png', bbox_inches='tight', pad_inches=0)
            fig2.savefig(curr_save_dir / f'{Path(curr_images[1]).stem}_corresp.png', bbox_inches='tight', pad_inches=0)
            plt.close('all')

            if point_sim_score > best_patch_score[0]:
                best_patch_score[0] = point_sim_score
                best_patch_name[0] = pair_dir.name
            if homography_score > best_patch_score[1]:
                best_patch_score[1] = homography_score
                best_patch_name[1] = pair_dir.name
            if mult_score > best_patch_score[2]:
                best_patch_score[2] = mult_score
                best_patch_name[2] = pair_dir.name
        print(f"Best matched patch based on point similarity score is {best_patch_name[0]}, with the score {best_patch_score[0]}")
        print(f"Best matched patch based on homography score is {best_patch_name[1]}, with the score {best_patch_score[1]}")
        print(f"Best matched patch based on multiplied score is {best_patch_name[2]}, with the score {best_patch_score[2]}")

