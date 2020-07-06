import cv2
import numpy as np


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def norm_face(img, pts, crop_type='default', max_size=500):
    if len(pts) > 2:
        pts = [pts[36], pts[45]]
    pts = np.array(pts)
    v = pts[1] - pts[0]
    v_len = np.sqrt(np.sum(v ** 2))
    angle_cos = v[0] / v_len
    angle_sin = v[1] / v_len
    cx = pts[0][0]
    cy = pts[0][1]

    if type(crop_type) is tuple and len(crop_type) == 4:
        kx, ky, target_size_w, target_size_h = crop_type
    elif crop_type == 'default':
        kx = 0.32
        ky = 0.395
        target_size_w = min(img.shape[1], max_size)
        target_size_h = int(target_size_w * 5 / 4.0)
    elif crop_type == 'exp01':
        kx = 0.33
        ky = 0.44
        target_size_w = min(img.shape[1], max_size)
        target_size_h = int(target_size_w * 5 / 4.0)
    elif crop_type == 'face':
        kx = 19.0 / 128.0
        ky = 31.0 / 128.0
        target_size_w = 128
        target_size_h = 128
    elif crop_type == 'head':
        kx = 41.0 / 124.0 #128.0
        ky = 59.0 / 124.0 #128.0
        target_size_w = 224
        target_size_h = 224
    elif crop_type == '3d_landmarks':
        kx = 70.0 / 240.0
        ky = 73.0 / 240.0
        target_size_w = 160
        target_size_h = 160
    elif crop_type == 'hopenet':
        kx = 41.0 / 128.0 # 0.36
        ky = 62.0 / 128.0 # 0.52
        target_size_w = 224
        target_size_h = 224

    target_dist = target_size_w * (1 - 2 * kx)
    scale = target_dist / v_len

    alpha = scale * angle_cos
    beta = scale * angle_sin
    shift_x = (1 - alpha) * cx - beta * cy
    shift_y = beta * cx + (1 - alpha) * cy

    tx = kx * target_size_w
    ty = ky * target_size_h
    dx = tx - (cx * alpha + cy * beta + shift_x)
    dy = ty - (-cx * beta + alpha * cy + shift_y)

    M = [
        [alpha, beta, shift_x + dx],
        [-beta, alpha, shift_y + dy]
    ]
    M = np.array(M)

    return cv2.warpAffine(img, M, (target_size_w, target_size_h)), M