#!/usr/bin/env python3
"""
recon_hand.py: Single-image 3D hand reconstruction using MANO.

Requirements:
    pip install numpy==1.23.5 torch torchvision opencv-python mediapipe smplx trimesh chumpy
"""

import os
import cv2
import numpy as np
import torch
import trimesh
import mediapipe as mp
from smplx import MANO

# MediaPipe 21 landmarks 중에서 MANO의 16 joints에 대응되는 인덱스
JOINT_INDICES = [0,         # wrist
                 1,2,3,     # thumb: CMC, MCP, IP (drop tip=4)
                 5,6,7,     # index: MCP, PIP, DIP (drop tip=8)
                 9,10,11,   # middle: MCP, PIP, DIP (drop tip=12)
                 13,14,15,  # ring: MCP, PIP, DIP (drop tip=16)
                 17,18,19]  # pinky: MCP, PIP, DIP (drop tip=20)


def get_hand_keypoints(image_path):
    """
    MediaPipe로 2D 손 keypoint 21개를 추출합니다.
    Returns:
        keypoints_2d: np.ndarray shape (21,2)
        img_size: (height, width)
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"이미지 파일이 없습니다: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"이미지를 읽을 수 없습니다: {image_path}")
    h, w = img.shape[:2]

    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if not res.multi_hand_landmarks:
            raise RuntimeError("손이 인식되지 않았습니다.")

        kps = []
        for lm in res.multi_hand_landmarks[0].landmark:
            kps.append([lm.x * w, lm.y * h])
    return np.array(kps), (h, w)


def orthographic_project(joints3d):
    """
    Orthographic projection: (x,y,z)->(x,y)
    """
    return joints3d[..., :2]


def fit_mano(keypoints_2d, img_size, mano_dir):
    """
    2D keypoints(21) 중 16개를 골라 MANO 파라미터를 Adam으로 최적화합니다.
    Returns:
        verts: (778,3) np.ndarray
        faces: (1538,3) np.ndarray
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MANO 레이어
    mano_layer = MANO(
        model_path=mano_dir,
        use_pca=False,
        is_rhand=True,
        flat_hand_mean=True
    ).to(device)

    # 파라미터 초기화
    pose_params  = torch.zeros((1, 48), requires_grad=True, device=device)  # 16 joints×3
    shape_params = torch.zeros((1, 10), requires_grad=True, device=device)
    transl       = torch.zeros((1, 3), requires_grad=True, device=device)

    optimizer = torch.optim.Adam([pose_params, shape_params, transl], lr=0.01)

    # 21개 중 16개만 골라 텐서화
    kps16 = keypoints_2d[JOINT_INDICES]  # (16,2)
    kps_2d = torch.tensor(kps16, dtype=torch.float32, device=device).unsqueeze(0)  # (1,16,2)

    h, w   = img_size
    norm_f = torch.tensor([w, h], device=device)

    for step in range(300):
        out = mano_layer(
            global_orient=pose_params[:, :3],  # wrist
            hand_pose   =pose_params[:, 3:],   # 15 joints
            betas       =shape_params,
            transl      =transl
        )
        joints3d = out.joints                       # (1,16,3)
        proj2d    = orthographic_project(joints3d)  # (1,16,2)

        # -1~1 정규화
        tgt = (kps_2d  / norm_f) * 2 - 1
        src = (proj2d / norm_f) * 2 - 1
        loss = torch.nn.functional.mse_loss(src, tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"[{step:03d}] loss = {loss.item():.6f}")

    verts = out.vertices.detach().cpu().numpy().squeeze()  # (778,3)
    faces = mano_layer.faces                             # (1538,3)
    return verts, faces


def save_as_obj(vertices, faces, out_path):
    """
    trimesh로 .obj 포맷 저장
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(out_path)
    print(f"3D mesh saved to: {out_path}")


if __name__ == "__main__":
    # 프로젝트 루트에 이 스크립트, hand.jpg, mano/MANO_RIGHT.pkl 이 있어야 합니다.
    mano_dir   = "./mano"
    image_path = "./hand.jpg"
    out_obj    = "./hand_mesh.obj"

    kps2d, img_size = get_hand_keypoints(image_path)
    verts, faces   = fit_mano(kps2d, img_size, mano_dir)
    save_as_obj(verts, faces, out_obj)
