#!/usr/bin/env python3
"""
tube_hand.py

Simple hand mesh with cylinders connecting MediaPipe landmarks.
"""

import argparse
import cv2
import numpy as np
import mediapipe as mp
import trimesh
from trimesh.creation import cylinder
from trimesh.geometry import align_vectors

# MediaPipe 21 landmarks 중 뼈대 연결 쌍
BONES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

def detect_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        res = hands.process(rgb)
        if not res.multi_hand_landmarks:
            raise RuntimeError("No hand detected.")
        lm = res.multi_hand_landmarks[0]
        pts = []
        for lmk in lm.landmark:
            x_px = lmk.x * w
            y_px = lmk.y * h
            z_rel = -lmk.z * w
            pts.append([x_px, y_px, z_rel])
        return np.array(pts, dtype=np.float32)

def build_tube_mesh(landmarks):
    # 평균 bone 길이 기반 radius 설정
    lengths = [np.linalg.norm(landmarks[j] - landmarks[i]) for i, j in BONES]
    avg_len = float(np.mean(lengths))
    radius = avg_len * 0.15

    parts = []
    for i, j in BONES:
        p0, p1 = landmarks[i], landmarks[j]
        vec = p1 - p0
        length = np.linalg.norm(vec)
        if length < 1e-6:
            continue

        # unit cylinder along z축 생성
        cyl = cylinder(radius=radius, height=length, sections=24)

        # z축을 p1-p0 방향으로 정렬
        R_full = align_vectors([0,0,1], vec / length)
        R = R_full[:3, :3]

        # transform: 회전 + 평행이동
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = (p0 + p1) / 2.0
        cyl.apply_transform(T)

        parts.append(cyl)

    if not parts:
        raise RuntimeError("No cylinders created.")
    return trimesh.util.concatenate(parts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input image of a hand")
    parser.add_argument("--output", "-o", required=True, help="Output OBJ file path")
    args = parser.parse_args()

    # 1) 3D landmarks 추출
    landmarks = detect_landmarks(args.input)

    # 2) 튜브 메쉬 생성
    mesh = build_tube_mesh(landmarks)

    # 3) OBJ로 저장
    mesh.export(args.output)
    print(f"Saved mesh → {args.output}")

if __name__ == "__main__":
    main()
