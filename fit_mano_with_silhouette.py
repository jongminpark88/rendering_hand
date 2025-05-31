#!/usr/bin/env python3
"""
fit_mano_with_silhouette.py

Single‐image → 3D 관절 + 2D 실루엣 손실 → MANO 포즈 피팅

Usage:
  python fit_mano_with_silhouette.py \
    --image    hand.jpg \
    --mano-dir ./mano \
    --out-obj  hand_mano_final.obj \
    --steps    500 \
    --w3d      1.0 \
    --w2d      10.0

Dependencies:
  pip install torch torchvision opencv-python mediapipe smplx trimesh
  pip install fvcore iopath
  pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
"""
import cv2, torch, argparse, numpy as np, trimesh, mediapipe as mp
from smplx import MANO
from pytorch3d.renderer import (
    PerspectiveCameras, MeshRenderer, MeshRasterizer,
    SoftSilhouetteShader, RasterizationSettings
)
from pytorch3d.structures import Meshes

# MediaPipe world‐landmarks → MANO 16관절
JOINT_MAP = [0,1,2,3, 5,6,7, 9,10,11, 13,14,15, 17,18,19]

mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)

def get_world_joints(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = mp_hands.process(rgb)
    if not res.multi_hand_world_landmarks:
        raise RuntimeError("손 world‐landmarks를 못 찾았습니다.")
    pts = np.array([[p.x,p.y,p.z] for p in res.multi_hand_world_landmarks[0].landmark], dtype=np.float32)
    return pts

def get_silh_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    m = cv2.morphologyEx(m,cv2.MORPH_CLOSE,kernel)
    return (m/255.).astype(np.float32)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image","-i",    required=True)
    p.add_argument("--mano-dir","-m", required=True)
    p.add_argument("--out-obj","-o",  required=True)
    p.add_argument("--steps",  type=int,   default=500)
    p.add_argument("--lr",      type=float, default=0.02)
    p.add_argument("--w3d",     type=float, default=1.0)
    p.add_argument("--w2d",     type=float, default=10.0)
    args = p.parse_args()

    img = cv2.imread(args.image)
    H,W = img.shape[:2]

    # targets
    world_pts = get_world_joints(img)                  # (21,3)
    tgt3d = torch.from_numpy(world_pts[JOINT_MAP]).unsqueeze(0)  # (1,16,3)
    mask2d = get_silh_mask(img)                       # (H,W)

    # MANO
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mano = MANO(model_path=args.mano_dir, use_pca=False, is_rhand=True, flat_hand_mean=True).to(device)
    faces = torch.from_numpy(mano.faces.astype(np.int64)).to(device)

    # Silhouette renderer
    cameras = PerspectiveCameras(
        focal_length=((W,W),), principal_point=((W/2,H/2),),
        image_size=((H,W),), device=device
    )
    rast = RasterizationSettings(image_size=(H,W), blur_radius=0.0, faces_per_pixel=10)
    sil_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=rast),
        shader=SoftSilhouetteShader()
    )

    # params
    pose   = torch.zeros((1,48), requires_grad=True, device=device)
    betas  = torch.zeros((1,10), requires_grad=True, device=device)
    transl = torch.zeros((1,3),  requires_grad=True, device=device)
    opt = torch.optim.Adam([pose,betas,transl], lr=args.lr)

    tgt3d = tgt3d.to(device)
    mask2d_t = torch.from_numpy(mask2d).to(device).unsqueeze(0)  # (1,H,W)

    # simple norm for 3D
    norm3d = tgt3d.norm(dim=-1, keepdim=True)

    for i in range(args.steps):
        opt.zero_grad()
        out = mano(global_orient=pose[:,:3], hand_pose=pose[:,3:], betas=betas, transl=transl)
        # 3D joint loss
        pred3d = out.joints[:, :len(JOINT_MAP), :]
        loss3d = torch.nn.functional.mse_loss(pred3d/norm3d, tgt3d/norm3d)
        # 2D silhouette loss
        sil_pred = sil_renderer(Meshes(verts=out.vertices, faces=[faces]))[...,3]
        loss2d = torch.nn.functional.binary_cross_entropy(sil_pred, mask2d_t)

        loss = args.w3d * loss3d + args.w2d * loss2d
        loss.backward()
        opt.step()
        if (i+1)%50==0:
            print(f"[{i+1:03d}/{args.steps}] 3D={loss3d.item():.5f} 2D={loss2d.item():.5f}")

    # export
    final_verts = out.vertices.detach().cpu().squeeze().numpy()
    mesh = trimesh.Trimesh(vertices=final_verts, faces=mano.faces, process=False)
    mesh.export(args.out_obj)
    print("▶ Saved:", args.out_obj)

if __name__=="__main__":
    main()
