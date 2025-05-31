# 🖐️ Hand Reconstruction using MANO and Tube Mesh

This project reconstructs a 3D hand mesh from a single 2D image using a two-step process:
1. Generate a tube-shaped proxy mesh from the image.
2. Align it with a 3D statistical hand model (MANO) for accurate reconstruction.

The approach avoids optimization-based fitting and instead relies on geometric alignment of sampled surface points.

---

## 📦 Requirements

Install the following Python packages (Python 3.8+ recommended):

```bash
pip install numpy trimesh matplotlib open3d
```

---

## 📁 Folder Structure

```
hand_reconstruction/
├── mano/                        # Place MANO model file here
│   └── MANO_RIGHT.pkl
├── hand.jpg                    # Input image (hand)
├── tube_hand.py               # Creates tube mesh from image
├── align_tube_to_mano.py      # Aligns tube mesh to MANO
├── final_tube.obj             # Output of tube_hand.py
├── final_hand.obj             # Final aligned hand mesh
└── README.md
```

---

## 🔁 Reproduction Steps

### 1️⃣ Download MANO model

- Download from: https://mano.is.tue.mpg.de/download.php
- Unzip and place the file `MANO_RIGHT.pkl` inside the `./mano/` folder.

```bash
./mano/MANO_RIGHT.pkl
```

> 📂 If the `mano/` folder does not exist, create it manually.

---

### 2️⃣ Generate tube hand mesh

```bash
python tube_hand.py -i hand.jpg -o final_tube.obj
```

- `-i`: Input image of a hand (e.g., `hand.jpg`)  
- `-o`: Output proxy mesh (OBJ format)

This step produces a rough 3D tube-like mesh representing the hand's shape.

---

### 3️⃣ Align to MANO mesh

```bash
python align_tube_to_mano.py \
  --tube-obj   final_tube.obj \
  --mano-dir   ./mano \
  --out-obj    final_hand.obj \
  --n-points   20000 \
  --max-corr   0.05
```

Arguments:

- `--tube-obj`: Input tube mesh (`.obj`)  
- `--mano-dir`: Directory with `MANO_RIGHT.pkl`  
- `--out-obj`: Final reconstructed mesh  
- `--n-points`: Number of points for surface sampling  
- `--max-corr`: Max correspondence threshold for alignment

---

## ✅ Output

After step 3, you will obtain:

```
final_hand.obj  ← reconstructed hand mesh
```


---

## 🔗 Reference

- [MANO: Articulated Hand Model](https://mano.is.tue.mpg.de/)  
  > Romero et al., "Embodied Hands: Modeling and Capturing Hands and Bodies Together", SIGGRAPH Asia 2017


---

## © License

This code is released under the MIT License.  
Note: The MANO model is under a separate license and must be obtained directly from MPI-IS.
