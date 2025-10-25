# Evolving 3D Mesh & Object Tracking (Static → Dynamic LiDAR)

This project reconstructs evolving 3D geometry from a *stationary LiDAR sensor* observing a dynamic scene.  
It detects and tracks moving objects, estimates their rigid motion, and fuses both static and dynamic data into watertight 3D meshes using **TSDF reconstruction**.

---

## 🧭 Overview

- **Input:** Time-indexed LiDAR sweeps of a fixed scene (street corner, intersection, etc.)  
- **Output:**  
  - A static background mesh reconstructed via global TSDF  
  - Per-object TSDF volumes, motion-compensated using tracked $SE(3)$ transforms  
  - A composite 3D scene that evolves realistically over time

---

## 📂 Sprint 1 – Foundations

### [P0] T1: Data Loading & Preprocessing
- Implement dataset loader (SemanticKITTI / AI4CE Roadside / nuScenes mini)  
- Ground-plane removal (RANSAC)  
- Voxel downsampling (0.05–0.1 m)

### [P0] T2: Clustering & Object Detection
- DBSCAN clustering on non-ground points  
- Filter tiny clusters  
- Compute centroid, bounding box, and extent

---

## 📂 Sprint 2 – Motion & Tracking

### [P0] T3: Scene Flow Estimation
- kNN graph per frame pair  
- Per-point nearest-neighbor matching  
- Variational refinement:

$$
E(v) = \sum_i \|x_i' - (x_i + v_i)\|^2 + \lambda \sum_{(i,j)\in\mathcal{N}} \|v_i - v_j\|^2
$$

- Visualize as color-coded velocity fields

### [P0] T4: Rigid Motion Estimation (SE(3))
- Fit per-cluster rigid motion via ICP  
- Represent transformations using the exponential map $T = \exp(\hat{\xi}) \in SE(3)$  
- Estimate velocity and yaw rate  
- Store residuals as motion confidence

### [P0] T5: Multi-Object Tracking
- State vector $[x, y, dx, dy, \psi, d\psi]$  
- EKF or UKF on $SE(3)$  
- Data association: Mahalanobis gating + Hungarian  
- Track lifecycle: spawn / maintain / terminate

---

## 📂 Sprint 3 – Mesh Reconstruction

### [P0] T6: Static Background TSDF
Integrate all *non-moving* points into a **global TSDF volume**.  

Energy formulation:

$$
E(\Phi) = \sum_k w_k \big(s_k - \Phi(x_k)\big)^2 + \lambda \int \|\nabla \Phi\|^2 \, dx
$$

Solve as sparse linear system:

$$
(S^\top W S + \lambda L)\Phi = S^\top W s
$$

Extract surface via Marching Cubes ($\Phi = 0$) → watertight background mesh.

---

### [P0] T7: Dynamic Object TSDFs (Rigid-Compensated Fusion)

Each tracked object $i$ has:
- A local frame $\mathcal{F}_i$ with pose $T_i(t)\in SE(3)$  
- A local TSDF $\Phi_i(x)$ defined in that frame

For every frame:

$$
\Phi_i^{(t+1)}(x)
= \text{Fuse}\big(\Phi_i^{(t)}(x),
\, s_{k,t+1} - \Phi_i(T_i^{-1}(t+1) x_{k,t+1})\big)
$$

This **motion-compensated fusion** yields sharp, consistent meshes of moving objects.

---

### [P1] T8: Joint Optimization (Shape + Pose)

Refine both geometry and motion jointly:

$$
E(\Phi_i, T_i) =
\sum_{t,k} \rho\!\Big(s_{k,t} - \Phi_i(T_i^{-1}(t) x_{k,t})\Big)
+ \lambda \int \|\nabla \Phi_i\|^2 \, dx
$$

Alternate:
1. **Pose update:** minimize over $T_i$ (ICP on implicit surface)  
2. **Shape update:** minimize over $\Phi_i$ (variational TSDF solve)

---

## 📂 Sprint 4 – Visualization & Composition

### [P0] V1: Evolving Scene Renderer
- Extract meshes for background + each object  
- Apply current $T_i(t)$ to object meshes  
- Composite and render in Open3D (or Blender) for time-lapse visualization

### [P1] V2: Per-Vertex Velocity Coloring
- Store velocity from scene flow or tracking  
- Color vertices by magnitude or direction

### [P1] V3: Confidence Visualization
- Map ICP residuals or fusion weights to vertex color for debugging

---

## 📂 Sprint 5 – Evaluation & Metrics

### [P0] E1: Tracking Metrics
- IDF1, MOTA, ID switches

### [P0] E2: Velocity Metrics
- RMSE vs ground truth (if available)  
- Temporal consistency (Δ displacement)

### [P0] E3: Mesh Metrics
- Chamfer distance, completeness, voxel overlap, vertex churn

---

## 📂 Sprint 6 – Mathematical Enhancements

### [P1] M1: SE(3) Lie-Group Filtering
- Implement EKF with exponential-map updates  
- Compare to Euclidean filtering

### [P1] M2: Helmholtz Scene-Flow Decomposition
Decompose flow:

$$
v(x) = \nabla \phi(x) + \nabla \times A(x)
$$

Visualize divergence (compression) and curl (rotation).

### [P2] M3: Topology-Aware Mesh Regularization
- Persistent homology checks on TSDF slices  
- Flag spurious tunnels or holes

### [P2] M4: Uncertainty Propagation
- Use ICP residuals to estimate covariance of $T_i$  
- Confidence-weighted TSDF fusion

---

## 📂 Stretch Goals
- Multi-sensor fusion (2+ LiDARs)  
- Octree TSDF (adaptive resolution)  
- Radar or optical flow integration for direct velocity priors  
- Differentiable TSDF fusion for gradient-based optimization

---

## ✅ Deliverables
- Stable multi-object tracking with $SE(3)$ trajectories  
- Static background mesh and motion-compensated dynamic object meshes  
- Quantitative metrics and demo video of full evolving 3D scene  
- Clear mathematical foundations linking rigid motion and implicit surface optimization

---

## 🧠 References
- Curless & Levoy, *Volumetric Method for Building Complex Models from Range Images*, 1996  
- Newcombe et al., *KinectFusion*, 2011  
- Whelan et al., *ElasticFusion*, 2015  
- Zhou et al., *DynamicFusion*, 2015  
- Chern & Wang, *Computing Minimal Surfaces with Differential Forms*, 2022
