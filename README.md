# Hybrid LiDAR Reconstruction — Static Mesh + Dynamic Object Tracking

This project implements a complete LiDAR perception pipeline for a **stationary roadside LiDAR sensor**, including:

- Multi-sweep fusion into a dense, stable point cloud  
- Robust clustering with geometric + velocity refinement  
- Multi-object tracking using an EKF CTRV model  
- Static vs dynamic segmentation  
- **TSDF-based static mesh reconstruction**  
- Unified visualization of points, clusters, tracks, uncertainty, and mesh  

The pipeline produces a clean static reconstruction of the environment and overlays dynamic actors with long-term track identities — enabling HD mapping, roadside perception, and foundations for future SLAM systems.



https://github.com/user-attachments/assets/9600b923-4259-4231-85cc-455184c2291e



## 🚀 Implemented Features

### **1. Multi-Sweep Fusion**
A short temporal window of LiDAR frames is transformed into world coordinates and fused into a “super-sweep”:

- Higher density & reduced noise  
- Better cluster boundaries  
- More stable tracking  
- Stronger TSDF integration  

Optional ground removal and voxel downsampling increase stability even further.

---

### **2. Two-Phase Clustering**
#### **A. Geometry-First DBSCAN**
Pure geometric segmentation:

- Works for vehicles, pedestrians, walls, infrastructure  
- Removes noise and tiny fragments  
- Produces stable initial clusters  

#### **B. Velocity-Based Split & Merge**
Uses local motion to correct geometry-only errors:

- **Split** clusters if motion disagrees  
- **Merge** clusters if motion + shape match  
- Prevents:
  - Vehicles merging with buildings  
  - Large objects splitting into multiple pieces  
  - Slow objects “absorbing” into static background  

Result: clean, physically consistent clusters.

---

### **3. Multi-Object EKF Tracking (CTRV Model)**

State: $[p_x, p_y, \phi, v, \omega]$ for the position, yaw (orientation around Z-axis) and velocity alongside angular velocity.
Features:

- Analytic CTRV prediction  
- Covariance propagation  
- Mahalanobis + Euclidean gating  
- Hungarian global association  
- Track lifecycle (spawn → confirm → prune)  
- Shape & velocity consistency checks  
- Smooth long-term track IDs  

---

### **4. Static vs Dynamic Segmentation**

Static objects are identified using:

- EKF velocity  
- Short & long motion windows  
- Centroid drift  
- Shape overlap across frames  
- Temporal hysteresis  

**Static points → TSDF**  
**Dynamic objects → tracked actors**

This cleanly separates the world into a static mesh base layer and dynamic actors on top.

---

### **5. TSDF-Based Static Mesh Reconstruction**

A simple but fully functional TSDF implementation:

- World-aligned voxel grid  
- Truncated SDF computation  
- Weight accumulation  
- Integration of **static points only**  
- Mesh extraction via Marching Cubes  
- Output as an Open3D mesh  

This produces a watertight reconstruction of buildings, walls, poles, facades, and other immobile structures while fully excluding dynamic objects.

---

### **6. Unified 3D Visualizer (Mesh + Tracks + Clusters)**

An interactive Open3D viewer supports:

- Static TSDF mesh visualization  
- Per-frame LiDAR point clouds  
- Cluster bounding boxes  
- Track trajectories  
- EKF covariance ellipses  
- Velocity / intensity coloring  
- Keyboard navigation & overlay toggles  

**Key shortcuts:**

- `A / ←` – previous frame  
- `D / →` – next frame  
- `M` – toggle static mesh  
- `B` – toggle bounding boxes  
- `1–6` – visualization modes  
- `W` – toggle wireframe if mesh visualized
- `R` – reset view  
- `Q / ESC` – quit  

This provides an end-to-end view of the entire reconstruction pipeline.

---

## Dataset setup

This project has originally been developed to be used with the _UrbanIng-V2X_ dataset, and therefore the loader follows its folder structure definition. 

```
datasets/
└── UrbanIng-V2X/
    └── dataset/
        └── 20241126_0017_crossing1_00/
            ├── lidar1/
            │   ├── <timestamp_1>.pcd
            │   ├── <timestamp_2>.pcd
            │   └── ...
            ├── lidar2/
            │   ├── <timestamp_1>.pcd
            │   ├── <timestamp_2>.pcd
            │   └── ...
            ├── calibration.json
            └── timesync_info.csv
```


- The folder `20241126_0017_crossing1_00` corresponds to a **single recording sequence**.  
- Each subfolder (`lidar1/`, `lidar2/`, …) represents a **separate LiDAR sensor**, containing timestamped point cloud files.  
- `calibration.json` contains the extrinsic transformations needed to bring all LiDAR sensors into a **common world coordinate frame**.  
- `timesync_info.csv` contains metadata for **temporal synchronization** (e.g., timestamp offsets) between sensors.

This layout allows the loader to fuse multiple sensors into a time-aligned, unified point cloud stream.

For usage with different datasets, some transformation script should be created.


## 🤝 About LLM usage

This project is developed via a hybrid workflow between the human author (me :D) and ChatGPT (GPT-5 / GPT-5.1):

- **Design:** I set goals, ChatGPT refines algorithms and architecture.  
- **Coding:** I implement and test; ChatGPT provides refactors, structure, and numerical help.  
- **Math:** I write initial math implementation; ChatGPT assists errors and more tedius mathematical derivations.  
- **Iteration:** Rapid feedback cycles accelerate development dramatically.

Overall the project has been a very successful use of LLM assisted programming in an applied mathematics/statistical setting.

---

## 🔮 Future Work

- Unit tests & regression tests  
- Parameter auto-tuning  
- Improved TSDF integration (raycasting, multi-resolution, decay)  
- Per-object dynamic TSDF volumes  
- Real-time data streaming  
- Back-end SLAM & loop closure  

---

## 📚 References

- Curless & Levoy – *A Volumetric Method for Building Complex Models from Range Images*  
- Sekaran et al. – *UrbanIng-V2X: Cooperative Perception Dataset*  
- CTRV / EKF motion models  
