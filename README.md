# Hybrid LiDAR Reconstruction — Static Mesh + Dynamic Object Tracking

This repository implements a full LiDAR processing pipeline for a **stationary roadside LiDAR sensor**, capable of:

- Fusing multiple temporal sweeps into one dense frame  
- Extracting robust geometric clusters  
- Refining clusters using velocity to split/merge dynamic and static structures  
- Tracking objects using an EKF model  
- Separating persistent static geometry from moving objects  
- Preparing the foundation for a **hybrid TSDF-based static mesh + dynamic object reconstruction**

The medium-term goal is to produce a clean static mesh of the environment (walls, poles, buildings, infrastructure) and layer dynamic objects on top using either point-cloud fusion or per-object TSDF.

---

# Overview

This system processes raw LiDAR sweeps into a structured dynamic scene:

```
Raw Sweeps → Multi-Sweep Fusion → Clustering  
           → Velocity Refinement → EKF Tracking  
           → Static vs Dynamic → TSDF Reconstruction
```

The pipeline is built to support roadside perception, HD mapping, and dynamic environment modeling with high spatial and temporal resolution.

---

# Current System Features (Implemented)

## 1. Multi-Sweep Fusion

The system fuses a short temporal window of LiDAR sweeps into a single “super-sweep”:

- Several sweeps transformed into world coordinates  
- Combined for increased density  
- Ground removal and voxel downsampling applied afterward  
- Improves cluster stability and tracking performance significantly

**Benefits:**
- Cleaner cluster boundaries  
- Better separation between cars and infrastructure  
- Higher point coverage for TSDF later  

---

## 2. Two-Phase Clustering (Geometry → Velocity Refinement)

### A. Geometry-First DBSCAN

Initial clusters are purely geometric:

- DBSCAN using Euclidean distance  
- Captures large infrastructure and vehicle shapes  
- Removes tiny clusters and noise  
- Serves as a stable initial segmentation step

### B. Velocity-Based Split & Merge

After computing point-wise local motion via nearest-neighbor scene flow:

- **Split** clusters when subregions have significantly different velocities  
- **Merge** clusters when shapes and velocities agree  
- Prevents common failure cases such as:
  - Cars merging with walls or poles  
  - Vehicles split into multiple fragments  
  - Stationary infrastructure fused with moving objects  

The result is a clean, stable clustering suitable for tracking.

---

## 3. EKF Tracking (CTRV Model)

Each cluster feeds into a multi-object tracker using an Extended Kalman Filter with the CTRV (constant-turn-rate-and-velocity) motion model.

State vector:

```
[px, py, yaw, v, omega]
```

Implemented:

- Analytic CTRV prediction model  
- EKF covariance propagation  
- Mahalanobis gating  
- Hungarian global data association  
- Track lifecycle management (spawn / confirm / kill)  
- Shape consistency validation  
- Velocity smoothing  

This produces stable track identities over long sequences and dense traffic.

---

## 4. Static vs Dynamic Segmentation

Using EKF velocities and temporal consistency:

- Objects with near-zero velocity become **static**
- Others with persistent motion become **dynamic**
- Static points are accumulated for global TSDF  
- Dynamic points are grouped per object for later reconstruction  

This yields a clean separation between static infrastructure and moving objects.

---

## 5. TSDF Mesh

---


# About the Collaboration (Human + ChatGPT 5/5.1)

This project is developed as a **collaboration between the human author and ChatGPT models (GPT-5 and GPT-5.1)**.

### How the Collaboration Works

- **Design:** I define goals, architecture, and constraints. ChatGPT proposes alternative algorithms, and improvements.  
- **Code:** I write and test code, ChatGPT is used to refactor code, clarify logic, and some numerical optimization.  
- **Math:** ChatGPT assists with more tedius algebraic derivations (e.g., CTRV Jacobians) and some data transforms.  
- **Iteration:**  
  - I test, evaluate output and adjust direction.  
  - ChatGPT refines ideas, rewrites documentation, and provides alternatives.  
  - System evolves through cyclic co-design.

This collaborative style of working is used to drastically increase velocity.


# Future work 

- Testing of modules
- Automated optimization of parameters

# References

- Curless & Levoy - *A Volumetric Method for Building Complex Models from Range Images* 
- Sekaran et al. - *UrbanIng-V2X: A Large-Scale Multi-Vehicle, Multi-Infrastructure Dataset Across Multiple Intersections for Cooperative Perception*
- CTRV / EKF motion models  
