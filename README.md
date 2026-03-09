## Project Topic:

     Monocular Visual Odometry to Visual-Inertial Odometry on TUM VI

## Supervisor:

    Zaar Khizar

## Project OverView

    This repository contains the implementation of a geometry based VIO pipeline, developed as part of a Master's Course in Computer Vision and Robotics. The project transitions from a Monocular Visual Odometry (VO) baseline to a tightly-coupled VIO system using IMU preintegration and sliding-window optimization.

## Project Goal

    The objective of this project is to estimate a robot's ego-motion from camera images alone, then extend it to Visual-Inertial Odometry by using,
        -Monocular VO by performing a feature-based tracking (ORB/SIFT) and epipolar geometry to recover camera motion.
        -Visual-Inertial Fusion to integrate high-frequency IMU data via preintegration to resolve scale ambiguity and improve trajectory robustness.

## Folder Structure

    vio-project/

├── src/
│ ├── vo/ #Part 1 1: Monocular Visual Odometry
│ │ ├── feature_detection.py # ORB/SIFT feature detection
│ │ ├── feature_matching.py # BFMatcher + ratio test
│ │ ├── essential_matrix.py # Essential matrix + RANSAC
│ │ ├── tracking.py # Optical flow / PnP tracking
│ │ └── pipeline.py # Main VO pipeline
│ ├── vio/ #Part 2: Visual-Inertial Odometry
│ │ ├── imu_preintegration.py # IMU preintegration between keyframes
│ │ ├── sliding_window.py # Sliding-window Bundle Adjustment
│ │ └── pipeline.py # Main VIO pipeline
│ ├── utils/
│ │ ├── data_loader.py # TUM VI data loading
│ │ └── visualization.py # Trajectory plotting
│ └── evaluation/
│ ├── ate.py # Absolute Trajectory Error
│ └── rpe.py # Relative Pose Error
├── configs/
│ └── tum_vi.yaml # Camera intrinsics, IMU noise params
├── data/ # Download TUM VI sequences here (not tracked)
├── results/
│ ├── plots/ # Trajectory plots (PNG)
│ ├── trajectories/ # Output .txt files in TUM format
│ └── tables/ # ATE/RPE metrics (CSV)
├── report/ # IEEE paper source
├── notebooks/ # Jupyter exploration notebooks
├── tests/ # Unit tests
├── requirements.txt
└── README.md

## Tech Stack

- Language: Python 3.x (Optional C++ for optimization)
- Libraries: OpenCV, NumPy, SciPy, Matplotlib, Open3D

## Team

- Ajepe Fiyinfoluwa
- Jordan Fogan
