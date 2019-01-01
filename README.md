# BASIC STEREO SLAM

The front end of the SLAM system is completed and is tested on Ubuntu16.04, AMD A6 & 4GB RAM.

# Prerequisites
### Opencv (Open Source Computer Vision Library)
We use [Opencv](https://opencv.org/) 3.4.5 to do most of tasks in this system, such as features detection, matching, tracking and mapping. The tutorial of installing opencv can be found in: https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html

### Opencv_contrib
We use some nonfree modules of opencv contrib, such as SIFT and BRIEF features. The tutorial of installing opencv contrib can be found in: https://github.com/opencv/opencv_contrib. A [blog](https://blog.csdn.net/dell5200/article/details/85547460) is also provided for your consideration.

### VTK (Visualization Toolkit)
We install VTK 8.1.1 for visulization functions of opencv (opencv viz). The tutorial of VTK can be found in: https://www.vtk.org. A blog is also provided [here](https://blog.csdn.net/dell5200/article/details/81142951) as a tutorial for the configuration of VTK.

### GSL (GNU Scientific Library)
The GSL is used for calculating linear algebra. You can download it in: ftp://ftp.gnu.org/gnu/gsl/. A blog is provided [here](https://blog.csdn.net/dell5200/article/details/81058418) as a tutorial for installation and configuration.

### Library testing
To make sure that all required libraries are intalled correctly, a [repository](https://github.com/GentleDell/BasicCVProgram) is provided to test these libraries. 

# Results
In July 2018, the front end has been completed and has been tested on KITTI dataset 00. A brief result of the front end is shown below: 
[demo 1](https://github.com/GentleDell/StereoSLAM/blob/master/KITTI_00.png)

# What to do
1. Loopclosing
In recent years, many matchine learning methods perform very well in scenaries recognition, so we would like to use matchine learning technques to realize a robust loop closure.
2. Optimization
In ORBSLAM, g2o is used for optimization. In google's catographer and HKUST-Aerial-Robotics VINS, ceres is used. So, We would like to use g2o or ceres as our solver. 

