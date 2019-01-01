# BASIC STEREO SLAM

The front end of the system has been completed and was tested on a Laptop with AMD A6 & 4GB RAM. This project is only based on computer vision functions provided by OpenCV.

# Prerequisites
### Opencv (Open Source Computer Vision Library)
We use [Opencv](https://opencv.org/) 3.4.5 to do most of tasks, such as features detection, matching, tracking and mapping. The tutorial of installing OpenCV can be found in: https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html

### Opencv_contrib
We use some nonfree modules of OpenCV Contrib, such as SIFT and BRIEF features. The tutorial of installing OpenCV Contrib can be found in: https://github.com/opencv/opencv_contrib. A [blog](https://blog.csdn.net/dell5200/article/details/85547460) is also provided for your consideration.

### VTK (Visualization Toolkit)
We install VTK 8.1.1 for visualization functions of OpenCV (OpenCV viz). The tutorial of VTK can be found in: https://www.vtk.org. A blog is also provided [here](https://blog.csdn.net/dell5200/article/details/81142951) as a tutorial for the configuration of VTK.

### GSL (GNU Scientific Library)
The GSL is used for calculating linear algebra. You can download it in: ftp://ftp.gnu.org/gnu/gsl/. A blog is provided [here](https://blog.csdn.net/dell5200/article/details/81058418) as a tutorial for installation and configuration.

### Library testing
To make sure that all required libraries are installed correctly, a [repository](https://github.com/GentleDell/BasicCVProgram) is provided to test these libraries. 

# Results
In July 2018, the front end was completed and was tested on KITTI dataset 00. A brief result of the front end is shown below: 
![kitti_00](https://user-images.githubusercontent.com/23701665/50575271-fdf2a200-0dfb-11e9-95d9-212ac70930ba.png)

# What to do
### 1. Loopclosing
In recent years, many machine learning methods perform very well on scenes recognition, so we would like to use machine learning techniques to realize a robust loop closure.

### 2.Optimization
In ORBSLAM, g2o is used for optimization. In google's cartographer and HKUST-Aerial-Robotics group's VINS, Ceres is used. So, We would like to choose g2o or ceres as our solver.

