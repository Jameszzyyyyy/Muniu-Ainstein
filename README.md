# Muniu-Ainstein

### _This repository contains codes about basic algorithms in analyzing and processing 3D point cloud data_  

---

## Directories
- **Directory _[pca_normal_voxel:](./pca_normal_voxel)_** The following algorithms and principle are most basic ones in processing 3D Point Cloud Data
    * **_Principal Components Analysis:_** Also Known as PCA, this is a data dimensionality 
    reduction technique, by using it we can remove unnecessary noise and reduce a lot of computation.
  
    * **_Normal Estimation:_** Normal vector is a concept of spatial analytic geometry, 
    where the vector represented by a line perpendicular to the plane is the normal vector of the plane. 
    The role of point cloud normal vectors is to describe the geometry of the surrounding environment 
    as accurately as possible, helping the robot to locate and build a map
  
    * **_Voxel Filter:_** This is a technique for Downsampling the data, reducing the amount of computations. 
    By randomly or averagely selecting specific amount of point within each small portion of the whole point cloud,
    we are able to reduce the amount of the data selected while keeping the original shape of the object.


- **Directory _[trees](./trees)_**
    * _KDTree:_
    * _OCTree:_
- **Directory _[Clustering](./Clustering)_**
    * _KMeans:_
    * _GMM:_
    * _Spectral:_

---