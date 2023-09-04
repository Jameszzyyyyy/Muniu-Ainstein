import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from pyntcloud import PyntCloud
import open3d as o3d
import sys
# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    mean = np.mean(np.asarray(data), axis=0)
    #中心化
    X_mean = data - mean
    # 计算协方差矩阵
    if correlation:
        H = np.corrcoef(X_mean.T)
    else:
        H = np.cov(X_mean.T)
    # SVD分解
    eigenvalues, eigenvectors = np.linalg.eig(H)
    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]
    return eigenvalues, eigenvectors

#function：计算法向量函数
#input：pcd_tree: open3d用flann来构建kdTree
#       points: 输入的点云数据
#       neightbors_num: 设定进行曲线拟合的临近点个数
#output：normals：把法向量信息存入一个array中
def normal_estimation(pcd_tree, points, neighbors_num):
    normals = []
    for p in points:
        #搜索领域
        [k, idx, _] = pcd_tree.search_knn_vector_3d(p, neighbors_num)
        neighbors = points[idx]
        #pca得到的向量特征
        _, eigVector = PCA(neighbors, sort=True)
        normal = eigVector[:, -1]
        normals.append(normal)
    return normals

#function：实现体素滤波
#input：point_cloud：点云数据
#       leaf_size: 体素的尺寸
#       method：random：随机采样/ mean：均值采样
#output：filtered_points 降采样之后的点云数据

def voxel_filter(point_cloud, leaf_size, method="mean"): 
    filtered_points = []
    points = np.asarray(point_cloud)
    # 计算xyz的范围
    x_min = min(points[:, 0]) 
    x_max = max(points[:, 0]) 
    y_min = min(points[:, 1]) 
    y_max = max(points[:, 1]) 
    z_min = min(points[:, 2]) 
    z_max = max(points[:, 2]) 
    # 确定xyz轴的维度
    D_x = (x_max - x_min) // leaf_size
    D_y = (y_max - y_min) // leaf_size
    D_z = (z_max - z_min) // leaf_size
    #计算每个点的索引
    ids = []
    for p in points:
        h_x = (p[0] - x_min) // leaf_size
        h_y = (p[1] - y_min) // leaf_size
        h_z = (p[2] - z_min) // leaf_size
        h = h_x + h_y * D_x + h_z * D_x * D_y
        ids.append(h)
    #对索引进行排序，记录排序后的索引变化
    sorted_ids = np.sort(ids) 
    sort_ids_ind = np.argsort(ids) 
    #遍历排序后的店，进行滤波
    local_points = []                   #存放索引相同的点，用于下采样
    previous_id = sorted_ids[0]         #存放上一个索引的值
    for i in range(len(sorted_ids)):
        # 如果当前索引等于上一个索引值，则将其加入local_points中
        if sorted_ids[i] == previous_id:
            local_points.append(points[sort_ids_ind[i]])
        #当前索引是第一次出现
        else:
            #计算上一个采样点，将其加入最终的输出数组
            if method == "mean":
                new_point = np.mean(local_points, axis=0)
                filtered_points.append(new_point)
            elif method == "random":
                np.random.shuffle(local_points)
                filtered_points.append(local_points[0]) 
                #更新previous_id，清空local points，并将当前点加入
            previous_id = sorted_ids[i]
            local_points = [points[sort_ids_ind[i]]]
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

def main():
    #提取数据
    point_cloud_raw = np.genfromtxt(sys.argv[1], delimiter=",")
    point_cloud_raw = DataFrame(point_cloud_raw[:, 0:3]) #只提取xyz三维数据
    point_cloud_raw.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
    point_cloud_pynt = PyntCloud(point_cloud_raw)  # 将points的数据 存到结构体中

    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)  # 实例化

    o3d.visualization.draw_geometries([point_cloud_o3d], window_name="raw") # 显示原始点云

    #运行pca降维
    w, v = PCA(point_cloud_raw) 
    point_cloud_vector = v[:,0:2] 
    point_cloud_encode = (np.dot(point_cloud_vector.T,point_cloud_raw.T)).T   #主成分的转置 dot 原数据
    x = []
    y = []
    pca_point_cloud = np.asarray(point_cloud_encode)
    for i in range(10000):
        x.append(pca_point_cloud[i][0])
        y.append(pca_point_cloud[i][1])
    plt.scatter(x, y)
    plt.title('pca')
    plt.show()

    #使用主方向进行升维
    point_cloud_decode = (np.dot(point_cloud_vector,point_cloud_encode.T)).T
    # matplotlib显示点云函数
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud_decode[:, 0], point_cloud_decode[:, 1], point_cloud_decode[:, 2], cmap='spectral', s=2, linewidths=0, alpha=1, marker=".")
    plt.title('Point Cloud After Decode')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


    #法向量
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    normals = normal_estimation(pcd_tree, np.asarray(point_cloud_o3d.points), int(sys.argv[2]))
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([point_cloud_o3d], window_name="normal_estimation")

    #体素滤波
    filtered = voxel_filter(point_cloud_o3d.points,float(sys.argv[4]),sys.argv[3])
    point_cloud_o3d_filter = o3d.geometry.PointCloud()
    point_cloud_o3d_filter.points = o3d.utility.Vector3dVector(filtered)
    o3d.visualization.draw_geometries([point_cloud_o3d_filter], window_name="voxel_filter (" + sys.argv[3] + ')')

if __name__ == '__main__':
    main()