# 文件功能： 实现 K-Means 算法
import numpy as np
import random
import matplotlib.pyplot as plt
from result_set import KNNResultSet, RadiusNNResultSet
import kdtree as kdtree
import sys
# 二维点云显示函数
def Point_Show(point,color):
    x = []
    y = []
    point = np.asarray(point)
    for i in range(len(point)):
        x.append(point[i][0])
        y.append(point[i][1])
    plt.scatter(x, y,color=color)
    #plt.show()

class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, data):
        self.centers_ = data[random.sample(range(data.shape[0]),self.k_)]   #随机生成k个中心
        old_centers = np.copy(self.centers_)
        #expectation:
        leaf_size = 1
        k = 1  # 结果每个点选取属于自己的类中心
        for _ in range(self.max_iter_):
            labels = [[] for i in range(self.k_)]        #用于分类所有数据点
            root = kdtree.kdtree_construction(self.centers_, leaf_size=leaf_size)    #对中心点进行构建kd-tree
            for i in range(data.shape[0]):
                result_set = KNNResultSet(capacity=k)
                query =  data[i]
                kdtree.kdtree_knn_search(root, self.centers_, result_set, query)
                output_index = result_set.knn_output_index()[0]                 #获取最邻近点的索引
                labels[output_index].append(data[i])             #将点放入类中

            #maximization：
            for i in range(self.k_):
                points = np.array(labels[i])    #第i个cluster中的点
                self.centers_[i] = points.mean(axis=0)
            if np.sum(np.abs(self.centers_ - old_centers)) < self.tolerance_ * self.k_:  # 如果前后聚类中心的距离相差小于self.tolerance_ * self.k_ 输出
                break
            old_centers = np.copy(self.centers_)     #保存旧中心点
        self.fitted = True
        return old_centers


    def predict(self, p_datas):
        result = []
        if not self.fitted:
            print('Unfitter. ')
            return result
        for point in p_datas:
            diff = np.linalg.norm(self.centers_ - point, axis=1)     #使用二范数求解每个点对新的聚类中心的距离
            result.append(np.argmin(diff))                           #返回离该点距离最小的聚类中心，标记rnk = 1
        return result

if __name__ == '__main__':
    k = int(sys.argv[2])
    x = np.genfromtxt(sys.argv[1],delimiter="").reshape((-1,2))
    #显示原始点云
    Point_Show(x,color="purple")
    k_means = K_Means(n_clusters=k)    #计算迭代后的中心点
    y = k_means.fit(x)                     #计算每个点属于哪个类
    cat = k_means.predict(x)
    #显示运行KMeans后的中心点
    Point_Show(y,color="red")
    plt.show()
    print(cat)
    #将所有点分到对应的cluster中
    cluster = [[] for i in range(k)]        #用于分类所有数据点
    for i in range(len(x)):
        for j in range(k):
            if cat[i] == j:
                cluster[j].append(x[i])
    
    #显示最终结果
    for i in range(k):
        Point_Show(cluster[i], sys.argv[3+i])  
        Point_Show(y,"red")
    plt.show()
