import numpy as np

# 计算欧几里得距离
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# k-means算法
def kmeans(points, mean_points, k):
    points = np.array(points)
    centroids = np.array(mean_points)
    while(True):
        # 分配每个点到最近的质心
        clusters = [[] for _ in range(k)]
        for point in points:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            closest_centroid = np.argmin(distances)
            clusters[closest_centroid].append(point)
        
        # 记录旧质心
        old_centroids = centroids.copy()
        
        # 重新计算每个簇的质心
        for i in range(k):
            if clusters[i]:  # 防止空簇
                centroids[i] = np.mean(clusters[i], axis=0)
        
        # 检查质心是否发生变化，如果没有变化则停止
        if np.all(old_centroids == centroids):
            break

    return centroids, clusters

points=[[15., 10.],[3., 10.],[15., 12.],[3., 14.],[18., 13.],[1., 7.],[10., 1.],[10., 30.]]
mean_points=[[10.,1.],[10.,30.],[3.,10.],[15.,10.]]
k = 4

# 运行k-means算法
centroids, clusters = kmeans(points, mean_points, k)
print("最终质心: ", centroids)
print("每个簇中的点: ", clusters)