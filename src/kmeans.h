#ifndef KMEANS_H
#define KMEANS_H

#include <vector>

struct Point {
    int id;
    int assignedCluster;
    int numDimensions;
    std::vector<float> coordinates;
};

struct Cluster {
    int id;
    std::vector<float> centroid;
    std::vector<int> pointIds;
};

// Declaration of the CUDA wrapper function
void kmeans_cuda(Point* d_points, Cluster* d_clusters, int numPoints, int numClusters, int numDimensions, int maxIterations, float convergenceThreshold);

#endif // KMEANS_H
