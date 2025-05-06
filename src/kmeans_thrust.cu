#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <sstream>

static unsigned long int k_next = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand() {
    k_next = k_next * 1103515245 + 12345;
    return (unsigned int)(k_next / 65536) % (kmeans_rmax + 1);
}

void kmeans_srand(unsigned int seed) {
    k_next = seed;
}

__device__ unsigned int device_kmeans_rand(unsigned int* state) {
    *state = (*state * 1103515245 + 12345) & 0x7fffffff;
    return (*state);
}

struct IndexTransform {
    int numDimensions;
    int d;

    // Ensures this functor can be called both from the host and device
    __host__ __device__
    IndexTransform(int numDimensions, int d) : numDimensions(numDimensions), d(d) {}

    // The operator() should now be callable from both host and device
    __host__ __device__
    int operator()(int idx) const {
        return idx * numDimensions + d;
    }
};



struct KMeansCentroidInitFunctor {
    const float* d_points;
    int numPoints;
    int numDimensions;
    unsigned int seed;

    KMeansCentroidInitFunctor(const float* points, int nPoints, int nDim, unsigned int s)
        : d_points(points), numPoints(nPoints), numDimensions(nDim), seed(s) {}

    __device__
    float operator()(const thrust::tuple<int, int>& t) {
        int centroidIndex = thrust::get<0>(t);
        int dimensionIndex = thrust::get<1>(t);
        
        // Use a simple hash function to generate a unique seed for each thread
        unsigned int state = seed + centroidIndex + dimensionIndex;
        
        // Generate a random index for each centroid
        int randomIndex = device_kmeans_rand(&state) % numPoints;
        
        // Return the value of the randomly selected point for this dimension
        return d_points[randomIndex * numDimensions + dimensionIndex];
    }
};

struct NearestCentroidFunctor {
    const float* d_points;
    const float* d_centroids;
    int numClusters;
    int numDimensions;

    NearestCentroidFunctor(const float* points, const float* centroids, int k, int dim)
        : d_points(points), d_centroids(centroids), numClusters(k), numDimensions(dim) {}

    __device__ int operator()(const int pointIndex) const {
        float minDistance = FLT_MAX;
        int nearestClusterId = -1;

        for (int clusterId = 0; clusterId < numClusters; clusterId++) {
            float distance = 0.0;
            for (int d = 0; d < numDimensions; d++) {
                float diff = d_points[pointIndex * numDimensions + d] - d_centroids[clusterId * numDimensions + d];
                distance += diff * diff;
            }

            if (distance < minDistance) {
                minDistance = distance;
                nearestClusterId = clusterId;
            }
        }

        return nearestClusterId;
    }
};

struct AddPointsToCentroidFunctor {
    const float* d_points;
    float* d_centroids;
    const int* d_labels;
    int numDimensions;

    AddPointsToCentroidFunctor(const float* points, float* centroids, const int* labels, int dim)
        : d_points(points), d_centroids(centroids), d_labels(labels), numDimensions(dim) {}

    __device__ void operator()(const int pointIndex) {
        int clusterId = d_labels[pointIndex];

        for (int d = 0; d < numDimensions; d++) {
            atomicAdd(&d_centroids[clusterId * numDimensions + d], d_points[pointIndex * numDimensions + d]);
        }
    }
};

struct IncrementClusterSize {
    int* d_cluster_sizes;

    IncrementClusterSize(int* cluster_sizes) : d_cluster_sizes(cluster_sizes) {}

    __device__ void operator()(const int label) {
        atomicAdd(&d_cluster_sizes[label], 1);
    }
};

struct NormalizeCentroid {
    float* d_centroids;
    const int* d_cluster_sizes;
    int numDimensions;

    NormalizeCentroid(float* centroids, const int* cluster_sizes, int dim)
        : d_centroids(centroids), d_cluster_sizes(cluster_sizes), numDimensions(dim) {}

    __device__ void operator()(const int index) {
        int cluster = index / numDimensions;
        int size = d_cluster_sizes[cluster];
        if (size > 0) {
            d_centroids[index] /= size;
        }
    }
};

struct MaxCentroidChange {
    __device__ float operator()(const thrust::tuple<float, float>& t) const {
        return fabs(thrust::get<0>(t) - thrust::get<1>(t));
    }
};


void kmeans_thrust(thrust::device_vector<float>& d_points, thrust::device_vector<float>& d_centroids,
                   thrust::device_vector<int>& d_labels, int numPoints, int numClusters, int numDimensions,
                   int maxIterations, float convergenceThreshold) {
    int iteration = 0;
    bool converged = false;

    thrust::device_vector<float> d_old_centroids = d_centroids;
    thrust::device_vector<int> d_cluster_sizes(numClusters);

    thrust::device_vector<int> d_point_indices(numPoints);
    thrust::sequence(d_point_indices.begin(), d_point_indices.end()); // This gives us [0, 1, 2, ..., numPoints-1]

    while (iteration < maxIterations && !converged) {
        iteration++;

        // Assign points to the nearest centroids
        thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(numPoints),
                          d_labels.begin(), NearestCentroidFunctor(thrust::raw_pointer_cast(d_points.data()),
                                                                  thrust::raw_pointer_cast(d_centroids.data()),
                                                                  numClusters, numDimensions));

        // Sort point indices by their labels (centroid assignments)
        thrust::stable_sort_by_key(d_labels.begin(), d_labels.end(), d_point_indices.begin());

        // Prepare temporary storage for aggregation
        thrust::device_vector<float> d_summed_centroids(numClusters * numDimensions, 0.0f);
        thrust::device_vector<int> d_cluster_sizes(numClusters, 0);

        // Reduce by key to sum points for each centroid
        for (int d = 0; d < numDimensions; d++) {
            thrust::reduce_by_key(
                d_labels.begin(), d_labels.end(),
                thrust::make_permutation_iterator(
                    d_points.begin() + d, thrust::make_transform_iterator(d_point_indices.begin(), IndexTransform(numDimensions, d))),
                thrust::make_discard_iterator(), // we discard the reduced labels
                d_summed_centroids.begin() + d, // output summed centroids
                thrust::equal_to<int>(), thrust::plus<float>());
        }


        // Calculate new centroids by dividing summed values by the number of points per cluster
        for (int d = 0; d < numDimensions; d++) {
            thrust::for_each(
                thrust::make_counting_iterator(0), thrust::make_counting_iterator(numClusters),
                NormalizeCentroid(thrust::raw_pointer_cast(d_summed_centroids.data()), 
                                  thrust::raw_pointer_cast(d_cluster_sizes.data()), numDimensions));
        }

        // Check for convergence
        float maxChange = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(d_old_centroids.begin(), d_centroids.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(d_old_centroids.end(), d_centroids.end())),
            MaxCentroidChange(), 0.0f, thrust::maximum<float>());

        converged = maxChange < convergenceThreshold;

        thrust::copy(d_centroids.begin(), d_centroids.end(), d_old_centroids.begin());
    }
}

int main(int argc, char* argv[]) {
    int numClusters = 0, numDimensions = 0, maxIterations = 150, seed = 0;
    float convergenceThreshold = 1e-5;
    std::string inputFilename;
    bool printCentroids = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-k") == 0) numClusters = atoi(argv[++i]);
        else if (strcmp(argv[i], "-d") == 0) numDimensions = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0) inputFilename = argv[++i];
        else if (strcmp(argv[i], "-m") == 0) maxIterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0) convergenceThreshold = atof(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0) seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "-c") == 0) printCentroids = true;
    }

    std::ifstream infile(inputFilename);
    if (!infile) {
        std::cerr << "Error: Could not open file " << inputFilename << std::endl;
        return 1;
    }

    std::vector<float> h_points;
    std::string line;
    int numPoints = 0;

    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        float value;
        while (ss >> value) {
            h_points.push_back(value);
        }
        numPoints++;
    }
    infile.close();

    thrust::device_vector<float> d_points = h_points;
    thrust::device_vector<float> d_centroids(numClusters * numDimensions);
    thrust::device_vector<int> d_labels(numPoints);

    kmeans_srand(seed);
    thrust::counting_iterator<int> centroidIter(0);
    thrust::counting_iterator<int> dimensionIter(0);
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(
            thrust::make_counting_iterator(0),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                thrust::placeholders::_1 % numDimensions)
        )),
        thrust::make_zip_iterator(thrust::make_tuple(
            thrust::make_counting_iterator(0),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                thrust::placeholders::_1 % numDimensions)
        )) + (numClusters * numDimensions),
        d_centroids.begin(),
        KMeansCentroidInitFunctor(thrust::raw_pointer_cast(d_points.data()), numPoints, numDimensions, seed)
    );

    cudaEvent_t start, stop;
    float delta_ms = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kmeans_thrust(d_points, d_centroids, d_labels, numPoints, numClusters, numDimensions, maxIterations, convergenceThreshold);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&delta_ms, start, stop);

    std::cout << "Time taken: " << delta_ms << " ms" << std::endl;

    if (printCentroids) {
        thrust::host_vector<float> h_centroids = d_centroids;
        for (int i = 0; i < numClusters; i++) {
            std::cout << i << ": ";
            for (int d = 0; d < numDimensions; d++) {
                std::cout << h_centroids[i * numDimensions + d] << " ";
            }
            std::cout << std::endl;
        }
    } else {
        thrust::host_vector<int> h_labels = d_labels;
        std::cout << "Cluster labels for points: ";
        for (int label : h_labels) {
            std::cout << label << " ";
        }
        std::cout << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}