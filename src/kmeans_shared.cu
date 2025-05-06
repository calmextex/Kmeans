#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <cfloat>
#include <cuda_runtime.h>
#include <cuda.h>
#include <chrono>

using namespace std;

struct Point {
    int id;               // Point ID
    int assignedCluster;   // Cluster ID
    int numDimensions;     // Number of dimensions
    float* coordinates;    // Raw pointer for values of each dimension
};

struct Cluster {
    int id;                // Cluster ID
    float* centroid;       // Raw pointer for centroid of the cluster
};

static unsigned long int k_next = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand() {
    k_next = k_next * 1103515245 + 12345;
    return (unsigned int)(k_next / 65536) % (kmeans_rmax + 1);
}

void kmeans_srand(unsigned int seed) {
    k_next = seed;
}

void initializePoint(Point& point, int id, const string& line, int numDimensions) {
    point.id = id;
    point.assignedCluster = -1;  // Not assigned to any cluster yet
    point.coordinates = new float[numDimensions];  // Dynamically allocate array

    stringstream ss(line);
    int skipInt;
    float value;

    ss >> skipInt;  // Skip the first integer value

    for (int d = 0; d < numDimensions; d++) {
        if (ss >> value) {
            point.coordinates[d] = value;
        }
    }

    point.numDimensions = numDimensions;
}

void initializeClusters(vector<Cluster>& clusters, const vector<Point>& points, int numClusters, int numDimensions) {
    for (int i = 0; i < numClusters; i++) {
        Cluster cluster;
        cluster.centroid = new float[numDimensions];  // Dynamically allocate array for centroid

        int idx = kmeans_rand() % points.size();  

        for (int d = 0; d < numDimensions; d++) {
            cluster.centroid[d] = points[idx].coordinates[d];  // Assign the centroid from a point
        }
        clusters.push_back(cluster);
    }
}

// Cuda kernel using shared mem to assign each point to nearest cluster.
__global__ void assignPointsToClusters(Point* points, Cluster* clusters, int numPoints, int numClusters, int numDimensions) {
    extern __shared__ float sharedCentroids[];  // Dynamically allocated shared memory for centroids

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load cluster centroids into shared memory for all threads
    for (int c = threadIdx.x; c < numClusters * numDimensions; c += blockDim.x) {
        sharedCentroids[c] = clusters[0].centroid[c];
    }

    // Synchronize to ensure all centroids are loaded
    __syncthreads();

    if (idx < numPoints) {
        float minDistance = FLT_MAX;
        int nearestCluster = -1;

        // Calculate the distance to each centroid
        for (int c = 0; c < numClusters; c++) {
            float dist = 0.0f;
            for (int d = 0; d < numDimensions; d++) {
                float diff = points[idx].coordinates[d] - sharedCentroids[c * numDimensions + d];
                dist += diff * diff;
            }

            if (dist < minDistance) {
                minDistance = dist;
                nearestCluster = c;
            }
        }
        points[idx].assignedCluster = nearestCluster;
    }
}

// Cuda Kernel using shared memory to yodate the centroids of each cluster
__global__ void updateCentroids(Point* points, Cluster* clusters, int numPoints, int numClusters, int numDimensions) {
    extern __shared__ float sharedCentroidSum[];  // Dynamically allocated shared memory for sums

    int clusterIdx = blockIdx.x;  // One block per cluster
    int pointIdx = threadIdx.x;

    // Initialize shared memory for centroids to zero
    for (int d = threadIdx.x; d < numDimensions; d += blockDim.x) {
        sharedCentroidSum[d] = 0.0f;
    }
    __syncthreads();

    // Perform parallel reduction to sum points assigned to this cluster
    for (int i = pointIdx; i < numPoints; i += blockDim.x) {
        if (points[i].assignedCluster == clusterIdx) {
            for (int d = 0; d < numDimensions; d++) {
                sharedCentroidSum[d] += points[i].coordinates[d];
            }
        }
    }
    __syncthreads();

    // Reduction step: sum within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            for (int d = 0; d < numDimensions; d++) {
                sharedCentroidSum[d] += sharedCentroidSum[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    // Normalize and update the centroids in global memory
    if (threadIdx.x == 0) {
        int numAssignedPoints = 0;
        for (int i = 0; i < numPoints; i++) {
            if (points[i].assignedCluster == clusterIdx) {
                numAssignedPoints++;
            }
        }

        if (numAssignedPoints > 0) {
            for (int d = 0; d < numDimensions; d++) {
                clusters[clusterIdx].centroid[d] = sharedCentroidSum[d] / numAssignedPoints;
            }
        }
    }
}


// kmeans function for shared mem implementation
void kmeans_cuda(Point* h_points, Cluster* h_clusters, int numPoints, int numClusters, int numDimensions, int maxIterations, float convergenceThreshold) {
    Point* d_points;
    Cluster* d_clusters;

    // Allocate device memory for Points and Clusters
    cudaMalloc(&d_points, numPoints * sizeof(Point));
    cudaMalloc(&d_clusters, numClusters * sizeof(Cluster));

    // Create CUDA events for timing
    cudaEvent_t startTransferToGPU, endTransferToGPU, startKernel, endKernel, startTransferFromGPU, endTransferFromGPU;
    cudaEventCreate(&startTransferToGPU);
    cudaEventCreate(&endTransferToGPU);
    cudaEventCreate(&startKernel);
    cudaEventCreate(&endKernel);
    cudaEventCreate(&startTransferFromGPU);
    cudaEventCreate(&endTransferFromGPU);

    // Record the start time for transfer to GPU
    cudaEventRecord(startTransferToGPU);

    // Copy points and clusters from host (CPU) to device (GPU)
    cudaMemcpy(d_points, h_points, numPoints * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_clusters, h_clusters, numClusters * sizeof(Cluster), cudaMemcpyHostToDevice);

    // Record the end time for transfer to GPU
    cudaEventRecord(endTransferToGPU);
    cudaEventSynchronize(endTransferToGPU);

    float timeTransferToGPU;
    cudaEventElapsedTime(&timeTransferToGPU, startTransferToGPU, endTransferToGPU);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numPoints + threadsPerBlock - 1) / threadsPerBlock;

    // Record the start time for the kernel execution
    cudaEventRecord(startKernel);

    // Main loop: run maxIterations times
    for (int iter = 0; iter < maxIterations; iter++) {
        assignPointsToClusters<<<blocksPerGrid, threadsPerBlock>>>(d_points, d_clusters, numPoints, numClusters, numDimensions);
        updateCentroids<<<numClusters, threadsPerBlock>>>(d_points, d_clusters, numPoints, numClusters, numDimensions);
        cudaDeviceSynchronize();
    }

    // Record the end time for the kernel execution
    cudaEventRecord(endKernel);
    cudaEventSynchronize(endKernel);

    float timeKernel;
    cudaEventElapsedTime(&timeKernel, startKernel, endKernel);

    // Record the start time for transfer from GPU
    cudaEventRecord(startTransferFromGPU);

    // Copy points and clusters back from device (GPU) to host (CPU)
    cudaMemcpy(h_points, d_points, numPoints * sizeof(Point), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_clusters, d_clusters, numClusters * sizeof(Cluster), cudaMemcpyDeviceToHost);

    // Record the end time for transfer from GPU
    cudaEventRecord(endTransferFromGPU);
    cudaEventSynchronize(endTransferFromGPU);

    float timeTransferFromGPU;
    cudaEventElapsedTime(&timeTransferFromGPU, startTransferFromGPU, endTransferFromGPU);

    // Free device memory
    cudaFree(d_points);
    cudaFree(d_clusters);

    // Print out the times
    cout << "Time spent transferring data to GPU: " << timeTransferToGPU << " ms" << endl;
    cout << "Time spent executing kernels: " << timeKernel << " ms" << endl;
    cout << "Time spent transferring data from GPU: " << timeTransferFromGPU << " ms" << endl;

    // Cleanup CUDA events
    cudaEventDestroy(startTransferToGPU);
    cudaEventDestroy(endTransferToGPU);
    cudaEventDestroy(startKernel);
    cudaEventDestroy(endKernel);
    cudaEventDestroy(startTransferFromGPU);
    cudaEventDestroy(endTransferFromGPU);
}



// Main function
int main(int argc, char* argv[]) {
    int numClusters = 0, numDimensions = 0, maxIterations = 150, seed = 0;
    string inputFilename;
    float convergenceThreshold = 1e-5;
    bool printCentroids = false;

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-k") == 0) numClusters = atoi(argv[++i]);
        else if (strcmp(argv[i], "-d") == 0) numDimensions = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0) inputFilename = argv[++i];
        else if (strcmp(argv[i], "-m") == 0) maxIterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0) convergenceThreshold = atof(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0) seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "-c") == 0) printCentroids = true;
    }

    vector<Point> points;
    ifstream infile(inputFilename);
    string line;

    // Read the total number of points
    int totalPoints;
    if (getline(infile, line)) {
        totalPoints = stoi(line);
    } else {
        cerr << "Error: Unable to read the number of points." << endl;
        return 1;
    }

    // Read each point
    int pointId = 0;
    while (getline(infile, line) && pointId < totalPoints) {
        Point point;
        initializePoint(point, pointId, line, numDimensions);
        points.push_back(point);
        pointId++;
    }
    infile.close();

    vector<Cluster> clusters;
    kmeans_srand(seed);
    initializeClusters(clusters, points, numClusters, numDimensions);
    
    // Run the CUDA implementation
    auto start = chrono::high_resolution_clock::now();
    kmeans_cuda(points.data(), clusters.data(), totalPoints, numClusters, numDimensions, maxIterations, convergenceThreshold);
    auto end = chrono::high_resolution_clock::now();

    double totalTime = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    double timePerIteration = totalTime / maxIterations;


    cout << maxIterations << "," << fixed << setprecision(6) << timePerIteration << endl;
    if (printCentroids) {
        for (int clusterId = 0; clusterId < numClusters; clusterId++) {
            cout << clusterId << " ";
            for (int d = 0; d < numDimensions; d++) {
                cout << fixed << setprecision(5) << clusters[clusterId].centroid[d] << " ";
            }
            cout << endl;
        }
    } else {
        cout << "clusters:";
        for (const auto& point : points) {
            cout << " " << point.assignedCluster;
        }
        cout << endl;
    }

    for (auto& point : points) {
        delete[] point.coordinates;
    }
    for (auto& cluster : clusters) {
        delete[] cluster.centroid;
    }

    return 0;
}
