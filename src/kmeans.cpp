#include <iostream>   
#include <vector>     
#include <cmath>      
#include <limits>     
#include <sstream>    
#include <fstream>    
#include <algorithm>  
#include <chrono>     
#include <cstring> 
#include <cfloat> 
#include <unordered_set>  
#include <iomanip>

using namespace std;

static unsigned long int k_next = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand() {
    k_next = k_next * 1103515245 + 12345;
    return (unsigned int)(k_next / 65536) % (kmeans_rmax + 1);
}

void kmeans_srand(unsigned int seed) {
    k_next = seed;
}


struct Point {
    int id;  // Point ID
    int assignedCluster;  // Cluster ID
    int numDimensions;  // Number of dimensions
    vector<float> coordinates;  // Values of each dimension
};

struct Cluster {
    int id;  // Cluster ID
    vector<float> centroid;  // Centroid of the cluster
    vector<int> pointIds;  // IDs of points assigned to this cluster
};

// Function converts the input into point objects with float coordinates.
void initializePoint(Point &point, int id, const string &line) {
    point.id = id;
    point.assignedCluster = -1;  // Not assigned to any cluster yet
    point.coordinates.clear();

    stringstream ss(line);
    int skipInt;
    float value;

    // skipping the first integer that is read.
    ss >> skipInt; 

    while (ss >> value) {
        point.coordinates.push_back(value);
    }

    point.numDimensions = point.coordinates.size();
}

// Initialialization of each cluster centroid. 

void initializeClusters(vector<Cluster>& clusters, const vector<Point>& points, int numClusters, int seed) {
    kmeans_srand(seed);
    
    for (int i = 0; i < numClusters; i++) {
        int idx = kmeans_rand() % points.size();
        
        Cluster cluster;
        cluster.id = i;
        cluster.centroid = points[idx].coordinates;
        clusters.push_back(cluster);
    }
}

// Calculating the Euclidean distanve between points and cluster centroids
int findNearestCluster(const vector<Cluster> &clusters, const Point &point) {
    double minDistance = DBL_MAX;
    int nearestClusterId = -1;

    for (const auto &cluster : clusters) {
        double distance = 0.0;
        for (size_t d = 0; d < point.coordinates.size(); d++) {
            double diff = cluster.centroid[d] - point.coordinates[d];
            distance += diff * diff;
        }

        if (distance < minDistance) {
            minDistance = distance;
            nearestClusterId = cluster.id;
        }
    }

    return nearestClusterId;
}

// Add points back to the clusters
void addPointToCluster(Cluster &cluster, int pointId) {
    cluster.pointIds.push_back(pointId);
}

// remove points from the cluster
bool removePointFromCluster(Cluster &cluster, int pointId) {
    auto it = find(cluster.pointIds.begin(), cluster.pointIds.end(), pointId);
    if (it != cluster.pointIds.end()) {
        cluster.pointIds.erase(it);
        return true;
    }
    return false;
}

// Updates the centroid of clusters. Averages coordinates of points assigned to each cluster.
void updateCentroids(const vector<Point>& points, vector<Cluster>& clusters) {
    for (auto& cluster : clusters) {
        vector<double> sum(cluster.centroid.size(), 0.0);
        int count = 0;

        for (const auto& point : points) {
            if (point.assignedCluster == cluster.id) {
                for (size_t d = 0; d < point.coordinates.size(); d++) {
                    sum[d] += point.coordinates[d];
                }
                count++;
            }
        }

        if (count > 0) {
            for (size_t d = 0; d < sum.size(); d++) {
                cluster.centroid[d] = sum[d] / count;
            }
        }
    }
}
// check for convergence
bool hasConverged(const vector<Cluster>& oldClusters, const vector<Cluster>& newClusters, float threshold) {
    for (size_t i = 0; i < oldClusters.size(); ++i) {
        float distance = 0.0;
        for (size_t j = 0; j < oldClusters[i].centroid.size(); ++j) {
            float diff = oldClusters[i].centroid[j] - newClusters[i].centroid[j];
            distance += diff * diff;
        }
        if (sqrt(distance) > threshold) {
            return false;
        }
    }
    return true;
}

// Kmeans process
int kmeans(vector<Point>& points, vector<Cluster>& clusters, int maxIterations, float convergenceThreshold) {
    int iteration = 0;
    vector<Cluster> oldClusters;

    while (iteration < maxIterations) {
        oldClusters = clusters;

        // Assign points to nearest cluster
        for (auto& point : points) {
            point.assignedCluster = findNearestCluster(clusters, point);
        }

        // Update centroids
        updateCentroids(points, clusters);

        // Check for convergence
        if (hasConverged(oldClusters, clusters, convergenceThreshold)) {
            break;
        }

        iteration++;
    }

    return iteration;
}

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

    // Read the points
    int pointId = 0;
    while (getline(infile, line) && pointId < totalPoints) {
        Point point;
        initializePoint(point, pointId, line);
        points.push_back(point);
        pointId++;
    }
    infile.close();

    if (pointId != totalPoints) {
        cerr << "Warning: Expected " << totalPoints << " points, but read " << pointId << " points." << endl;
    }
    

    // Initialize clusters
    vector<Cluster> clusters;
    kmeans_srand(seed);
    initializeClusters(clusters, points, numClusters, seed);

    //cout << "Initial Centroids: " << endl;
    //for (int i = 0; i < numClusters; i++) {
        //cout << "Cluster " << i << ": ";
        //for (const auto& coord : clusters[i].centroid) {
            //cout << coord << " ";
        //}
        //cout << endl;
    //}

    auto start = chrono::high_resolution_clock::now();
    int iterations = kmeans(points, clusters, maxIterations, convergenceThreshold);
    auto end = chrono::high_resolution_clock::now();

    double totalTime = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    double timePerIteration = totalTime / iterations;

    // Output iterations and time per iteration
    cout << iterations << "," << fixed << setprecision(6) << timePerIteration << endl;

    // Output centroids or point assignments based on -c flag
    if (printCentroids) {
        for (int clusterId = 0; clusterId < numClusters; clusterId++) {
            cout << clusterId << " ";
            for (const auto& coord : clusters[clusterId].centroid) {
                cout << fixed << setprecision(5) << coord << " ";
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

    return 0;
}