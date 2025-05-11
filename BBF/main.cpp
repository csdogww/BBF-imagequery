#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <queue>   // For priority_queue in BBF
#include <limits>  // For std::numeric_limits
#include <chrono>  // For timing
#include <fstream> // For file reading
#include <iomanip> // For output formatting
// For memory usage (platform-dependent, this is a common approach for Linux/macOS)
#ifdef __linux__
#include <sys/resource.h>
#elif __APPLE__
#include <sys/resource.h>
#include <mach/mach.h>
#endif

// --- 1. Point/Vector Representation ---
struct Point
{
    std::vector<long long> coordinates; // d-dimensional coordinates
    int id;                             // Optional: original index or ID

    Point(int dim = 0) : coordinates(dim) {}

    long long &operator[](int index)
    {
        return coordinates[index];
    }
    const long long &operator[](int index) const
    {
        return coordinates[index];
    }
    int dim() const
    {
        return coordinates.size();
    }
};

// --- 2. k-d Tree Node ---
struct KdNode
{
    Point point; // Point stored at this node (for leaf nodes, or median point for internal nodes)
                 // For a more typical k-d tree, internal nodes just store split_dim and split_val
                 // and points are only in leaves, or all nodes store a point.
                 // Let's try a version where internal nodes store the splitting point.
    KdNode *left = nullptr;
    KdNode *right = nullptr;
    int split_dimension = -1; // Dimension used for splitting at this node

    // Constructor for internal node (stores splitting point)
    KdNode(const Point &p, int dim) : point(p), split_dimension(dim) {}
    // Constructor for leaf node (can also store a point, or a list of points if not splitting to single points)
    // For simplicity, if a node has no children, it's effectively a leaf containing its 'point'.
};

// Global variable for dimensionality, loaded from file
int K_DIM = 0;

// --- 4. Distance Function ---
double euclidean_distance_sq(const Point &p1, const Point &p2)
{
    if (p1.dim() != K_DIM || p2.dim() != K_DIM)
    {
        throw std::runtime_error("Points have different or incorrect dimensions for distance calculation.");
    }
    double dist_sq = 0;
    for (int i = 0; i < K_DIM; ++i)
    {
        double diff = static_cast<double>(p1[i]) - p2[i];
        dist_sq += diff * diff;
    }
    return dist_sq; // Return squared distance to avoid sqrt until the very end
}

double euclidean_distance(const Point &p1, const Point &p2)
{
    return std::sqrt(euclidean_distance_sq(p1, p2));
}

// --- 3. k-d Tree Construction ---
// Helper for sorting points along a specific dimension
struct CompareDim
{
    int dim;
    CompareDim(int d) : dim(d) {}
    bool operator()(const Point &a, const Point &b) const
    {
        return a[dim] < b[dim];
    }
};

KdNode *build_kdtree_recursive(std::vector<Point> &points, int depth)
{
    if (points.empty())
    {
        return nullptr;
    }

    int axis = depth % K_DIM; // Cycle through dimensions

    // Sort points along the current axis and find median
    std::sort(points.begin(), points.end(), CompareDim(axis));
    int median_idx = points.size() / 2;
    Point median_point = points[median_idx];

    KdNode *node = new KdNode(median_point, axis);

    std::vector<Point> left_points(points.begin(), points.begin() + median_idx);
    std::vector<Point> right_points(points.begin() + median_idx + 1, points.end());

    node->left = build_kdtree_recursive(left_points, depth + 1);
    node->right = build_kdtree_recursive(right_points, depth + 1);

    return node;
}

KdNode *build_kdtree(std::vector<Point> &points)
{
    if (points.empty() || points[0].dim() == 0)
    {
        K_DIM = 0;
        return nullptr;
    }
    K_DIM = points[0].dim(); // Set global K_DIM
    return build_kdtree_recursive(points, 0);
}

void destroy_kdtree(KdNode *node)
{
    if (!node)
        return;
    destroy_kdtree(node->left);
    destroy_kdtree(node->right);
    delete node;
}

// --- 5. Brute-force Search ---
std::pair<Point, double> brute_force_nn(const std::vector<Point> &points, const Point &query)
{
    if (points.empty())
    {
        throw std::runtime_error("Point list is empty for brute-force search.");
    }
    Point best_point = points[0];
    double min_dist_sq = euclidean_distance_sq(query, points[0]);

    for (size_t i = 1; i < points.size(); ++i)
    {
        double dist_sq = euclidean_distance_sq(query, points[i]);
        if (dist_sq < min_dist_sq)
        {
            min_dist_sq = dist_sq;
            best_point = points[i];
        }
    }
    return {best_point, std::sqrt(min_dist_sq)};
}

// --- 6. Standard k-d Tree Search (Exact NN) ---
struct BestNeighbor
{
    Point point;
    double dist_sq = std::numeric_limits<double>::infinity();
};

void kdtree_nn_search_recursive(KdNode *current_node, const Point &query, BestNeighbor &best)
{
    if (!current_node)
    {
        return;
    }

    // 1. Calculate distance from current node's point to query
    double d_sq = euclidean_distance_sq(query, current_node->point);
    if (d_sq < best.dist_sq)
    {
        best.dist_sq = d_sq;
        best.point = current_node->point;
    }

    // 2. Decide which subtree to visit first
    int axis = current_node->split_dimension;
    KdNode *nearer_subtree = nullptr;
    KdNode *farther_subtree = nullptr;

    if (query[axis] < current_node->point[axis])
    {
        nearer_subtree = current_node->left;
        farther_subtree = current_node->right;
    }
    else
    {
        nearer_subtree = current_node->right;
        farther_subtree = current_node->left;
    }

    // 3. Recursively search the nearer subtree
    kdtree_nn_search_recursive(nearer_subtree, query, best);

    // 4. Check if the farther subtree needs to be searched
    // (if the hypersphere violência the splitting hyperplane)
    double dist_to_plane_sq = static_cast<double>(query[axis]) - current_node->point[axis];
    dist_to_plane_sq *= dist_to_plane_sq;

    if (dist_to_plane_sq < best.dist_sq)
    {
        kdtree_nn_search_recursive(farther_subtree, query, best);
    }
}

std::pair<Point, double> kdtree_nn_search(KdNode *root, const Point &query)
{
    if (!root)
    {
        throw std::runtime_error("KD-Tree is empty or not built.");
    }
    BestNeighbor best;
    best.point = root->point; // Initialize with root point
    best.dist_sq = euclidean_distance_sq(query, root->point);

    kdtree_nn_search_recursive(root, query, best);
    return {best.point, std::sqrt(best.dist_sq)};
}

// --- 7. BBF Search ---
// For BBF, priority queue stores pairs: (negative_distance_to_region_boundary, node_ptr)
// Or, for non-leaf nodes, it's distance to the point. For leaves, it's 0.
// The pseudo code suggests priority = 1/(distance_to_split), which means smaller distance = higher priority.
// std::priority_queue is a max-heap.
// We need to store distance from query to the *bounding box* of the subtree for more accurate BBF.
// The provided pseudocode is a bit simplified on priority; a common way is to prioritize nodes whose bounding boxes are closer to the query.
// Let's try to follow the spirit of the provided pseudocode.
// "distance_to_split" can be interpreted as distance from query point to the splitting hyperplane of the *parent* of the node being inserted.
// A more robust BBF uses distances to bounding boxes of child nodes.

// Simpler BBF Node for priority queue: {priority (e.g., negative distance), Node pointer}
// The pseudocode uses "priority=1/(distance_to_split)" implies smaller distance = higher priority
// Let's use negative distance or 1/distance for max-heap. Or use std::pair<double, KdNode*> and make distance negative.
// We need a slightly different definition for priority: distance from query to node's bounding box.
// For simplicity, let's follow the provided pseudo-code closely first, where priority is related to distance to the split plane.
// When descending, if we go to 'nearer_subtree', the priority to explore 'farther_subtree' later would be based on
// abs(query[split_dim] - node.point[split_dim]). A smaller such distance means higher priority.

// Priority for a node: how "promising" it is.
// A leaf node is promising if it's close.
// An internal node is promising if its region is close.
// The pseudocode seems to prioritize subtrees based on the distance to the splitting plane.

struct BBFQueueEntry
{
    double priority; // Higher value = higher priority.
                     // We can use negative distance, or 1/distance.
                     // Let's use negative distance so smaller distance means higher priority for max-heap.
    KdNode *node;
    bool is_leaf_path; // True if this entry represents path to a leaf, false if internal node to expand

    // For std::priority_queue (max-heap)
    bool operator<(const BBFQueueEntry &other) const
    {
        return priority < other.priority; // Smaller priority value means "less" for max-heap
    }
};

std::pair<Point, double> bbf_search(KdNode *root, const Point &query, int t_max_leaf_nodes)
{
    if (!root)
    {
        throw std::runtime_error("KD-Tree is empty for BBF search.");
    }

    std::priority_queue<BBFQueueEntry> pq;
    BestNeighbor best_bbf; // Stores current best point and its squared distance

    // Initial state: push root with highest priority (or 0 distance if interpreting priority differently)
    // The pseudocode pushes root with priority 0. Let's assume this means direct exploration.
    // Distance to split for root's children isn't well-defined this way.
    // Let's adapt: We push nodes to explore. If it's an internal node, we expand it.
    // If it's a leaf, we process it.
    // Priority can be distance from query to the *point* in the node for internal nodes,
    // and for leaves, it's already a candidate.

    // Revised PQ approach more aligned with typical BBF:
    // PQ stores { -distance_to_node_region_boundary, node_ptr }
    // For simplicity, let's try a path-based BBF as per the general idea of "bins"

    pq.push({0.0, root, false}); // Priority 0, not necessarily leaf path yet
                                 // A common BBF pushes children based on distance to bounding box.
                                 // The provided pseudocode is tricky with priority.

    int searched_leaf_nodes = 0;

    // Initialize best_bbf with a point from the tree, e.g., root, or check root's children first.
    // Or, more simply, the first leaf we encounter.
    // For now, best_bbf.dist_sq is infinity.

    // Iterative descent to first leaf to initialize best_bbf
    KdNode *temp_node = root;
    while (temp_node && (temp_node->left || temp_node->right))
    { // while not leaf
        double d_sq_temp = euclidean_distance_sq(query, temp_node->point);
        if (d_sq_temp < best_bbf.dist_sq)
        {
            best_bbf.dist_sq = d_sq_temp;
            best_bbf.point = temp_node->point;
        }
        int axis = temp_node->split_dimension;
        if (query[axis] < temp_node->point[axis])
        {
            temp_node = temp_node->left;
        }
        else
        {
            temp_node = temp_node->right;
        }
    }
    if (temp_node)
    { // temp_node is now a leaf
        double d_sq_temp = euclidean_distance_sq(query, temp_node->point);
        if (d_sq_temp < best_bbf.dist_sq)
        {
            best_bbf.dist_sq = d_sq_temp;
            best_bbf.point = temp_node->point;
        }
        searched_leaf_nodes++;
    }

    // Re-initialize PQ for BBF logic after finding first leaf estimate
    while (!pq.empty())
        pq.pop();                 // Clear PQ
    pq.push({-0.0, root, false}); // Push root again, priority based on dist to point for internal

    while (!pq.empty() && searched_leaf_nodes < t_max_leaf_nodes)
    {
        BBFQueueEntry current_entry = pq.top();
        pq.pop();
        KdNode *node = current_entry.node;

        if (!node)
            continue;

        // If node's region cannot possibly contain a better point, prune
        // (This requires distance to bounding box, which is more complex.
        // The pseudocode doesn't explicitly show this pruning for internal nodes before processing)
        // Distance to point in internal node is a heuristic.
        if (-current_entry.priority > best_bbf.dist_sq && !current_entry.is_leaf_path)
        { // Negative of priority is dist_sq
          // If the point itself in this internal node is already worse than best,
          // and we are not forced to go down this path to a leaf.
          // This pruning might be too aggressive or not perfectly aligned with "distance_to_split".
          // continue; // This is tricky. Let's follow the given pseudo logic more closely.
        }

        if (!node->left && !node->right)
        { // Node is a leaf
            searched_leaf_nodes++;
            double d_sq = euclidean_distance_sq(query, node->point);
            if (d_sq < best_bbf.dist_sq)
            {
                best_bbf.dist_sq = d_sq;
                best_bbf.point = node->point;
            }
        }
        else
        { // Node is internal
            // Process point in internal node as potential candidate
            double d_sq_internal = euclidean_distance_sq(query, node->point);
            if (d_sq_internal < best_bbf.dist_sq)
            {
                best_bbf.dist_sq = d_sq_internal;
                best_bbf.point = node->point;
            }

            int axis = node->split_dimension;
            double dist_to_plane = std::abs(static_cast<double>(query[axis]) - node->point[axis]);
            // Priority: higher for smaller dist_to_plane. Using -dist_to_plane for max_heap.
            // The pseudo code uses 1/distance. If distance is 0, this is an issue.
            // Using -distance is safer.
            double priority_val = (dist_to_plane == 0) ? std::numeric_limits<double>::max() : -dist_to_plane; // Max priority if on plane

            KdNode *nearer_subtree, *farther_subtree;
            if (query[axis] < node->point[axis])
            {
                nearer_subtree = node->left;
                farther_subtree = node->right;
            }
            else
            {
                nearer_subtree = node->right;
                farther_subtree = node->left;
            }

            if (farther_subtree)
            {
                // The priority in pseudocode "1/(distance_to_split)" suggests the parent's split.
                // Here, it's the current node's split that determines the path to children.
                // Let's use priority based on distance to hyperplane for the farther path.
                // A leaf path should have high priority to be explored once decided.
                pq.push({priority_val, farther_subtree, false});
            }
            if (nearer_subtree)
            {
                // Nearer subtree should be higher priority (less negative or more positive)
                // Or, simply explore it next in a depth-first manner then add farther to PQ.
                // The pseudocode suggests adding both to PQ.
                // Priority for nearer can be set very high to explore it "almost" first from PQ.
                pq.push({std::numeric_limits<double>::max(), nearer_subtree, false}); // Max priority for nearer
            }
        }
    }
    if (best_bbf.dist_sq == std::numeric_limits<double>::infinity() && root)
    {                                                                               // Should not happen if tree not empty
        return {root->point, std::sqrt(euclidean_distance_sq(query, root->point))}; // Fallback
    }
    return {best_bbf.point, std::sqrt(best_bbf.dist_sq)};
}

// --- 8. Data Loading ---
bool load_data(const std::string &filepath,
               std::vector<Point> &points,
               std::vector<Point> &queries,
               int &n_out, int &m_out, int &d_out)
{
    std::ifstream infile(filepath);
    if (!infile.is_open())
    {
        std::cerr << "Error opening file: " << filepath << std::endl;
        return false;
    }

    infile >> n_out >> m_out >> d_out;
    K_DIM = d_out; // Set global dimension

    points.resize(n_out, Point(d_out));
    for (int i = 0; i < n_out; ++i)
    {
        points[i].id = i; // Assign an ID
        for (int j = 0; j < d_out; ++j)
        {
            infile >> points[i][j];
        }
    }

    queries.resize(m_out, Point(d_out));
    for (int i = 0; i < m_out; ++i)
    {
        queries[i].id = i; // Assign an ID (for query identification)
        for (int j = 0; j < d_out; ++j)
        {
            infile >> queries[i][j];
        }
    }
    infile.close();
    return true;
}

// --- 9. Memory Usage (Helper) ---
long get_memory_usage_kb()
{
#if defined(__linux__) || defined(__APPLE__)
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0)
    {
#ifdef __APPLE__
        return usage.ru_maxrss / 1024; // macOS reports in bytes
#else
        return usage.ru_maxrss; // Linux reports in kilobytes
#endif
    }
#endif
    return -1; // Not supported or error
}

int main(int argc, char *argv[])
{
    // Instead of taking a single filepath, we might loop through files
    // or take a base path and number of files.
    // For simplicity, let's assume we modify it to loop from 1.txt to 100.txt

    int T_datasets = 100; // Number of dataset files to process as per generateData.cpp
    int t_bbf = 500;

    std::vector<double> all_files_bf_avg_times;
    std::vector<double> all_files_kdt_avg_times;
    std::vector<double> all_files_bbf_avg_times;

    std::vector<double> all_files_bf_accuracies; // If you need to average accuracy from different files
    std::vector<double> all_files_kdt_accuracies;
    std::vector<double> all_files_bbf_accuracies;

    long total_process_mem_kb_sum = 0;
    int valid_mem_reads = 0;

    for (int t_file_idx = 1; t_file_idx <= T_datasets; ++t_file_idx)
    {
        char current_filepath_cstr[150];
        sprintf(current_filepath_cstr, "./data/%d.txt", t_file_idx); // Assuming data is in ./data/
        std::string current_filepath = current_filepath_cstr;

        std::cout << "\n======================================================" << std::endl;
        std::cout << "Processing Dataset: " << current_filepath << std::endl;
        std::cout << "======================================================" << std::endl;

        std::vector<Point> points, queries;
        int n_actual, m_actual, d_actual;

        if (!load_data(current_filepath, points, queries, n_actual, m_actual, d_actual))
        {
            std::cerr << "Failed to load " << current_filepath << ", skipping." << std::endl;
            continue;
        }
        std::cout << "Data loaded: " << n_actual << " points, "
                  << m_actual << " queries, " << d_actual << " dimensions." << std::endl;

        if (points.empty() || queries.empty())
        {
            std::cerr << "No points or queries loaded for " << current_filepath << ", skipping." << std::endl;
            continue;
        }
        K_DIM = d_actual;

        std::vector<Point> points_for_kdtree = points;
        KdNode *kdtree_root = build_kdtree(points_for_kdtree);

        if (!kdtree_root && !points.empty())
        { // check if kdtree build failed for non-empty points
            std::cerr << "KD-Tree build failed for " << current_filepath << ", skipping." << std::endl;
            destroy_kdtree(kdtree_root); // ensure cleanup if partially built
            continue;
        }

        int num_queries_to_run_current_file = queries.size();
        std::vector<std::pair<Point, double>> ground_truth_nns(num_queries_to_run_current_file);

        // --- Ground Truth for current file ---
        for (int i = 0; i < num_queries_to_run_current_file; ++i)
        {
            ground_truth_nns[i] = brute_force_nn(points, queries[i]);
        }

        // --- Brute-force Test for current file ---
        int bf_correct_count_current_file = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_queries_to_run_current_file; ++i)
        {
            auto nn_pair = brute_force_nn(points, queries[i]);
            if (nn_pair.second <= ground_truth_nns[i].second * 1.000001)
            {
                bf_correct_count_current_file++;
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto bf_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double bf_avg_time_us_current_file = num_queries_to_run_current_file > 0 ? static_cast<double>(bf_total_duration.count()) / num_queries_to_run_current_file : 0.0;
        all_files_bf_avg_times.push_back(bf_avg_time_us_current_file);
        all_files_bf_accuracies.push_back(num_queries_to_run_current_file > 0 ? static_cast<double>(bf_correct_count_current_file) / num_queries_to_run_current_file * 100.0 : 0.0);

        // --- Standard k-d Tree Test for current file ---
        int kdt_correct_count_current_file = 0;
        start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_queries_to_run_current_file; ++i)
        {
            if (!kdtree_root && points.empty())
            { // Handle case where tree is empty because points were empty
              // Cannot perform search, maybe log this or assign a specific error value
            }
            else if (!kdtree_root && !points.empty())
            {
                std::cerr << "Error: kdtree_root is null but points list was not empty. Skipping k-d tree search for this query." << std::endl;
            }
            else
            {
                auto nn_pair = kdtree_nn_search(kdtree_root, queries[i]);
                if (nn_pair.second <= ground_truth_nns[i].second * 1.000001)
                {
                    kdt_correct_count_current_file++;
                }
            }
        }
        end_time = std::chrono::high_resolution_clock::now();
        auto kdt_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double kdt_avg_time_us_current_file = num_queries_to_run_current_file > 0 ? static_cast<double>(kdt_total_duration.count()) / num_queries_to_run_current_file : 0.0;
        all_files_kdt_avg_times.push_back(kdt_avg_time_us_current_file);
        all_files_kdt_accuracies.push_back(num_queries_to_run_current_file > 0 ? static_cast<double>(kdt_correct_count_current_file) / num_queries_to_run_current_file * 100.0 : 0.0);

        // --- BBF Test for current file ---
        int bbf_correct_count_current_file = 0;
        start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_queries_to_run_current_file; ++i)
        {
            if (!kdtree_root && points.empty())
            {
                // Handle
            }
            else if (!kdtree_root && !points.empty())
            {
                std::cerr << "Error: kdtree_root is null but points list was not empty. Skipping BBF search for this query." << std::endl;
            }
            else
            {
                auto nn_pair = bbf_search(kdtree_root, queries[i], t_bbf);
                if (ground_truth_nns[i].second == 0 && nn_pair.second == 0)
                {
                    bbf_correct_count_current_file++;
                }
                else if (ground_truth_nns[i].second > 0 && (nn_pair.second / ground_truth_nns[i].second) <= 1.05)
                {
                    bbf_correct_count_current_file++;
                }
            }
        }
        end_time = std::chrono::high_resolution_clock::now();
        auto bbf_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double bbf_avg_time_us_current_file = num_queries_to_run_current_file > 0 ? static_cast<double>(bbf_total_duration.count()) / num_queries_to_run_current_file : 0.0;
        all_files_bbf_avg_times.push_back(bbf_avg_time_us_current_file);
        all_files_bbf_accuracies.push_back(num_queries_to_run_current_file > 0 ? static_cast<double>(bbf_correct_count_current_file) / num_queries_to_run_current_file * 100.0 : 0.0);

        destroy_kdtree(kdtree_root); // Clean up tree for this file
        kdtree_root = nullptr;

        long current_mem = get_memory_usage_kb();
        if (current_mem != -1)
        {
            total_process_mem_kb_sum += current_mem;
            valid_mem_reads++;
        }
        std::cout << "Finished processing: " << current_filepath << std::endl;
    }

    // --- Calculate overall averages ---
    double overall_bf_avg_time = 0;
    if (!all_files_bf_avg_times.empty())
    {
        for (double t : all_files_bf_avg_times)
            overall_bf_avg_time += t;
        overall_bf_avg_time /= all_files_bf_avg_times.size();
    }
    // ... similar calculations for kdt_avg_time, bbf_avg_time
    // ... and for accuracies

    double overall_kdt_avg_time = 0;
    if (!all_files_kdt_avg_times.empty())
    {
        for (double t : all_files_kdt_avg_times)
            overall_kdt_avg_time += t;
        overall_kdt_avg_time /= all_files_kdt_avg_times.size();
    }

    double overall_bbf_avg_time = 0;
    if (!all_files_bbf_avg_times.empty())
    {
        for (double t : all_files_bbf_avg_times)
            overall_bbf_avg_time += t;
        overall_bbf_avg_time /= all_files_bbf_avg_times.size();
    }

    double overall_bf_accuracy = 0;
    if (!all_files_bf_accuracies.empty())
    {
        for (double acc : all_files_bf_accuracies)
            overall_bf_accuracy += acc;
        overall_bf_accuracy /= all_files_bf_accuracies.size();
    }
    double overall_kdt_accuracy = 0;
    if (!all_files_kdt_accuracies.empty())
    {
        for (double acc : all_files_kdt_accuracies)
            overall_kdt_accuracy += acc;
        overall_kdt_accuracy /= all_files_kdt_accuracies.size();
    }
    double overall_bbf_accuracy = 0;
    if (!all_files_bbf_accuracies.empty())
    {
        for (double acc : all_files_bbf_accuracies)
            overall_bbf_accuracy += acc;
        overall_bbf_accuracy /= all_files_bbf_accuracies.size();
    }

    std::cout << "\n--- Overall Results Summary (Averaged over " << T_datasets << " datasets) ---" << std::endl;
    std::cout << std::left << std::setw(25) << "Algorithm"
              << std::setw(25) << "Avg Time/Query (us)"
              << std::setw(20) << "Avg Accuracy (%)" << std::endl;
    std::cout << std::setw(25) << "Brute-force"
              << std::setw(25) << overall_bf_avg_time
              << std::setw(20) << overall_bf_accuracy << std::endl;
    std::cout << std::setw(25) << "Standard k-d Tree"
              << std::setw(25) << overall_kdt_avg_time
              << std::setw(20) << overall_kdt_accuracy << std::endl;
    std::cout << std::setw(25) << "BBF (t=" + std::to_string(t_bbf) + ")"
              << std::setw(25) << overall_bbf_avg_time
              << std::setw(20) << overall_bbf_accuracy << std::endl;
    std::string filename = "log.txt";
    std::fstream fs;
    fs.open(filename, std::ios::out | std::ios::app);
    fs << "\n--- Overall Results Summary (Averaged over " << T_datasets << " datasets) ---" << std::endl;
    fs << std::left << std::setw(25) << "Algorithm"
       << std::setw(25) << "Avg Time/Query (us)"
       << std::setw(20) << "Avg Accuracy (%)" << std::endl;
    fs << std::setw(25) << "Brute-force"
       << std::setw(25) << overall_bf_avg_time
       << std::setw(20) << overall_bf_accuracy << std::endl;
    fs << std::setw(25) << "Standard k-d Tree"
       << std::setw(25) << overall_kdt_avg_time
       << std::setw(20) << overall_kdt_accuracy << std::endl;
    fs << std::setw(25) << "BBF (t=" + std::to_string(t_bbf) + ")"
       << std::setw(25) << overall_bbf_avg_time
       << std::setw(20) << overall_bbf_accuracy << std::endl;
    fs << K_DIM << ' ' << std::endl;
    fs.close();

    if (valid_mem_reads > 0)
    {
        std::cout << "\nAverage peak process memory usage across runs: " << total_process_mem_kb_sum / valid_mem_reads << " KB (if supported)" << std::endl;
    }
    else
    {
        std::cout << "\nMemory usage reporting not fully supported or no files processed." << std::endl;
    }
    std::cout << "Note: Memory usage is total for the process (RSS). For specific structure sizes, estimate based on counts and sizeof." << std::endl;

    std::cout << "\nDone." << std::endl;

    return 0;
}