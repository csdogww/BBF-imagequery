import time
import cv2
import numpy as np
import os
from sklearn.cluster import AgglomerativeClustering # 或者 KMeans
from sklearn.neighbors import KDTree # 或者使用OpenCV的FLANN
from scipy.cluster.hierarchy import linkage, fcluster # 另一种层次聚类方式
from sklearn.preprocessing import normalize
import glob # 用于查找文件路径

# --- 配置参数 ---
K = 2000  # 视觉词典大小 (可调整)
TOP_K_RESULTS = 5 # 返回最相似的图像数量
FEATURE_EXTRACTOR_TYPE = 'ORB' # 'SIFT' or 'ORB'
DATASET_PATH = './ImageSet/' # 你的图像数据集路径

QUERY_IMAGE_PATH_POST = '358.png' # 你的查询图像路径 (示例)
queries=['11.png','101.png','358.png']

# --- 1. 特征提取 ---
def extract_features(image_path, extractor_type='SIFT'):
    """
    从单个图像提取特征描述符
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None

    # 添加图像预处理
    # 1. 调整图像大小
    min_size = 300
    h, w = img.shape[:2]
    if min(h, w) < min_size:
        scale = min_size / min(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
    
    # 2. 增强对比度
    img = cv2.equalizeHist(img)

    if extractor_type == 'SIFT':
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
    elif extractor_type == 'ORB':
        orb = cv2.ORB_create(
            nfeatures=1000,          # 增加特征点数量
            scaleFactor=1.2,         # 金字塔缩放因子
            nlevels=8,               # 金字塔层数
            edgeThreshold=31,        # 边缘阈值
            firstLevel=0,
            WTA_K=2,                 # 每个描述符的点数
            patchSize=31,            # 特征点邻域大小
            fastThreshold=10         # 降低 FAST 阈值，使其更容易检测到角点
        )
        keypoints, descriptors = orb.detectAndCompute(img, None)
        
        # 添加调试信息
        if keypoints is None or len(keypoints) == 0:
            print(f"No keypoints found in {image_path}")
        elif descriptors is None:
            print(f"Found {len(keypoints)} keypoints but no descriptors in {image_path}")
        else:
            #print(f"Found {len(keypoints)} keypoints and descriptors shape: {descriptors.shape}")
            a=0
            
    else:
        raise ValueError("Unsupported feature extractor type")

    return descriptors

def load_all_descriptors(dataset_path, extractor_type='SIFT'):
    """
    从数据集中所有图像提取并收集所有特征描述符
    """
    all_descriptors_list = []
    image_paths = []
    # 支持多种常见图像格式
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff'):
        image_paths.extend(glob.glob(os.path.join(dataset_path, ext)))

    if not image_paths:
        print(f"No images found in {dataset_path}")
        return None, []

    print(f"Found {len(image_paths)} images. Extracting features...")

    for i, img_path in enumerate(image_paths):
        #print(f"Processing image {i+1}/{len(image_paths)}: {img_path}")
        descriptors = extract_features(img_path, extractor_type)
        if descriptors is not None and len(descriptors) > 0:
            all_descriptors_list.append(descriptors)
        else:
            print(f"Warning: No descriptors found for {img_path}")
            # 从 image_paths 中移除没有提取到特征的图像，或者在后续处理中跳过
            # 为简单起见，这里我们只收集有效的描述符，但最好记录哪些图像失败了

    if not all_descriptors_list:
        print("No descriptors extracted from any image.")
        return None, image_paths # image_paths 仍然返回，即使有些可能没有描述符

    all_features = np.vstack(all_descriptors_list)
    print(f"Total descriptors extracted: {all_features.shape[0]}")
    return all_features, image_paths

# --- 2. 构建视觉词典 ---
def build_visual_vocabulary(all_features, K_clusters, method='agglomerative'):
    """
    使用特征描述符构建视觉词典
    """
    print(f"Building visual vocabulary with K={K_clusters} using {method} clustering...")
    if all_features is None or all_features.shape[0] < K_clusters:
        print("Not enough features to build vocabulary or no features provided.")
        return None

    if method == 'agglomerative':
        # Scikit-learn的AgglomerativeClustering可能对大量数据较慢且内存消耗大
        # AgglomerativeClustering 不直接返回簇中心，需要自己计算
        # cluster_model = AgglomerativeClustering(n_clusters=K_clusters)
        # cluster_labels = cluster_model.fit_predict(all_features)

        # 或者使用 scipy 的层次聚类，然后提取簇
        # 注意：对非常大的 all_features (例如几十万x128维)，linkage也会很慢
        # 可能需要对 all_features 进行采样或使用 MiniBatchKMeans 等更高效的方法
        if all_features.shape[0] > 60000: # 如果特征太多，层次聚类会非常慢
            print(f"Warning: Too many features ({all_features.shape[0]}) for direct Agglomerative Clustering. Consider sampling or KMeans.")
            # 对特征进行采样
            sample_indices = np.random.choice(all_features.shape[0], size=min(60000, all_features.shape[0]), replace=False)
            features_to_cluster = all_features[sample_indices]
            print(f"Clustering on a sample of {features_to_cluster.shape[0]} features.")
        else:
            features_to_cluster = all_features

        # 'ward' 是一种常用的 linkage 方法
        linked_features = linkage(features_to_cluster, method='ward', metric='euclidean')
        # fcluster 从层次聚类结果中形成扁平簇
        # 'maxclust' 意味着我们想要 K_clusters 个簇
        cluster_labels_sampled = fcluster(linked_features, K_clusters, criterion='maxclust')

        # 计算视觉单词 (簇中心)
        # 注意：这里的cluster_labels_sampled 是针对 features_to_cluster 的
        # 真正的词典应该代表所有特征的分布，所以用KMeans可能更直接
        # 或者，如果你坚持用层次聚类，并且聚类的是采样点，那么这些采样点的簇中心就是词典
        vocabulary = np.array([features_to_cluster[cluster_labels_sampled == i].mean(axis=0)
                               for i in np.unique(cluster_labels_sampled)])

    elif method == 'kmeans': # 更常用且高效的方法
        from sklearn.cluster import KMeans # 或者 MiniBatchKMeans 更快
        kmeans = KMeans(n_clusters=K_clusters, random_state=0, n_init='auto')
        kmeans.fit(all_features)
        vocabulary = kmeans.cluster_centers_
    else:
        raise ValueError("Unsupported clustering method")

    print(f"Vocabulary built with shape: {vocabulary.shape}") # 应该是 K x D
    return vocabulary

# --- 3. 生成数据库词袋 ---
def descriptors_to_bow(descriptors, vocabulary_tree, K_vocab): # <--- 增加 K_vocab 参数
    """
    将一张图像的描述符转换为BoW向量
    """
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(K_vocab) # 使用传入的 K_vocab

    distances, indices = vocabulary_tree.query(descriptors, k=1)
    indices = indices.flatten()

    # 生成词频直方图
    hist, _ = np.histogram(indices, bins=np.arange(K_vocab + 1)) # 使用传入的 K_vocab

    hist_norm = normalize(hist.reshape(1, -1), norm='l2')[0]
    return hist_norm

def generate_database_bows(dataset_path, image_paths_from_extraction, vocabulary, extractor_type='SIFT'):
    """
    为数据库中的所有图像生成BoW表示
    """
    print("Generating Bag-of-Words for database images...")
    if vocabulary is None:
        print("Vocabulary is not built. Cannot generate BoW.")
        return None, None

    # 用视觉词典构建KD树，用于快速查找最近的视觉单词
    vocab_tree = KDTree(vocabulary)
    K_vocab = vocabulary.shape[0]

    database_bows = []
    valid_image_paths = [] # 存储实际生成了BoW的图像路径

    # 重新提取特征并生成BoW，或者如果之前保存了每个图像的描述符，可以重用
    # 这里选择重新提取，以保持流程清晰
    for img_path in image_paths_from_extraction:
        descriptors = extract_features(img_path, extractor_type)
        if descriptors is not None and len(descriptors) > 0:
            bow_vector = descriptors_to_bow(descriptors, vocab_tree,K_vocab)
            database_bows.append(bow_vector)
            valid_image_paths.append(img_path)
        else:
            print(f"Skipping BoW generation for {img_path} due to no descriptors.")
            # 或者可以添加一个全零的BoW向量，但这可能影响相似度计算
            # database_bows.append(np.zeros(K_vocab))
            # valid_image_paths.append(img_path)


    if not database_bows:
        print("No BoW vectors generated for the database.")
        return None, None

    return np.array(database_bows), valid_image_paths

# --- 4. 查询处理 ---
def cosine_similarity_calc(vec1, vec_array):
    """
    计算一个向量与一个向量数组中每个向量的余弦相似度
    """
    vec1_norm = np.linalg.norm(vec1)
    if vec1_norm == 0: return np.zeros(vec_array.shape[0])

    vec_array_norm = np.linalg.norm(vec_array, axis=1)
    # 处理分母为0的情况
    denominator = vec1_norm * vec_array_norm
    # similarities = np.dot(vec_array, vec1) / denominator
    # 改为安全的除法
    similarities = np.zeros(vec_array.shape[0])
    valid_indices = denominator > 1e-9 # 避免除以非常小的数
    similarities[valid_indices] = np.dot(vec_array[valid_indices], vec1) / denominator[valid_indices]

    return similarities

def query_image(query_image_path, vocabulary, vocab_tree, database_bows, db_image_paths, top_k, extractor_type='SIFT'):
    """
    处理查询图像并返回最相似的图像
    """
    print(f"\nProcessing query image: {query_image_path}")
    if vocabulary is None or vocab_tree is None or database_bows is None or not db_image_paths:
        print("System not ready. Vocabulary, KDTree, or database BoWs missing.")
        return []

    query_descriptors = extract_features(query_image_path, extractor_type)
    if query_descriptors is None or len(query_descriptors) == 0:
        print("Could not extract features from query image.")
        return []

    query_hist = descriptors_to_bow(query_descriptors, vocab_tree,vocabulary.shape[0]) # 使用传入的 K_vocab

    if np.sum(query_hist) == 0: # 如果查询图像的BoW是全零（例如，没有匹配到任何视觉单词）
        print("Query image BoW is all zeros. Cannot compute similarity meaningfully.")
        return []

    similarities = cosine_similarity_calc(query_hist, database_bows)

    # 获取相似度最高的 top_k 个索引
    # argsort 返回的是从小到大排序的索引，所以取最后 top_k 个
    sorted_indices = np.argsort(similarities)
    top_indices = sorted_indices[-top_k:][::-1] # 取最后k个并反转，得到从大到小

    results = []
    print("\nTop similar images:")
    for i in top_indices:
        results.append({
            "path": db_image_paths[i],
            "similarity": similarities[i]
        })
        print(f"- {db_image_paths[i]} (Similarity: {similarities[i]:.4f})")
        with open('results.txt', 'a') as f:
            f.write(f"{db_image_paths[i]} (Similarity: {similarities[i]:.4f})\n")
    with open('results.txt', 'a') as f:
        f.write("\n")
    return results



# --- 主流程 ---
if __name__ == "__main__":
    K_values_to_test = [500,750, 1000,1250,1500,1750, 2000] # 测试不同的K值
    for K in K_values_to_test:
        total_feature_extraction_duration = 0
        total_build_vocabulary_duration = 0
        total_Bow_generate_duration=0
        query_time=0
        
        
        # 1. 特征提取 (对整个数据集)
        start_feature_extraction_time = time.time()
        all_features, all_image_paths = load_all_descriptors(DATASET_PATH, FEATURE_EXTRACTOR_TYPE)
        end_feature_extraction_time = time.time()
        feature_extraction_duration = end_feature_extraction_time - start_feature_extraction_time
        
        #print(f"Total feature extraction time: {total_feature_extraction_duration:.2f} seconds")
    
        if all_features is not None and len(all_image_paths) > 0:
                # 2. 构建视觉词典
                # 对于层次聚类，如果特征非常多，可能需要对all_features进行采样或使用KMeans
                # 这里我们用KMeans，因为它更适合大规模数据
            start_build_vocabulary_time = time.time()
            visual_vocabulary = build_visual_vocabulary(all_features, K, method='kmeans') # 改为kmeans
            end_build_vocabulary_time = time.time()
            build_vocabulary_duration = end_build_vocabulary_time - start_build_vocabulary_time
            

            if visual_vocabulary is not None:
                    # 3. 生成数据库词袋
                    # 使用构建词典时的 image_paths，因为有些图像可能没有特征
                    # 需要确保 database_bows 和 db_image_paths_valid 的顺序和对应关系正确
                start_Bow_generate_time= time.time()
                    
                vocab_kd_tree = KDTree(visual_vocabulary) # 为词典构建KDTree
                db_bow_vectors, db_image_paths_valid = generate_database_bows(DATASET_PATH, all_image_paths, visual_vocabulary, FEATURE_EXTRACTOR_TYPE)
                    
                end_Bow_generate_time = time.time()
                Bow_generate_duration = end_Bow_generate_time - start_Bow_generate_time
                
                    

                for query in queries:
                    if db_bow_vectors is not None and len(db_image_paths_valid) > 0:
                    # 4. 查询处理 (选择一个图像作为查询示例)
                        if len(db_image_paths_valid) > 0:
                    # 随机选择一张数据库中的图片作为查询，或指定一张
                    # query_image_example_path = np.random.choice(db_image_paths_valid)
                    # 或者你可以手动指定一张图片路径
                            query_image_example_path = os.path.join(DATASET_PATH, query) # 你需要替换成一个实际的查询图片名

                            if not os.path.exists(query_image_example_path):
                                print(f"Query image {query_image_example_path} not found. Using first valid DB image as query.")
                                query_image_example_path = db_image_paths_valid[0]

                            start_query_time = time.time()
                            
                            top_results = query_image(query_image_example_path,
                                                visual_vocabulary,
                                                vocab_kd_tree,
                                                db_bow_vectors,
                                                db_image_paths_valid,
                                                TOP_K_RESULTS,
                                                FEATURE_EXTRACTOR_TYPE)
                            
                            end_query_time = time.time()
                            query_duration = end_query_time - start_query_time
                            query_time += query_duration
                            
                        else:
                            print("No valid images in database to perform query.")
                    else:
                        print("Failed to generate BoW for database images.")
            else:
                print("Failed to extract features from dataset.")
            
        build_time=build_vocabulary_duration+Bow_generate_duration
        query_time=query_time/len(queries)
        with open ('time_results.txt', 'a') as f:
            f.write('-------------------------------------------------------\n')
            f.write(f"Total feature extraction time for K={K} : {feature_extraction_duration:.2f} seconds\n")
            f.write(f"Total build vocabulary time for K={K} : {build_vocabulary_duration:.2f} seconds\n")
            f.write(f"Total Bow generate time for K={K} : {Bow_generate_duration:.2f} seconds\n")
            f.write(f"Total build time for K={K} : {build_time:.2f} seconds\n")
            f.write(f"Total query time for K={K} : {query_time:.2f} seconds\n")
            f.write('-------------------------------------------------------\n')
        print(f"Total feature extraction time for K={K} : {feature_extraction_duration:.2f} seconds")
        print(f"Total build vocabulary time for K={K} : {build_vocabulary_duration:.2f} seconds")
        print(f"Total Bow generate time for K={K} : {Bow_generate_duration:.2f} seconds")
        print(f"Total build time for K={K} : {build_time:.2f} seconds")
        print(f"Total query time for K={K} : {query_time:.2f} seconds")
        print('-------------------------------------------------------')
        
        