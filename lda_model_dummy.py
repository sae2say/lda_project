import pandas as pd
import random
import math
import numpy as np
import matplotlib.pyplot as plt

#######################################################
# 데이터 불러오기, dummy feature 추가, 데이터 전처리
file_path = 'heart_statlog_cleveland_hungary_final.csv'
data = pd.read_csv(file_path)

target = data['target']
data = data.drop(columns=['target']) # dummy data 추가 후에도 target이 마지막 열에 존재하도록. 

random.seed(0)
num_dummy_features = 2
for i in range(num_dummy_features):
    data[f'dummy_{i+1}'] = [random.random() for _ in range(data.shape[0])]

data['target'] = target
feature_names = data.columns[:-1] # 마지막 열은 target

X = data.iloc[:, :-1].values.tolist()
y = data.iloc[:, -1].values.tolist()

# 데이터 전처리: NaN 및 inf 값 확인 및 제거
def nan_inf(data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] is None or data[i][j] == float('inf') or data[i][j] == float('-inf'):
                data[i][j] = 0.0
nan_inf(X)
#######################################################

#######################################################
# 구현 시 필요한 보조 함수 작성

def mean(data): # 평균
    return sum(data) / len(data)

def covariance(X): # 공분산
    n_samples = len(X)
    mean_vec = [mean(feature) for feature in zip(*X)]
    cov_matrix = []

    for i in range(len(mean_vec)):
        cov_row = []
        for j in range(len(mean_vec)):
            cov = sum((X[k][i] - mean_vec[i]) * (X[k][j] - mean_vec[j]) for k in range(n_samples)) / (n_samples - 1)
            cov_row.append(cov)
        cov_matrix.append(cov_row)
    
    return cov_matrix

def argsort(arr): # argument 정렬
    return sorted(range(len(arr)), key=lambda x: arr[x])

def unique(arr): # unique한 값 리스트로 반환
    unique_arr = []
    for elem in arr:
        if elem not in unique_arr:
            unique_arr.append(elem)
    return unique_arr

def train_test_split(X, y, test_size=0.3): # 데이터셋 분할
    indices = list(range(len(X)))
    random.shuffle(indices)
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]
    return X_train, X_test, y_train, y_test

def accuracy_score(y_true, y_pred): # 정확도 계산
    return sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred)) / len(y_true)

def euclidean_distance(a, b): # 간단한 KNN 분류기
    return math.sqrt(sum((a_i - b_i) ** 2 for a_i, b_i in zip(a, b)))

def predict(X_train, y_train, X_test, k=3):
    y_pred = []
    for test_point in X_test:
        distances = [euclidean_distance(test_point, x) for x in X_train]
        k_nearest = argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_nearest]
        y_pred.append(max(set(k_nearest_labels), key=k_nearest_labels.count))
    return y_pred

#######################################################

#######################################################
# 메인 구현 기능

def gradual_selection(X_train, y_train, X_test, y_test, k=3):
    n_features = len(X_train[0])
    selected_features = []
    remaining_features = list(range(n_features))
    best_accuracy = 0
    best_feature_set = None

    while remaining_features:
        accuracies = []
        for feature in remaining_features:
            current_features = selected_features + [feature]
            X_train_subset = [[row[f] for f in current_features] for row in X_train]
            X_test_subset = [[row[f] for f in current_features] for row in X_test]

            mean_vectors = []
            for cl in unique(y_train):
                class_subset = [X_train_subset[i] for i in range(len(y_train)) if y_train[i] == cl]
                mean_vectors.append([mean(feature) for feature in zip(*class_subset)])
            overall_mean = [mean(feature) for feature in zip(*X_train_subset)]

            S_W = [[0.0] * len(current_features) for _ in range(len(current_features))]
            for cl, mv in zip(unique(y_train), mean_vectors):
                class_subset = [X_train_subset[i] for i in range(len(y_train)) if y_train[i] == cl]
                class_scatter = covariance(class_subset)
                S_W = [[S_W[i][j] + class_scatter[i][j] for j in range(len(class_scatter[i]))] for i in range(len(class_scatter))]

            S_B = [[0.0] * len(current_features) for _ in range(len(current_features))]
            for i, mean_vec in enumerate(mean_vectors):
                n = sum(1 for val in y_train if val == unique(y_train)[i])
                mean_diff = [mean_vec[j] - overall_mean[j] for j in range(len(mean_vec))]
                mean_diff = [[m] for m in mean_diff] 
                outer_product = [[mean_diff[x][0] * mean_diff[y][0] for y in range(len(mean_diff))] for x in range(len(mean_diff))]
                S_B = [[S_B[i][j] + n * outer_product[i][j] for j in range(len(outer_product[i]))] for i in range(len(outer_product))]

            S_W_inv = np.linalg.inv(S_W)
            eig_vals, eig_vecs = np.linalg.eig(np.dot(S_W_inv, S_B))

            eig_pairs = sorted([(abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))], key=lambda k: k[0], reverse=True)

            k_eig = len(unique(y_train)) - 1
            W = np.hstack([eig_pairs[i][1].reshape(len(current_features), 1) for i in range(k_eig)]).real

            X_train_lda = np.dot(np.array(X_train_subset), W)
            X_test_lda = np.dot(np.array(X_test_subset), W)

            y_pred = predict(X_train_lda.tolist(), y_train, X_test_lda.tolist(), k)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append((accuracy, feature))

        accuracies.sort(reverse=True, key=lambda x: x[0])
        best_accuracy, best_feature = accuracies[0]
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

        if best_feature_set is None or best_accuracy > best_feature_set[1]:
          best_feature_set = (selected_features.copy(), best_accuracy)


    return best_feature_set[0]

# 기존 LDA 모델 구현, 정확도 계산
def lda_accuracy(X_train, X_test, y_train, y_test, k=3):
    mean_vectors = []
    for cl in unique(y_train):
        class_subset = [X_train[i] for i in range(len(y_train)) if y_train[i] == cl]
        mean_vectors.append([mean(feature) for feature in zip(*class_subset)])
    overall_mean = [mean(feature) for feature in zip(*X_train)]

    S_W = [[0.0] * len(X_train[0]) for _ in range(len(X_train[0]))]
    for cl, mv in zip(unique(y_train), mean_vectors):
        class_subset = [X_train[i] for i in range(len(y_train)) if y_train[i] == cl]
        class_scatter = covariance(class_subset)
        S_W = [[S_W[i][j] + class_scatter[i][j] for j in range(len(class_scatter[i]))] for i in range(len(class_scatter))]

    S_B = [[0.0] * len(X_train[0]) for _ in range(len(X_train[0]))]
    for i, mean_vec in enumerate(mean_vectors):
        n = sum(1 for val in y_train if val == unique(y_train)[i])
        mean_diff = [mean_vec[j] - overall_mean[j] for j in range(len(mean_vec))]
        mean_diff = [[m] for m in mean_diff] 
        outer_product = [[mean_diff[x][0] * mean_diff[y][0] for y in range(len(mean_diff))] for x in range(len(mean_diff))]
        S_B = [[S_B[i][j] + n * outer_product[i][j] for j in range(len(outer_product[i]))] for i in range(len(outer_product))]

    S_W_inv = np.linalg.inv(S_W)
    eig_vals, eig_vecs = np.linalg.eig(np.dot(S_W_inv, S_B))
    eig_pairs = sorted([(abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))], key=lambda k: k[0], reverse=True)

    k_eig = len(unique(y_train)) - 1
    W = np.hstack([eig_pairs[i][1].reshape(len(X_train[0]), 1) for i in range(k_eig)]).real

    X_train_lda = np.dot(np.array(X_train), W)
    X_test_lda = np.dot(np.array(X_test), W)

    y_pred = predict(X_train_lda.tolist(), y_train, X_test_lda.tolist(), k)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, X_test_lda, y_pred

def plot_classification_results_1d(X, y_true, y_pred, title):
    plt.figure()
    for cl in unique(y_true):
        plt.scatter(
            X[np.array(y_pred) == cl],
            [0.0] * sum(np.array(y_pred) == cl),  # y축을 약간 다르게 하여 시각화
            marker='x',
            label=f'Class {cl} Pred'
        )
    plt.title(title)
    plt.xlabel('LDA Component 1')
    plt.yticks([])
    plt.legend()
    plt.show()


#######################################################
# 코드 수행, 결과 도출

feature_selection_count = [0] * len(X[0])

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # 기존 LDA 모델 정확도
    lda_acc, X_test_lda, y_pred_lda = lda_accuracy(X_train, X_test, y_train, y_test, k=3)

    # SFS를 사용한 LDA 모델 정확도
    best_features = gradual_selection(X_train, y_train, X_test, y_test, k=3)

    best_feature_names = [feature_names[bf] for bf in best_features]  # 피처 이름으로 변환

    print(f"반복 {i+1}: 선택된 피처 (SFS): {best_feature_names}")
    X_train_best = [[row[bf] for bf in best_features] for row in X_train]
    X_test_best = [[row[bf] for bf in best_features] for row in X_test]
    sfs_acc, X_test_lda_best, y_pred_sfs = lda_accuracy(X_train_best, X_test_best, y_train, y_test, k=3)

    print(f"반복 {i+1}: 기존 LDA 모델의 정확도: {lda_acc * 100:.2f}%")
    print(f"반복 {i+1}: SFS를 사용한 LDA 모델의 정확도: {sfs_acc * 100:.2f}%")
    print("-" * 50)

    # 선택된 피처의 카운트 증가
    for feature in best_features:
        feature_selection_count[feature] += 1

    plot_classification_results_1d(X_test_lda, y_test, y_pred_lda, f'LDA Classification Result (Iteration {i+1})')
    plot_classification_results_1d(X_test_lda_best, y_test, y_pred_sfs, f'SFS-LDA Classification Result (Iteration {i+1})')

selection_df = pd.DataFrame({
    'Feature': feature_names,
    'Selection Count': feature_selection_count
})

print(selection_df)

# 피처 선택 횟수 시각화
plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_names)), feature_selection_count, tick_label=feature_names)
plt.xlabel('Features')
plt.ylabel('Selection Count')
plt.title('Feature Selection Count Over 10 Iterations')
plt.xticks(rotation=90)
plt.show()

