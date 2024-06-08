import numpy as np
import pandas as pd

# 데이터 로드
file_path = 'heart_statlog_cleveland_hungary_final.csv'
data = pd.read_csv(file_path)

# target 열을 따로 저장
target = data['target']

# target 열을 제거한 데이터프레임
data = data.drop(columns=['target'])

# 더미 피처 추가
np.random.seed(0)  # 재현성을 위해 랜덤 시드 설정
num_dummy_features = 2
for i in range(num_dummy_features):
    data[f'dummy_{i+1}'] = np.random.uniform(low=0.0, high=1.0, size=data.shape[0])

# target 열을 다시 데이터프레임의 마지막에 추가
data['target'] = target

# 피처 이름 저장 (target 열을 제외하고 나머지 피처 이름만 저장)
feature_names = data.columns[:-1]  # 마지막 열이 target이므로 제외

print(feature_names)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 데이터 전처리: NaN 및 inf 값 확인 및 제거
if np.any(np.isnan(X)) or np.any(np.isinf(X)):
    print("NaN or inf values found in the dataset. Replacing with zeros.")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

class_labels = np.unique(y)

# 데이터 분할 함수
def train_test_split(X, y, test_size=0.3):
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# 정확도 계산 함수
def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# 간단한 분류기 구현 (k-NN)
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def predict(X_train, y_train, X_test, k=3):
    y_pred = []
    for test_point in X_test:
        distances = [euclidean_distance(test_point, x) for x in X_train]
        k_nearest = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_nearest]
        y_pred.append(np.argmax(np.bincount(k_nearest_labels)))
    return np.array(y_pred)

# Sequential Feature Selection
def sequential_feature_selection(X_train, y_train, X_test, y_test):
    n_features = X_train.shape[1]
    selected_features = []
    remaining_features = list(range(n_features))
    best_accuracy = 0
    best_feature_set = None

    while remaining_features:
        accuracies = []
        for feature in remaining_features:
            current_features = selected_features + [feature]
            X_train_subset = X_train[:, current_features]
            X_test_subset = X_test[:, current_features]
            
            mean_vectors = []
            for cl in np.unique(y_train):
                mean_vectors.append(np.mean(X_train_subset[y_train == cl], axis=0))
            overall_mean = np.mean(X_train_subset, axis=0)

            S_W = np.zeros((X_train_subset.shape[1], X_train_subset.shape[1]))
            for cl, mv in zip(np.unique(y_train), mean_vectors):
                if len(X_train_subset[y_train == cl]) > 1:
                    class_scatter = np.cov(X_train_subset[y_train == cl].T)
                else:
                    class_scatter = np.zeros((X_train_subset.shape[1], X_train_subset.shape[1]))
                S_W += class_scatter

            S_B = np.zeros((X_train_subset.shape[1], X_train_subset.shape[1]))
            for i, mean_vec in enumerate(mean_vectors):
                n = X_train_subset[y_train == np.unique(y_train)[i], :].shape[0]
                mean_vec = mean_vec.reshape(X_train_subset.shape[1], 1)
                overall_mean = overall_mean.reshape(X_train_subset.shape[1], 1)
                S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

            eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

            eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
            eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

            k = len(np.unique(y_train)) - 1
            W = np.hstack([eig_pairs[i][1].reshape(X_train_subset.shape[1], 1) for i in range(k)])

            X_train_lda = X_train_subset.dot(W).real
            X_test_lda = X_test_subset.dot(W).real

            y_pred = predict(X_train_lda, y_train, X_test_lda, k=3)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append((accuracy, feature))

        accuracies.sort(reverse=True, key=lambda x: x[0])
        best_accuracy, best_feature = accuracies[0]
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

        if best_feature_set is None or best_accuracy > best_feature_set[1]:
            best_feature_set = (selected_features.copy(), best_accuracy)

    return best_feature_set[0]

# 기존 LDA 모델의 정확도 계산
def lda_accuracy(X_train, X_test, y_train, y_test):

    if X_train.ndim > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)
    if X_test.ndim > 2:
        X_test = X_test.reshape(X_test.shape[0], -1)
        
    mean_vectors = []
    for cl in np.unique(y_train):
        mean_vectors.append(np.mean(X_train[y_train == cl], axis=0))
    overall_mean = np.mean(X_train, axis=0)

    S_W = np.zeros((X_train.shape[1], X_train.shape[1]))
    for cl, mv in zip(np.unique(y_train), mean_vectors):
        if len(X_train[y_train == cl]) > 1:
            class_scatter = np.cov(X_train[y_train == cl].T)
        else:
            class_scatter = np.zeros((X_train.shape[1], X_train.shape[1]))
        S_W += class_scatter

    S_B = np.zeros((X_train.shape[1], X_train.shape[1]))
    for i, mean_vec in enumerate(mean_vectors):
        n = X_train[y_train == np.unique(y_train)[i], :].shape[0]
        mean_vec = mean_vec.reshape(X_train.shape[1], 1)
        overall_mean = overall_mean.reshape(X_train.shape[1], 1)
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    k = len(np.unique(y_train)) - 1
    W = np.hstack([eig_pairs[i][1].reshape(X_train.shape[1], 1) for i in range(k)])

    X_train_lda = X_train.dot(W).real
    X_test_lda = X_test.dot(W).real

    y_pred = predict(X_train_lda, y_train, X_test_lda, k=3)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
# 기존 LDA 모델 정확도
lda_acc = lda_accuracy(X_train, X_test, y_train, y_test)
    
# SFS를 사용한 LDA 모델 정확도
best_features = sequential_feature_selection(X_train, y_train, X_test, y_test)
best_feature_names = feature_names[best_features]  # 피처 이름으로 변환
print(f"선택된 피처 (SFS): {best_feature_names}")
X_train_best = X_train[:, best_features]
X_test_best = X_test[:, best_features]
sfs_acc = lda_accuracy(X_train_best, X_test_best, y_train, y_test)
    
print(f"기존 LDA 모델의 정확도: {lda_acc * 100:.2f}%")
print(f"SFS를 사용한 LDA 모델의 정확도: {sfs_acc * 100:.2f}%")
print("-" * 50)
