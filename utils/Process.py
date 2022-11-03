from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

iris = load_iris()


def plot(X, y, title, x_label, y_label):
    ax = plt.subplot(1, 1, 1)
    for label, marker, color in zip(
            range(3), ('^', 's', 'o'), ('blue', 'red', 'green')):
        plt.scatter(x=X[:, 0].real[y == label],
                    y=X[:, 1].real[y == label],
                    color=color, alpha=0.5, label=label_dict[label]
                    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title(title)


iris_X, iris_y = iris.data, iris.target
# {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
label_dict = {i: k for i, k in enumerate(iris.target_names)}
print(label_dict)
print(30 * '-')

mean_vectors = []

for cl in [0, 1, 2]:
    class_mean_vector = np.mean(iris_X[iris_y == cl], axis=0)
    mean_vectors.append(class_mean_vector)
    print(label_dict[cl], class_mean_vector)

# 类内散布矩阵
S_W = np.zeros((4, 4))
# 对于每种鸢尾花
for cl, mv in zip([0, 1, 2], mean_vectors):
    class_sc_mat = np.zeros((4, 4))
    for row in iris_X[iris_y == cl]:
        row, mv = row.reshape(4, 1), mv.reshape(4, 1)

        class_sc_mat += (row - mv).dot((row - mv).T)
    S_W += class_sc_mat

print(S_W)
print(30 * '-')
# 类间散布矩阵
overall_mean = np.mean(iris_X, axis=0).reshape(4, 1)

S_B = np.zeros((4, 4))
for i, mean_vec in enumerate(mean_vectors):
    n = iris_X[iris_y == i, :].shape[0]
    mean_vec = mean_vec.reshape(4, 1)
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

print(S_B)

eig_vals, eig_vecs = np.linalg.eig(np.dot(np.linalg.inv(S_W), S_B))
eig_vecs = eig_vecs.real
eig_vals = eig_vals.real

for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:, i]
    print("Eigenvector {}: {}".format(i + 1, eigvec_sc))
    print("Eigenvalue {:}: {}".format(i + 1, eig_vals[i]))
print(30 * '-')
linear_discriminants = eig_vecs.T[:2]
print(linear_discriminants)

print(eig_vals / eig_vals.sum())

# LDA 投影数据s
lda_iris_projection = np.dot(iris_X, linear_discriminants.T)
plot(lda_iris_projection, iris_y, "LDA Projection", "LDA1", "LDA2")

# 实例化
lda = LinearDiscriminantAnalysis(n_components=2)

X_lda_iris = lda.fit_transform(iris_X, iris_y)
plot(X_lda_iris, iris_y, "Package LDA Projection", "LDA1", "LDA2")

# 创建有一个主成分的PCA模块
single_pca = PCA(n_components=1)

# 创建有一个判别式的LDA模块
single_lda = LinearDiscriminantAnalysis(n_components=1)

# 实例化KNN模型
knn = KNeighborsClassifier(n_neighbors=3)

# 不用特征转换 （需要击败的基准准确率）（交叉验证）
knn_average = cross_val_score(knn, iris_X, iris_y).mean()
print(knn_average)
print(30 * '-')
# 创建执行 LDA 的流水线
lda_pipeline = Pipeline([('lda', single_lda), ('knn', knn)])
lda_average = cross_val_score(lda_pipeline, iris_X, iris_y).mean()
print(lda_average)
print(30 * '-')

# 创建执行 PCA 的流水线
pca_pipeline = Pipeline([('pca', single_pca), ('knn', knn)])
pca_average = cross_val_score(pca_pipeline, iris_X, iris_y).mean()
print(pca_average)
print(30 * '-')

# 使用两个判别式的LDA模块
lda_pipeline = Pipeline([('lda', LinearDiscriminantAnalysis(n_components=2)), ('knn', knn)])
lda_average = cross_val_score(lda_pipeline, iris_X, iris_y).mean()
print(lda_average)
print(30 * '-')

# 尝试所有的 k 值，但是不包括全部保留
for k in [1, 2, 3]:
    # 构建流水线
    select_pipeline = Pipeline([('select', SelectKBest(k=k)), ('knn', knn)])
    # 交叉验证流水线
    select_average = cross_val_score(select_pipeline, iris_X, iris_y).mean()
    print(k, "best feature has accuracy:", select_average)


def get_best_model_and_accuracy(model, params, X, y):
    grid = GridSearchCV(model,  # 网格搜索模型
                        params,  # 试验的参数
                        error_score=0.)  # 如果出错，当作结果是0

    grid.fit(X, y)
    print("Best Accuracy: {}".format(grid.best_score_))
    print("Best Parameters: {}".format(grid.best_params_))
    print("Average Time to Fit (s): {} ".format(round(grid.cv_results_['mean_fit_time'].mean(), 3)))
    print("Average Time to Score (s): {}".format(round(grid.cv_results_['mean_score_time'].mean(), 3)))


iris_params = {
    'preprocessing__scale__with_std': [True, False],
    'preprocessing__scale__with_mean': [True, False],
    'preprocessing__pca__n_components': [1, 2, 3, 4],
    'preprocessing__lda__n_components': [1, 2],
    'clf__n_neighbors': range(1, 9)
}

preprocessing = Pipeline([('scale', StandardScaler()),
                          ('pca', PCA()),
                          ('lda', LinearDiscriminantAnalysis())])

iris_pipeline = Pipeline(steps=[('preprocessing', preprocessing),
                                ('clf', KNeighborsClassifier())])

get_best_model_and_accuracy(iris_pipeline, iris_params, iris_X, iris_y)
