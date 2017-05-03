from Data_Process import Data_Process
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

class Data_Visualization():

    # to visualize the PCA variance ratio bar
    def pca_vis(self, X):
        _, ratio, cum_ratio=Data_Process().data_PCA(X)
        plt.bar(np.arange(1, X.shape[1]+1), ratio, 1)
        plt.plot(np.arange(1, X.shape[1]+1), cum_ratio, 'r-')
        plt.xlabel("n-component")
        plt.ylabel('variance ratio')
        plt.title('Principal Components variance ratio')
        plt.show()

    # to visualize the classification after PCA treatment (2 principal components visualization)
    def classification_vis(self, model, i, X_train, X_test, y_train, y_test):
        name=['Logistic Regression', 'Decision Tree Classification', 'SVC', 'Random Forest Classification']
        x1_min, x1_max = X_test[:, 0].min(), X_test[:, 0].max()  # range of first column (PC1)
        x2_min, x2_max = X_test[:, 1].min(), X_test[:, 1].max()  # range of second column (PC2)
        t1 = np.linspace(x1_min, x1_max)
        t2 = np.linspace(x2_min, x2_max)
        x1, x2 = np.meshgrid(t1, t2)  # generate meshgrid
        xx_test = np.stack((x1.flat, x2.flat), axis=1)

        light_color = mpl.colors.ListedColormap(['#77E0A0', '#FF8080'])  # for coloring the classified mesh
        dark_color = mpl.colors.ListedColormap(['g', 'r'])  # for coloring the data point
        yy_test_pred = model.predict(xx_test)   # prediction
        yy_test_pred = yy_test_pred.reshape(x1.shape)  # make the shape compatible with the mesh
        plt.figure(facecolor='w')  # white background
        plt.pcolormesh(x1, x2, yy_test_pred, cmap=light_color)   # coloring the classified mesh
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', s=120, cmap=dark_color,
                    marker='*')  # plot test dataset
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train.ravel(), edgecolors='k', s=40,
                    cmap=dark_color)  # plot training dataset
        plt.xlabel('PC1', fontsize=12)
        plt.ylabel('PC2', fontsize=12)
        plt.title(name[i])    # select the proper algorithm for the title
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.grid()
        plt.show()

    # to visualize feature importance in decision tree and random forest
    def feature_importance_vis(self, model):
        feature_importance_list=model.feature_importances_
        plt.bar(np.arange(1, 31), feature_importance_list, 1)
        plt.xlabel('features')
        plt.ylabel('feature importance')
        plt.title('feature importance presentation')
        plt.show()

    # to visualize the scatter plot of the data
    def scatter_vis(self, X):
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1])
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('data distribution after PCA')
        plt.show()

    # to visualize the scatter plot of the data in different color
    def scatter_vis2(self, X1, X2, n):
        name=['Actual Groups after PCA', 'Predictive Clustering after PCA']
        plt.figure()
        plt.scatter(X1[:, 0], X1[:, 1], color='r', marker='^')    # red stands for 'M'
        plt.scatter(X2[:, 0], X2[:, 1], color='b', marker='s')    # blue stands for 'B'
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(name[n])
        plt.show()

    # to visualize the comparison of the centers of clusters
    def center_comp_vis(self, cluster_center):
        plt.bar(np.arange(1, len(cluster_center[0]) + 1), cluster_center[0], 0.5, color='b')
        plt.bar(np.arange(1.5, len(cluster_center[1]) + 1), cluster_center[1], 0.5, color='r')
        plt.xlabel('features')
        plt.ylabel('feature centers')
        plt.title('feature centers presentation')
        plt.show()