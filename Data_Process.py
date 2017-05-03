import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

class Data_Process():

    # get the features and labels of the dataset separately
    def get_data(self, Path='Cancer_Data.csv'):
        # load the breast_cancer dataset from csv
        BC_Data = pd.read_csv(Path, skiprows=1, header=None)
        # the second column is the types of Cancer (categorical--M/B)
        BC_Type = BC_Data[1].unique()
        # replace 'M' with 0, replace 'B' with 1
        for i, type in enumerate(BC_Type):
            # DataFrame.set_value(index, col, value, takeable=False)
            BC_Data.set_value(BC_Data[1] == type, 1, i)
        # split the features and labels
        # numpy.split(ary, indices_or_sections, axis=0)
        Y, X = np.split(BC_Data.values, (2,), axis=1)
        # set the features to float, set the labels to int
        X = X.astype(np.float)
        Y = Y.astype(np.int)
        # drop the 'id' column, since it is useless for analyzing
        Y=Y[:, 1]
        # return the features X and labels Y
        return X, Y

    # normalize the data by (x-mean)/std
    def data_normalization(self, X):
        self.X_=X
        X_col_mean = []
        X_col_std = []
        for i in range(0, X.shape[1]):
            mean = np.mean(X[:, i])
            X_col_mean.append(mean)
            std = np.std(X[:, i])
            X_col_std.append(std)

        for i in range(0, X.shape[1]):
            X[:, i] = (X[:, i] - X_col_mean[i]) / X_col_std[i]
        return X

    # split the data into training and test sets, 25% of the data is the test dataset
    def data_split(self, X, Y):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
        return X_train, X_test, y_train, y_test

    # perform PCA on data, return the transformed data and the variance ratio
    def data_PCA(self, X):
        pca=PCA(X.shape[1])
        pca.fit(X)
        ratio=pca.explained_variance_ratio_
        cum_ratio=[]
        cum=0
        for n in ratio:
            cum=n+cum
            cum_ratio.append(cum)
        X_Trans=pca.transform(X)

        return X_Trans, ratio, cum_ratio

    # feature extraction, n=number of principal components to be extracted
    def feature_extract(self, X, n):
        X_Trans, ratio, cum_ratio = Data_Process().data_PCA(X)
        X_extract = X_Trans[:, :n]
        return X_extract

    # replace 0 with 1, and replace 1 with 0, for comparing the true labels and the clustered labels in clustering algorithm
    def data_reverve(self, pred):
        predict = []
        if 1.0 * np.sum(pred) / len(pred) < 0.5:
            for i, label in enumerate(pred):
                if pred[i] == 0:
                    n = 1
                else:
                    n = 0
                predict.append(n)
        predict = np.array(predict)
        return predict