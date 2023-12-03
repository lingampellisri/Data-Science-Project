from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pandas as pd
import numpy as np

iris_data=pd.read_csv(r"C:\Users\DELL\Desktop\Datasets\IRIS.csv")
# print(iris_data.head(10))
# print(iris_data["species"].unique())
def Flower_species_prediction():
    # label encoding the categorical data into numerical data
    iris_data["species"] = iris_data["species"].astype('category')
    Target = iris_data["species"].cat.codes
    # print(iris_data.head(20))
    iris_data.drop(columns=["species"], inplace=True)
    # 0->Iris-setosa
    # 1->Iris-versicolor
    # 2->Iris-virginica
    flowers = {
        0: "setosa",
        1: "versicolor",
        2: "virginica"
    }
    # print(iris_data.head(20))

    # print(iris_data["Species"].unique())
    # print(iris_data.columns)
    # print(Target)

    classifier_knn = KNeighborsClassifier()
    classifier_knn.fit(iris_data, Target)
    # print(iris_data.columns)
    sepal_len = float(input("Enter sepal length :"))
    sepal_width = float(input("Enter sepal width :"))
    petal_len = float(input("Enter petal length :"))
    petal_width = float(input("Enter petal width :"))
    test = [[sepal_len, sepal_width, petal_len, petal_width]]
    # print(classifier_knn.predict(test))

    re = classifier_knn.predict(test)
    # print("accuracy is :",classifier_knn.score(iris_data,Target))
    # print("---------------result-----------------")
    print(f'Flower belongs to "{flowers[re[0]]}" Species')

Flower_species_prediction()

