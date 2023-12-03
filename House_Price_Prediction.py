import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def House_price_prediction():
    # Reading the House prices From csv file
    House_data_set = pd.read_csv(r"C:\Users\DELL\Desktop\Datasets\1553768847-housing.csv")
    # print(House_data_set)

    # ------Data Preprocessing or Data cleaning ----------

    # Statstical data measure using discribe method
    # print(House_data_set.describe())

    # getting the only unique values from the ocean_proximity
    # print(House_data_set["ocean_proximity"].unique())

    # Filling the NUll values in the Total_bedrooms column
    House_data_set.total_bedrooms.fillna(House_data_set["total_bedrooms"].value_counts().index[0], inplace=True)
    # print(House_data_set["ocean_proximity"].unique())

    # it will prints whether the columns has NULL values or not
    # print(House_data_set.isnull().sum())

    # Getting Info About dataset returns information regarding dataset example :data types and null values
    # print(House_data_set.info())

    # ocean_proximity column is Object type so it should to convert into numerical because machine learning algorithms cannot perform operations on Object datatype
    # datatype convertion using label encodings
    House_data_set["ocean_proximity"] = House_data_set["ocean_proximity"].astype("category")

    House_data_set["Ocean_proximity"] = House_data_set["ocean_proximity"].cat.codes

    # dropping the categorical ocean_proximity column
    House_data_set.drop(inplace=True, columns=["ocean_proximity"], axis=8)

    # print(House_data_set.info())
    # print(House_data_set.head())
    # print("unique",House_data_set["Ocean_proximity"].unique())
    # print(House_data_set.columns)
    # print(House_data_set.head())

    House_data_set.rename(columns={"median_house_value": "Price"}, inplace=True)
    # print(House_data_set.columns)

    # --------------Traning the model-------------

    Linear_Houser_price = LinearRegression()

    Target = House_data_set["Price"]
    House_data_set.drop(inplace=True, columns=["Price"])
    # House_data_set.drop(inplace=True,columns=["Index"])
    House_data_set.reset_index(drop=True, inplace=True)
    # print(House_data_set.columns)
    # print(Target)
    # print(House_data_set.head(20))

    Linear_Houser_price.fit(House_data_set, Target)
    longitude=float(input("Enter Longitude of House :"))
    latitude=float(input("Enter Latitude of House :"))
    house_age=float(input("Enter House median age :"))
    rooms=int(input("Enter number of rooms :"))
    bed_rooms=int(input("Enter number of bedrooms :"))
    population=float(input("Enter the population House area :"))
    house_holds=float(input("Enter number of Households :"))
    median_income=float(input("Enter the House median income :"))
    ocean_pr=int(input("Enter ocean_proximity : 0.<1H OCEAN, 1.INLAND, 2.NEAR OCEAN, 3.ISLAND :"))
    result = Linear_Houser_price.predict([[longitude, latitude,house_age, rooms, bed_rooms, population, house_holds, median_income,  ocean_pr]])
    print(f"House price is ={result}rs/-")
    # print("score is :",Linear_Houser_price.score(House_data_set,Target)

House_price_prediction()






