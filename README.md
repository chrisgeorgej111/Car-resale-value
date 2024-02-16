# Car-resale-value
Resale car value prediction

**Problem Statement:**

The primary objective of is project is to create a data science solution for predicting used car prices accurately by analyzing a diverse dataset including car model, no. of owners, age, mileage, fuel type, kilometers driven, features and location. The aim is to build a machine learning model that offers users to find current valuations for used cars.

**Data Understanding**

The Dataset contains multiple excel files, each represents its city, columns in each excel gives you an overview of each car, its details, specification and available features.

From this raw excel data,featues were extracted and cleaned the data, replaced the values of data, did typecasting..

After doing all these sort of preprocessing some catagorical features were encoded based on whether they had any inherent relationship or not..

Then several models were used to see the best of all.. Random Forest regressor gave the best output.. This gave the least mae and mse and better r2 score of 69%

Using streamlit an app has been developed that allows us to enter the necessary features and get the prediction of resale value.
