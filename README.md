# Prediction of real estate property prices

The model was made to predict the housing prices based on the data from the real estate website cian.ru that covers Moscow and Moscow Oblast.

1. The dataframe was prepared, and cleaned. The columns with more than 70% of missing values were dropped. 
The missing values were filled in. Some new features were generated, including time to reach the subway, on foot or
by public transport, time in months. 

![alt text](https://raw.githubusercontent.com/aleksanp/cian_prices/main/pictures/Region%20copy.png)

2. The geocoding was applied for the adressess. The multiprocessing library was applied to increase the speed. New feature distance was calculated from the latitude and the longitude.

![alt text](https://github.com/aleksanp/cian_prices/blob/main/data/pictures/map_2.png?raw=true)

3. The data were splitted on test, valid and test sets. The features were encoded with Target Encoding, 
One Hot Encoding and MinMax Scaler. Other methods such as Count encoding and Normalising didn't improve the result.

4. Some features with low variance were dropped with Variance Threshold method.

5. Several regression models were applied to the prepared dataframe. The following regression models have 
the best result: KNeighborsRegressor, Ridge, Lasso, LinearRegression, DecisionTreeRegressor, 
GradientBoostingRegressor, RandomForestRegressor 

6. The parameters of the best model were tuned.

Valid set

![alt VALID](https://raw.githubusercontent.com/aleksanp/cian_prices/main/pictures/Random%20Forest%20Regressor%20valid.png)

Test set 

![alt TEST](https://raw.githubusercontent.com/aleksanp/cian_prices/main/pictures/Random%20Forest%20Regressor%20test.png)



