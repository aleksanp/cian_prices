# cian_prices
learn and predict flat prices

The data were downloaded from cian.ru. It includes the prices of new buildings in Moscow and Moscow region.

1. The dataframe was prepared, cleaned. The columns with more than 70% of missing values were dropped. 
The missing values were filled in. Some new feature were generated such as a time to the subway, on foot or
on a public transport, time in months. 

2. Also for the adressess the geocoding were applied. To make it faster the multiprocessing library was applied.
New feature distance was calculated from the latitude and the longitude.

![alt text](https://raw.githubusercontent.com/aleksanp/cian_prices/main/Date%20-%20Price.png)

3. The data were splitted on test, valid and test sets. The features were encoded with Target Encoding, 
One Hot Encoding and MinMax Scaler. Other methods such as Count encoding and Normalising didn't improve the result.

4. Some features with low variance were dropped with Variance Threshold method.

5. Several regression model were applied to the prepared dataframe. The following regression models have 
the best result: KNeighborsRegressor, Ridge, Lasso, LinearRegression, DecisionTreeRegressor, 
GradientBoostingRegressor, RandomForestRegressor 

![alt text](https://raw.githubusercontent.com/aleksanp/cian_prices/main/Random%20Forest%20Regressor.png)

6. Tune the parameters of the best model RandomForestRegressor

