# cian_prices
learn and predict flat prices

The data were downloaded from cian.ru. It includes the prices of new buildings in Moscow and Moscow region.

1. The dataframe was prepared, cleaned. The columns with more than 70% of missing values were dropped. 
The missing values were filled in. Some new feature were generated such as a time to the subway, on foot or
on a public transport, time in months. 

Also for the adressess the geocoding was applied. To make it faster the multiprocessing library was applied.
New feature distance was calculated from the latitude and the longitude.

2. The data were splitted on test, valid and test sets. The features were encoded with Target Encoding, 
One Hot Encoding and MinMax Scaler. Other methods such as Count encoding and Normalising didn't improve the result.

3. Some features with low variance were dropped with Variance Threshold method.

4. Several regression model were applied to the prepared dataframe. The following regression models have 
the best result: KNeighborsRegressor, Ridge, Lasso, LinearRegression, DecisionTreeRegressor, 
GradientBoostingRegressor, RandomForestRegressor 

5. Tune the parameters of the best model RandomForestRegressor
