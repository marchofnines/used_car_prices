# WHAT DRIVES THE PRICE OF A CAR? 
[Link to used_car_prices.ipynb Jupyter Notebook](https://github.com/marchofnines/used_car_prices/blob/main/used_car_prices.ipynb)


## Business Question
- How can a car dearler fine tune their inventory to optimize for sales price? Specifically ... 
- What are the **5 top/key drivers** that will cause customers to want to pay more for the cars?  

## Data Cleaning
- Many cars are included many times at the same price in different markets.  We considered those duplicates and eliminated the region and state columns
- ID and VIN were dropped
- Extreme outliers for Price and odometer were explored using visualizations and dropoped.  This was done so as to get a better IQR value.  
- After extreme outliers were dropped, we computed lower and upper bounds to remove the rest of the outliers.  
- Dropped models for which there were only 2 cars or less in the dataset
- Dropped data that did not make sense to include:
    - Cars that have a condition of new even though they show a reading on the odometer.  
    - Cars that have a transmission type of 'other' because most cars in the dataset are known to have automatic transmission.  Also there were only 189 cars in this category anyway
- Year and Odometer were converted to int

## Data Understanding
- We noticed that the distribution of price was skewed so we examined the logarithm of the price and it seemed to have a more normal distribution which we used for modeling
- Heatmap Correlations
    - I temporarily converted cylinders to a numerical column to see how strong the correlation was
    - Highest correlations with price are:
        - year (+0.52)
        - odometer (-0.50)
    - Cylinders are weakly correlated with price (+0.22)
- Findings on cylinders through visualizations:
    - The majority of cars have 4,6 and 8 cylinders and exist at all price points
    - Higher percentage of 4 cylinder cars in general but especially for cheaper cars
    - As prices go up we see more 6 and 8 cylinder cars
- Findings on drive trains through visualizations:
    - Drive trains of all types exist at all prices
    - As price goes up we see a higher percentage of 4wd and rwd
- Findings on fuel types through visualizations:
    - Other than gas cars which exist at all price points, the highest percentage of expensive cars are diesel

## Modeling
### Feature encoding
#### Ordinal Encoding
- condition and size
#### One Hot Encoding
- cylinders, fuel, title_status, transmission, drive, type, paint_color
#### James Stein and Binary Encoding
- Due to high cardinality, we tried **both** James Stein and Binary Encoding for model and manufacturer

### Model Creation
Our goal was to end up with a model using only 5 interpretable features so that we can provide clear recommendations to our client

#### Approach for kfold Cross-Validation
- I had created a hyperparameter search function during a previous activity which I reused here to do my GridSearches with k-fold Cross-Validation. Here are some details about the function:
    - It allows for flexible creation of transformers, pipelines and GridSearchCV
    - It runs the same GridSearchCV for different values of cv (k=5, 10, etc)
    - It compares the best_score_ attribute from the GridSearchCV.  I used **neg_mean_squared_error** because I wanted to emphasize large errors.  There was no sense in using RMSE since we had thoroughly handled outliers. 
    - Output a dataframe containing a condensed version of grid's cv_results
    - Output a graph showing how the different models scored against each otehr
    - Output a dataframe containing the coefficients for the selected features
    - Ran an initial holdout validation 


- GridSearchCV1, k=5, 10
    - Transformer1 (uses JSE for model and manufacturer)
    - SequentialFeatureSelection
    - StandardScaler
    - TransformedTargetRegressor with LinearRegression
- GridSearchCV2, k=5, 10
    - Transformer2 (uses Binary Encoding for model and manufacturer)
    - SequentialFeatureSelection
    - StandardScaler
    - TransformedTargetRegressor with LinearRegression
    - **Since GridSearchCV2 did better, we used Transformer2 for the remaining GridSearches**
    - **Since k=10 did better, we used k=10 for the remaining GridSearches**
- GridSearchCV3, k=10
    - PolynomialFeatures - degrees: 1 & 2
    - StandardScaler
    - RFE with LinearRegression - n_features_to_select: 6, 7, 8
    - TransformedTargetRegressor with LinearRegression
- GridSearchCV4, k=10
    - PolynomialFeatures - degrees: 1 & 2
    - StandardScaler
    - Lasso Selector 
    - TransformedTargetRegressor with LinearRegression
- GridSearchCV5, k=10
    - PolynomialFeatures - degrees: 1 & 2
    - StandardScaler
    - SequentialFeatureSelection
    - TransformedTargetRegressor with LinearRegression
    - **Even though the Lasso Selector had a better MSE, it came back with way too many features. We cannot use this to advise our dealer so we will use SequentialFeatureSelector going forward**
    - **This GridSearch preferred a model with 8 features, but the score for the one using only 5 features wasn't much lower. We selected the model rank 3 which which uses 5 features and polynomial degree = 1 as the one to beat.**
- GridSearchCV6, k=10
    - PolynomialFeatures - degrees: 1
    - StandardScaler
    - SequentialFeatureSelector -  n_features_to_select: 4, 5
    - TransformedTargetRegressor with HuberRegressor - epsilon 1, 1.35 and alpha 0.0001 and 0.001
- GridSearchCV7 k =10
    - PolynomialFeatures - degrees: 1
    - StandardScaler
    - SequentialFeatureSelector -  n_features_to_select: 4, 5
    - TransformedTargetRegressor with Ridge - alpha 0.0001, 0.001, 0.1, 1, 10
- GridSearchCV8 k =10
    - PolynomialFeatures - degrees: 1,2
    - StandardScaler
    - SequentialFeatureSelector/Lasso -  n_features_to_select: 3,4,5,8
    - TransformedTargetRegressor with LinearRegression
- GridSearchCV9 k =10
    - PolynomialFeatures - degrees: 1,2,3
    - StandardScaler
    - SequentialFeatureSelector -  n_features_to_select: 4, 5
    - TransformedTargetRegressor with Lasso - alpha 0.001, 0.1, 1


### Evaluation

#### Holdout validation on top 2 models 
Performed a holdout validation for the top 2 models

    - Poly Degree 1, 5 features, HuberRegressor
        - Train MSE: 0.31, Test MSE: 0.30

    - Poly Degree 1, 5 features, LinearRegression
        - Train MSE: 0.30, Test MSE: 0.29

The model that scored second best in the Cross-Validations, actually scored #1 during hold-out validation.  So I picked it.  

#### Permutation Importance
- We note that for most score types, the rank was:
    - year
    - drive
    - odometer
    - cylinders
    - fuel

#### Coefficient Interpretation
Note that all interpretations assume that all other factors remain constant - which in real life they are not

- Cars with 8 cylinders tend to be 44.68% more expensive than all other types of cars commbined
- Diesel cars tend to be 106% more expensive than all other types cars combined 
- fwd cars then to be 34.58% less expensive than all other types of cars combined
- For a car that is newer by 1 year, the price goes up by 7.1%
- For every mile driven, the car depreciates by 0.0004%.  Stated differently for every 1000 miles driven, the car depreciates by 0.4%.

#### Plotted Residuals

### Deployment
#### Advice to Dealerships
- In order to increase used car prices, our top 5 recommendations are that you prioritize cars that: 
   - run on Diesel
   - have 8 cyinder cars (and secondarily 6 cylinder cars based on the visuasations we saw)
   - are 4wd and rwd cars
   - are newer
   - have fewer miles driven

#### Next Steps
 - Reduce usage of categories of 'other' or 'missing' by reclassifying them properly when possible
 - Correct bad data such as zero USD pricing or odometer readings in the billions
 - Fill in the data where there are missing values 
 - Rerun the model 
 - Clearly note the rank of drive trains, number of cylinders and fuel types for their impact on price 
 - Evaluate the importance of the other features 
 - Update this document as new findings are made
