# BIG MART SALES PREDICTIONS
The code predicts the sales of bigmart
# Requirements
- Have tensorflow installed
- Have python installed 
# Libraries used
- numpy and pandas are used for data manipulation and analysis.
- seaborn and matplotlib.pyplot are used for data visualization.
- LabelEncoder is used from sklearn.preprocessing to encode categorical features as integers.
- train_test_split is used from sklearn.model_selection to split data into training and testing sets.
- LinearRegression is used from sklearn.linear_model to create a linear regression model.
- r2_score and mean_squared_error are used from sklearn.metrics to evaluate the model's performance.
- train_test_split is again used from sklearn.model_selection to split data into training and testing sets.
- RandomForestRegressor from sklearn.ensemble is for constructing a multitude of decision trees 
- VotingRegressor is an ensemble learning method in scikit-learn that combines multiple regression models to improve the overall prediction accuracy.
- Tensorflow is used to build a deep neural network
# The dataset
The dataset is divided into 2 the train dataset and the test dataset
# models used
 3 models have been used
- Linear regression model - This model has been divided into 2. The first linear regression model is a stand alone model. The second is a combination of linear regression models to get the best score.
- Random forest model
- A deep neural network
# working on the dataset
The train dataset has 8523 rows and the test dataset has 5681 rows
The train and test datasets are joined to perform feature engineering.
A column called outlet age was calculated and added to the dataset

**Filling missing values**
- The missing values in the outlet_size column were filled with the mode of the column
- The missing values in the Item_Weight column were filled with the mean of the column
- The missing values in the item_vis column were replaced with  'low_viz'.

The outliers were removed.
A column Item_type_combined was created. It contains values mapped from the Item_Identifier column.
Dummy variables of 'Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type', 'Item_type_combined' columns were created.
The distribution of the target variable Item_Outlet_Sales was first checked to see its distribution in order to know what to use to replace the missing variable in it. After filling the null values in the Item_Outlet_Sales column, the dataset was then divided to its original state.
# Model building
The y variable was defined
The train dataset was the divided into the train set and the validation set
### Linear regression model
- LinearRegression() class is used to define a linear regression model. The linear regression model is fit on the training set
- The intercept and the coefficients of the model were displayed
- The RMSE of the model is 1127.618
- The R-Squared of the model is 0.563
### Random Forest Regressor Model
- RandomForestRegressor() model is fitted on the dataset to create random forests.
- The R_Squared of the model is 0.553
### Voting Regressor
- Voting regressor was used to combine three linear regression models.
- The R-Squared of the model is 0.563

### Deep Neural Network
- A deep neural network is created using Tensorflow and keras
- The model is a sequential model.
- It has 4 layers.
- The first layer is an input layer. It has 32 neurons and uses relu activation function
- The second layer has 64 neurons and uses relu activation function
- The third layer contains 128 neurons and uses a relu activation function
- The fourth layer is the output layer. It contains only one neuron and a relu activation function.

**Compiling the model**

The sequential model has been compiled using an adam optimizer with a learning rate of 0.01. It uses MAE as the measuring metrics
The model has been fitted and trained using 10 epochs and the dataset has been divided into batches of 50
The neural network has a R-Squared of 0.430
So far this is the poorest performing model.
# OVERVIEW
The voting regressor and the linear regression model have the same r-squared.
So far these are the best performing model.





