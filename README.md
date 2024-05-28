# Linear-Regression-Car-Price-Prediction-and-Data-Analysis
Problem Statement
Consider there’s a client that specializes in trading used cars across different states in the US. As a Data Scientist, you are given the task of creating an automated system that predicts the selling price of cars based on various features (information) such as the car’s model name, manufacture year, the current price when bought new, kilometers driven, fuels type and owners it had.

The price estimation system will be used to set a competitive selling price for the cars in the used car market, also it will gain trust from customers, by providing detailed explanations for the predicted selling price outputted by your system.

# Introduction
Linear regression is a foundational statistical technique in data science, offering a window into understanding relationships between variables. In this blog, we’ll dive into a hands-on project where we apply linear regression to a real-world dataset. The goal is to demystify the process and showcase the practical application of this method in deriving meaningful insights from data.

# Setting Up the Environment
Before delving into the data, it’s crucial to set up our environment with the right tools. We used a variety of Python libraries, each serving a specific purpose:

NumPy: Essential for numerical operations.
Pandas: Perfect for data manipulation and analysis.
Statsmodels: Provides classes and functions for the estimation of statistical models.
Matplotlib and Seaborn: Our go-to libraries for data visualization.
Scikit-learn: A comprehensive library for machine learning, including linear regression models.

![image](https://github.com/SLJyothi/Linear-Regression-Car-Price-Prediction-and-Data-Analysis/assets/164232591/c0f69bfa-a369-4782-b651-16824f6687b5)

Each of these libraries has a specific role, from data manipulation (Pandas, NumPy) to visualization (Matplotlib, Seaborn) and statistical modeling (Statsmodels, Scikit-learn).

3 Data Exploration
The first step in our analysis was to load and explore the dataset. The following code snippet shows how we read the data and took a peek at the first few rows:
![image](https://github.com/SLJyothi/Linear-Regression-Car-Price-Prediction-and-Data-Analysis/assets/164232591/d6459b8f-ca91-46d6-bab7-ae7a8e7c9d3c)

![image](https://github.com/SLJyothi/Linear-Regression-Car-Price-Prediction-and-Data-Analysis/assets/164232591/a8141c1c-3231-4dad-8013-eac8a41d4c53)
The dataset contains 301 rows and 9 columns. Each row in the dataset contains information about one car. The task is to find a way to estimate the value in the “Selling_Price” column using the values in the other columns. If we can do this estimation for historical data, then we should be able to estimate selling_price for new cars that are not in this data too, simply by providing information like car name, year, present price, kilometers driven, fuel type, seller type, transmission and owner.

# Checking the data type of each column
![image](https://github.com/SLJyothi/Linear-Regression-Car-Price-Prediction-and-Data-Analysis/assets/164232591/ada10f7a-5810-4069-9866-27cc96a16cca)

We could see that Year, Selling Price, Present Price, and Kms Driven are numeric, whereas Fuel Type, Seller Type, Transmission, Owner, and Car Name are objects( string ) possibly categorical columns.

# Exploring statistics for the numerical columns:
![image](https://github.com/SLJyothi/Linear-Regression-Car-Price-Prediction-and-Data-Analysis/assets/164232591/1f2423a1-ced1-4595-bbba-8f25a5faeaeb)

The numerical columns are looking reasonable, there’s no anomaly in any of the columns. Just as a side info, looks like the selling price and present prices are encoding of real car prices. But that won’t affect our analysis since the values can always get translated to their real values from the data provider.

# Analysis of Numerical Features
# Year

The year column is a numerical column, as we can see the max value is 2018, and the min is 2003, we can visualize using a histogram, with 16 bins (one bin for each year). So below we could see the number of cars available (count) for each year.

![image](https://github.com/SLJyothi/Linear-Regression-Car-Price-Prediction-and-Data-Analysis/assets/164232591/41f0a7d1-a8d7-46ed-bfc1-3985f051d23f)

The distribution is skewed and has some outliers that could be handled. But since we don’t have too many examples (rows) in our data we could also ignore it and leave them as they are.

# Selling Price
![image](https://github.com/SLJyothi/Linear-Regression-Car-Price-Prediction-and-Data-Analysis/assets/164232591/63702d6f-2efc-4024-91ef-14c1d36e39ea)

The distribution is skewed and we also can observe that automatic cars are more expensive. As the selling price increases, we could see this in ur box plot above.

# Kilometers Driven
Let’s visualize the Selling Price, and also include the categorical column Transmission to distinguish selling prices for Manual cars and Automatic cars
![image](https://github.com/SLJyothi/Linear-Regression-Car-Price-Prediction-and-Data-Analysis/assets/164232591/9aa630fc-837f-420d-8085-8910c03511ac)

As we can see from the plot above, most of the numerical values range from 0 to 100k kms, and there are some outliers.

Another intuitive hypothesis that we could make is that the more kilometers a car has, the lower its price will be. Let’s see if this is true. We can use a scatter plot to visualize the relationship between two numerical features.

![image](https://github.com/SLJyothi/Linear-Regression-Car-Price-Prediction-and-Data-Analysis/assets/164232591/635991fe-9324-46d7-b5a5-ae6f50ec3445)

We can’t say that from this plot above we can’t conclude that the more kilometers the cheaper the prices. But this can also be because of the distribution of Kms since most of the values are in 0–100k.

Let’s check if Selling_Price and Present_Price are correlated.

![image](https://github.com/SLJyothi/Linear-Regression-Car-Price-Prediction-and-Data-Analysis/assets/164232591/76aa5973-1bf4-47ad-8db5-e58075be5837)

As we can see, the greater the present price the greater the selling price, this is also intuitive because the cars that are expensive, will probably also be sold at higher prices.
![image](https://github.com/SLJyothi/Linear-Regression-Car-Price-Prediction-and-Data-Analysis/assets/164232591/e328ac93-7161-49f6-8794-43fd2d4ae463)

Model
The goal will be to fit a line using these points in the 2D plot above. Where x represents our feature (Present Price) and our goal would be to predict y (Selling Price). A line in a 2D (X and Y) coordinates has the formula:

y = w * x + b

If we would substitute our feature and target in the formula above, it would look like this:

selling_price = w * present_price + b

The goal would be to find the values of w and b parameters which would result in a line that best fits the data (as a result our predictions will be correct, we would fit any present_price value and it will tell us the selling price of it).

This method we’re using is called linear regression, where the equation above is known as the linear regression model. Because it formulates the correlation between “Present Price” and “Selling Price” in a shape of a straight line. The ‘w’ and ‘b’ are usually known as model parameters or their weights.

In this dataset, the values that are under the column (feature) “Present_Price” are considered as the model’s input, while the values in the “Selling Price” column are known as “targets”.

# Correlation
As we can observe from the analysis, some columns are more closely related to the selling price, compared to the others. For example “Year” gets larger, and so does the Selling_Price. While Kilometers driven and selling prices do not grow together.

This relationship can be numerically expressed using a measure called correlation coefficient, which can be computed using the .corr method from the pandas' library.

# To Compute the correlation coefficient of selling_price and Year:
![image](https://github.com/SLJyothi/Linear-Regression-Car-Price-Prediction-and-Data-Analysis/assets/164232591/f1ccd661-4876-4269-ad8a-cea7e483800b)

We could observe from the values above, that there’s a high correlation between present_price and selling_price but less correlation between kilometers driven and selling price.

To interpret correlation coefficients:

Strength: The values can range between -1 and 1 indicating a perfectly linear relationship where a change in one variable is followed by a perfectly consistent change to the other. In practice, you usually won’t see this kind of relationship.
A zero value of the coefficient represents no linear relationship.
