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

