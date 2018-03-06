####################################################################
# Machine Learning - MIRI Master
# Lluís A. Belanche

# LAB 2: Linear regression and beyond (Part2)
# version of March 2018
####################################################################

####################################################################
## Exercise 2: Daily air quality measurements in New York, May to September 1973

summary(airquality)

# for information on the dataset
?airquality

# as you can see, there are several missing values ... you should fix this first

# The goal is to develop a predictive model of Ozone concentration given solar radiation, wind and temperature

## Do the following (in separate sections)

# 1. Fix the missing values; most are located in the target variable itself (Ozone); since this is a time series,
#    you can either remove the rows or replace the missing values by interpolation; then deal with Solar.R (knn)
# 2. Do any other pre-processing or visual inspection of the dataset that you think is adequate.
#    Split the data into training (2/3) and test (1/3); WARNING: do this randomly because the dataset
#    has an structure (days are consecutive, etc) or shuffle the data before partitioning
# 3. Fit a lm() model to predict 'Ozone' with solar radiation, wind and temperature in the training set
# 4. Inspect the results, residuals, and compute the predictive normalized root MSE
# 5. Try a second model by log-transforming the Ozone; is this an improvement? if it is, keep it
# 6. Try a third model that additionally includes 'Day'; is this an improvement? if it is, keep it
# 7. Try a last model that includes a second order polynomial on Wind, like this:
#        lm(log(Ozone) ~ Solar.R + Temp + poly(Wind,2), data = airquality)
#    is this an improvement? if it is, keep it
# 8. Do another modelling using ridge regression, and decide the best lambda using GCV
# 9. Do another modelling using LASSO regression
#10. Decide which is your best model and give a final prediction in the test set


# Your code starts here ...
