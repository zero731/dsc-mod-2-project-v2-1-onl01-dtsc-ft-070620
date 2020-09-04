# Mod 2 Final Project - Selling a House in King County: 
# Predicting and Maximizing Sale Price with Multiple Linear Regression

#### Author: Max Steele

The goal of this project was to build a multiple linear regression model to predict house prices in King County, Washington as accurately as possible. The main purpose of this model is to help homeowners understand the likely price range for their home, and what actions they can take (or avoid) to maximize their selling price.

## Data
The data used were obtained from the `'kc_house_data.csv'`. This dataset contains the following columns/information:
* **id** - unique identifier for a house
* **dateDate** - house was sold
* **pricePrice** - prediction target
* **bedroomsNumber** - of Bedrooms/House
* **bathroomsNumber** -  of bathrooms/bedrooms
* **sqft_livingsquare** - footage of the home
* **sqft_lotsquare** - footage of the lot
* **floorsTotal** - floors (levels) in house
* **waterfront** - house which has a view to a waterfront
* **view** - has been viewed
* **condition** - how good the condition is ( Overall )
* **grade** - overall grade given to the housing unit, based on King County grading system
* **sqft_above** - square footage of house apart from basement
* **sqft_basement** - square footage of the basement
* **yr_built** - built year
* **yr_renovated** - year when house was renovated
* **zipcode** - zip (postal code)
* **lat** - latitude coordinate
* **long** - longitude coordinate
* **sqft_living15** - the square footage of interior housing living space for the nearest 15 neighbors
* **sqft_lot15** - the square footage of the land lots of the nearest 15 neighbors


## Methods
I followed the OSEMN data science process to approach this problem. Initial exploration and scrubbing of the dataset identified and dealt with null or otherwise missing values. 

### Missing Values and Feature Engineering
The only columns found to contain missing values were `view`, `waterfront`, `yr_renovated`, and `sqft_basement`. I chose to drop `view` because the meaning of that column was unclear. The `waterfront` column was missing 11% of its values, and `yr_renovated` was missing 17.8% of its values. Both variables were already extremelt zero-inflated and replacing the missing values with 0 did not appear to dramatically alter the distributions of the two variables. The `basement` variable had placeholder values for 2.1% of the column, and to address this I created a new feature `est_sqft_basement` by subtracting `sqft_above` from `sqft_living`. The resulting feature matched with the original `sqft_basement` variable very well and did not have any missing values.

I also engineered additional features from the available information which I thought might be useful predictors of price. These included `month_sold`, `season_sold`, and `yr_sold` created from the original `date` column. I calculated the `age` of the house at the time it was sold by subtracting `yr_built` from `yr_sold`. This would have made more of a difference if the data had been collected over a longer period of time than just May 2014 to May 2015, but the use of `age` rather than `yr_built` seemed like it would aid in communicating results to homeowners. It would be easy to forget during a presentation or while looking at visual representations of the data that we're comparing `yr_built` to 5 - 6 years ago and not the present.

### Assumption of Independence
When examing the `.value_counts()` for `id`, I discovered that 175 of the houses had been sold twice and 1 house had been sold 3 times during the time period over which the data were collected. Since the ordinary least squares (OLS) method of multiple linear regression assumes that individual observations are independent, I dropped all but the most recent record of sale for these duplicate houses.

### Selecting Possible Predictors of Price
To focus on the best predictors of house price to include in the multiple linear regression models, I first investigated which variables were most clearly linearly associated with `price`. 
I found that `sqft_living`, `sqft_above`, `bathrooms`, and `grade` showed the clearest linear relationship with `price`:

<img src="Figures/Variables Linearly Related with Price.png" width = 1000 halign=center>


I also found that `month_sold` (and `season_sold`), `yr_sold`, `sqft_lot` (and `sqft_lot15`), and `long` showed no linear relationship with price and decided to drop these variables:

<img src="Figures/Variables Unrelated with Price.png" width = 1000 halign=center>


I also determined how each feature should be treated -- whether as a continuous, ordinal or nominal categorical variable -- if it were to be included in the models. This required an iterative process of data scrubbing and exploring until the data were ready for inclusion in initial models. 

To avoid problems associated with multicollinearity when building models, I did not include variables with correlation coefficients of 0.75 or greater. I chose one and dropped the other, and tested multiple combinations of variables in this way.

Based on examining the distribution of each variable, it's relationship with price, and its correlation with other variables, I determined that the following variables would be included in initial models, as the following types:

- **Continuous**
 - `sqft_living`
 - `sqft_above` - (include when `sqft_living` dropped from model)
 - `sqft_living15` - (include when `sqft_living` dropped from model)
 - `est_sqft_basement`
 - `age`
 - `lat`
 - `grade` -(include in final model when outliers have been dropped and correlation with `sqft_living` is not as strong)
 
 
- **Ordinal Categorical**
 - `bedrooms`
 - `bathrooms` - (include when `sqft_living` dropped from model)
 - `floors`
 

- **Categorical**
 - `waterfront`
 - `condition`
 - `renovated`
 - `zipcode`


### Outlier Removal
To maximize the predictive accuracy of my final model for houses representative of the majority within the county, I examined how 2 methods of outlier removal (based on z-scores or on IQR) impacted model performance and model residuals. While doing so, I also examined how the different methods of outlier removal impacted the range of prices and house types that the resulting model would be able to directly address. 

According to the z-score method, observations were considered outliers and removed if the absolute value of their z-score was 3 or greater (3 or more standard deviations above or below the mean). Based on the IQR method, observations were considered outliers and removed if the absolute value of they fell more than 1.5 times outside the interquartile range (IQR = Q3 - Q1).

Removing outliers based on the IQR method removed nearly 3 times the number of observations compared to the z-score method (1152 observations compared to 404). Removing price outliers reduced the skew of this variable, making the distributions appear closer to normal (the IQR cleaned data more so than the z-score cleaned data). The minimum price did not change across the datasets, but the maximum price dropped substantially from $7.7 million to $1.64 million when z-score outliers were removed, and then slightly again to $1.12 million.


<img src="Figures/Comparison of Living Square Footage Distributions for Different Methods of Outlier Removal.png" width = 1200 halign=center>

Other than price, the range of most variables was relatively unchanged. The minimum value was unchanged for all variables.

The maximum number of bathrooms dropped from 8 in the full dataset to 7.5 in both cleaned datasets.
The maximum sqft_living dropped from 13,540 to 7,480 in both cleaned datasets.
Maximum grade dropped from 13 to 12 in both cleaned datasets.
Maximum sqft_above dropped from 9,410 in the full dataset to 7,320 in the z-score cleaned data, and 5,710 in the IQR cleaned data.
Maximum sqft_living15 dropped from 6,210 in the full dataset to 5,790 in the z-score cleaned data, and 5,380 in the IQR cleaned data.
Maximum est_sqft_basement dropped from 4,820 in the full dataset to 2,850 in the z-score cleaned data, and 2,720 in the IQR cleaned data.

<img src="Figures/Comparison of Price Distributions for Different Methods of Outlier Removal.png" width = 1200 halign=center>

### Modeling
Models were built by fitting the data using the `statsmodels` `ols` method. The same two initial models were tested on each of the 3 datasets: the full dataset, the z-score cleaned data, and the IQR-cleaned data. None of the features were transformed or scaled. The final model was fit to the IQR-cleaned dataset, and no transformations or scaling was deemed necessary.

When tuning models, predictors were dropped from the model if the p-value exceeded alpha=0.05. For categorical variables, if some of the categories were non-significant, but the majority were significant predictors, that variable was retained in the model.

Model quality and performance were assessed based on R-squared values, as well as visual inspections of residual normality and homoscedasticity. The final model was further assessed with a train-test split using the full dataset.


## Results

### Initial Models
#### Initial Model Version 1 
The first version of the initial model was built using the following features:
```
target = 'price'
numerical = ['sqft_living', 'est_sqft_basement',
             'age', 'lat', 
             'bedrooms', 'floors']
categorical = ['waterfront', 'condition', 'renovated', 'zipcode']
```
Age was removed as a predictor from all three models fit to each of the three datasets based on its non-significant p-value (alpha=0.05). A few of the zip codes were found to be non-significant predictors for each model (11-22, depending on the dataset, with fewer non-significant zip codes as more outliers were removed). However, the majority of the 70 zip codes did explain a significant proportion of the variation in the data, so the category was retained as a predictor in all models.

| Model 1 | Full Dataset | Z-Score | IQR |
| :-: | :-: | :-: | :-: |
| R-squared | 0.783 | 0.796 | 0.794  |

Model performance for the first version of the initial model improved with the removal of outliers. As shown in the table above, the proportion of the variation explained by the model (R-squared) increased from 78.3% to 79.6% for the z-score cleaned data. This increase was very similar, though slightly lower for the IQR-cleaned data (79.4%).
The normality and homoscedasticity of model residuals improved dramatically with the removal of outliers. Residuals appeared to be the most normally distributed and homoscedastic for the model fit to the IQR-cleaned dataset.

**Residuals for Initial Model Version 1 - Full Dataset**
<img src="Figures/Initial Model 1-0_residuals.png" width = 1000 halign=center>

**Residuals for Initial Model Version 1 - Z-score Cleaned Dataset**
<img src="Figures/Z Model 1-0_residuals.png" width = 1000 halign=center>

**Residuals for Initial Model Version 1 - IQR Cleaned Dataset**
<img src="Figures/IQR Model 1-0_residuals.png" width = 1000 halign=center>


#### Initial Model Version 2
The second version of the initial model replaced `sqft_living` with three variables with which it was highly correlated in the full dataset (`sqft_above`, `sqft_living15`, and `bathrooms`):
```
target = 'price'
numerical = ['sqft_above', 'sqft_living15',
             'bathrooms', 'est_sqft_basement',
             'age', 'lat', 
             'bedrooms', 'floors',]
categorical = ['waterfront', 'condition', 'renovated', 'zipcode']
```
Unlike in the first version of the initial model, all predictors including age were retained as significant (p<0.05). Some zip codes were non-significant predictors as before, but the majority were significant, so zip code was retained. 

| Model 2 | Full Dataset | Z-Score | IQR |
| :-: | :-: | :-: | :-: |
| R-squared | 0.787 | 0.808 | 0.808  |

The second version of the model explained slightly more of the variation in the datathan the first version. Once again, model performance improved with the removal of outliers as shown in the table above. For this version of the model, the z-score and IQR-cleaned data produced the same high R-squared value of 0.808. The main difference in the two models were the improvements in residual normality and homoscedasticity for the IQR model compared to the z-score model. The changes were virtually identical to those grpahically depicted above for the first version of the initial model.

### Final Model
The removal of outliers for the final model decreased noise in the dataset, improving overall model performance compared to all initial models (R-Squared = 0.824). Outlier removal also had the added benefit of reducing correlations among variables that were all useful predictors of `price`. This allowed for the inclusion of most of these variables as predictors without introducing unacceptably high levels of multicollinearity. The final model was fit to the IQR-cleaned data and included the following variables with no transformation or scaling:
```
target = 'price'
numerical = ['sqft_living', 'est_sqft_basement',
             'age', 'lat', 'grade', 'sqft_living15',
             'bedrooms', 'floors', 'bathrooms']
categorical = ['waterfront', 'condition', 'renovated', 'zipcode']
```
For predictors that were included in initial models, their respective model parameters showed trends similar to those in the previous models. In other words, variables that were consistently associated with increased (or decreased) home prices in earlier models were still associated with increased (or decreased) home prices in this final model.

All the included features were significant predictors of price, except for 8 zipcodes out of the total 70. Since the vast majority of zipcodes were useful for predicting home price (some are associated with higher prices and some with decreased prices), this variable was kept in the model.

I evaluated multicollinearity in the final model by calculating variance inflation factors (VIF) for each of the predictors in the model. I considered a VIF of 6 to be the cutoff for inclusion in the model, but none of the predictors included in the model exceeded this value. Additionally, model residuals appeared to be reasonably normally distributed and homoscedastic as shown below:

**Final Model Residuals**
<img src="Figures/Final Model_residuals.png" width = 1000 halign=center>

When the final model was validated using a train-test split of the full dataset (no outliers removed), it performed very similarly and well for both the training data (R-Squared = 0.797) and the test data (R-Squared = 0.799). The R-squared values are likely less than that found for the final model (R-Squared = 0.824) because the train-test split was performed on the full dataset without outliers removed. The final model was fit to the IQR-cleaned data.

#### Interpretation of Final Model
Most of the predictors are associated with increasing home price, but a few of the predictors appear to have a negative impact on price:
**Negative**: The zip codes 98023 and 98092 are significant predictors associated with lower home prices compared to the reference zip code 98001 (98003, 98030, 98031, 98032, and 98042 have negative coefficients as well, but they are non-significant predictors). The area of the basement in square feet is also associated with a very slight decrease in home price of $38.68 per additional square foot. Increasing the number of bedrooms by 1 is associated with a $6,872.75 decrease in price and increasing the number of floors by 1 is associated with a $16,156.13 drop in price.

**Positive**: Features associated with increased home prices which the owners cannot control are age, and aspects of location such as zip code, latitude, and whether or not the property is on the water. Older homes tend to sell for slightly more money (one year older increases home price by $630). Homes at higher latitudes (further north) tend to sell for substantially more money (an increase of 1 degree of latitude is associated with a $135,200 increase in price). Additionally, homes in most of the zip codes sell for more money than the reference zipcode 98001. Homes with a waterfront view tend to sell for higher prices than properties off the water (a roughly $255,100 difference).

Aspects of homes which owners can attempt to change in an effort to increase home price include renovating the home, condition, grade, the number of bathrooms, and the square footage of interior living area. Homes that have been renovated tend to command higher prices. The model projects a difference in price of roughly $35,220 between homes that have been renovated compared to those that have not. Homes that are in better condition relative to their age tend to sell for more money. Homes with higher grades also command significantly higher prices (increasing grade by 1 is associated with a $45,410 increase in home price). Increasing the number of bathrooms by 1 is associated with a $14,760 increase in price. Additionally, an increase in living area of 1 square foot increases selling price by $113. Another factor that can change, but is not directly under the homeowner's control, is the average square footage of the 15 closest homes. The increase is not as great as increasing the square footage of the house itself, but for every 1 square foot increase in the average area of the 15 closest homes, price increases by $44.73.

## Conclusions
The final model appears to meet the assumptions of OLS regression (linearity, independence, normality and homoscedasticity of residuals) and it performed very well during the train-test split. As such, I am confident in its ability to predict the price of houses in King County, Washington, with some caveats. The data to which the model was fit were collected in 2014 and 2015. If the housing market has changed substantially, these predictions may not be as accurate now in 2020. Additionally, the model is best is best used to predict the price of houses between a range of $78,000 and $1.12 million, which represents nearly 95% of houses sold in King County between May 2014 and May 2015 (the full dataset before outlier removal contained houses that sold for a maximum of $7.7 million). The final model also applies best to houses with a maximum of 7,480 square feet of living area (as opposed to 13,540 sq ft in the full dataset), a maximum above ground square footage of 5,710 (compared to 7,320 sq ft), and maximum number of bathrooms of 7.5 (as opposed to 8 in the full dataset). It is also worth noting that none of the houses in this price range attained the highest grade of 13 (grades range from 3 to 12 for the final model). 

Overall, the model indicates that a substantial proportion of the variation in home price in King County is explained by aspects of location such as zip code and latitude. However, the model also shows that homeowners can potentially increase their selling price through renovations. Of course, the increase in selling price and thus the return on investment will vary depending on the quality and type of renovations.

<img src="Figures/House Prices in King County by Zip Code.png" width = 1000 halign=center>

<img src="Figures/Effect of Latitude on House Price.png" width = 700 halign=center>

<img src="Figures/Effect of Renovation on House Price.png" width = 600 halign=center>

## Recommendations
The results and insights from my final model support the following recommendations regarding the best ways to invest in maximizing the price of a home in King County:
> - **Improve condition through repairs** - If the home has received a condition score from the county of less than 3, efforts should be made to make whatever repairs are necessary to bring the score up to a 3. The repairs needed will be relative to the age of the home, and the homeowner may be able to contact county officials to determine the types of repairs needed. If the condition score is already 3 or greater, money should instead be invested in the renovations outlined below to maximize selling price.

<img src="Figures/Effect of Condition on House Price.png" width = 500 halign=center>

> - **Increase grade by renovating** - The final model indicates that, when all other variables are held constant, increasing grade by 1 is associated with a \$45,410 increase in selling price. It may be possible to increase the grade assigned to the home by increasing the interior square footage of the home, replacing finishes with higher quality materials, making improvements to architectural design, or other improvements mentioned in documentation provided by the county (<a href="https://www.kingcounty.gov/depts/assessor/Reports/area-reports/2017/residential-westcentral/~/media/depts/assessor/documents/AreaReports/2017/Residential/013.ashx">King County Assessments</a>).

<img src="Figures/Effect of Grade on House Price.png" width = 1000 halign=center>

> - **Add square footage to the interior living area** - Aside from potentially increasing a home's grade, adding square footage to the total interior living area can also increase selling price. According to the final model, an increase of 1 square foot tends to increase selling price by $113. Additionally, if the opportunity arises to influence the development or construction of neighboring houses, homeowners would benefit from an increase in the average square footage of their neighbors' homes.

<img src="Figures/Effect of Interior Living Area on House Price.png" width = 1000 halign=center>

> - **Add bathrooms** - Homeowners can increase the price of their home by adding bathrooms. On average, increasing the number of bathrooms by 1 is associated with a \$14,760 increase in price.

<img src="Figures/Effect of Number of Bathrooms on House Price.png" width = 800 halign=center>

> - **Do NOT add bedrooms or floors** - There appears to be no real benefit to doing so. The model actually predicts a decrease in selling price when either of these factors is increased, all else being equal.


## Future Work
* Since the final model was built using data collected from May 2014 through May 2015, a logical next step would be to obtain similar data of houses sold between that time and the present. This would help make the model more relevant to today's housing market.
* Collecting additional data from more recent years could allow for predicting the price of the more expensive houses (price range $1.12 million - $7.7 million) that were considered outliers and dropped from the data. The current model should not be used to predict the price of houses in this range with any confidence. Alternatively, with enough data collected on houses in this price range over recent years, a separate model could be fit specifically to these pricier homes.
* It could be helpful to build a model that helps predict the return on investment for specific types renovations based on the price of materials and cost of labor in this region. This would help homeowners prioritize how to invest their money to maximize not just their selling price, but also their profit when renovating and selling their home.
* It would also be helpful to use the predictions generated by this model as part of an interactive website. Ideally this website would allow homeowners to enter the exact information specific to their home and obtain an estimate of the expected price range for their home. They could then make adjustments to features that they have some control over, such as the number of bathrooms and total interior living area, to get a better idea of how certain renovations might impact that price.
