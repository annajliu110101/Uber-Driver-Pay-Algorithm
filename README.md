# Uber Fare Prediction — NYC FHV 2024–2025

## Introduction

Uber’s business model is a technology platform that connects service providers such as merchants and drivers to their customers for on-demand transportation and delivery services. It does not consider itself a ride-hailing service, and the algorithms created to manage dynamic pricing and driver pay are considered proprietary information. This model allows Uber to claim lower commission rates while maintaining full control over driver compensation, and by classifying drivers as independent contractors, the company avoids providing benefits such as health insurance or worker compensation. In response, the NYC Taxi and Limousine Commission (TLC) introduced a high-volume for-hire license class in 2019, requiring companies like Uber and Lyft to submit more detailed trip data. This transparency has already helped regulators uncover incidents of wage theft and contract violations.

For our project, we acquired monthly datasets between July 2024 and June 2025, totaling more than 170 million records. Each record contains pickup and dropoff times, locations, trip distance and duration, passenger fares, and driver pay. The richness of this dataset allowed us to analyze pricing patterns across New York City at scale. During early exploration, we found striking variations in driver pay that did not always align with the base passenger fare, making the dataset both interesting and useful for modeling. We selected this dataset from Kaggle because of its size and detail, which made it well-suited for building predictive models. Decision tree regression was chosen as the foundation of our modeling approach because trees capture non-linear patterns and produce interpretable results. They were particularly useful in identifying breakpoints in distance and time windows where fares shifted sharply.

A good predictive model for passenger fares has value beyond accuracy. It provides transparency for riders, giving them clear expectations about how location, time, and traffic conditions impact price. It also helps platforms and fleets plan supply and demand more efficiently, reducing wait times and optimizing promotions. Finally, policymakers can use these predictions to audit for fairness across neighborhoods and time periods. Strong models improve both consumer experience and regulatory oversight.

## Methods

Our methodology began with a preprocessing pipeline to clean and enrich the data. We filtered out invalid or zero fares, aligned schema changes across months, and standardized numeric fields such as tolls, surcharges, and booking fees. Using Polars, we engineered time-based features including pickup hour, weekday, month, rush-hour and late-night indicators, and weekend flags. Route identifiers combining pickup and dropoff zones were also introduced, along with distance buckets to capture non-linear effects of trip length.  

To enrich the dataset, we mapped U.S. federal holidays for the project period and integrated weather data from Meteostat. Zone centroids derived from the NYC taxi zones shapefile allowed us to assign hourly temperature, precipitation, humidity, wind speed, and snow indicators to each trip. This additional context helped capture external conditions that influence pricing and rider demand.  

Once features were built, we converted the dataset to pandas for modeling. We used correlation analysis to identify the strongest predictors of base fare, with trip time, trip miles, and dropoff location emerging as the top drivers. Scatter plots confirmed clear fare increases at breakpoints in distance and time. These steps provided confidence that our features captured both operational and contextual drivers of fare variation.

## Models

For the first model, we trained a decision tree regressor using XGBoost. We tuned depth, learning rate, and boosting rounds to balance fit and generalization. Tree-based methods proved effective because they captured sharp changes in fare patterns by distance, time, and location while handling the large dataset efficiently. Using the histogram algorithm, we were able to scale training across millions of rows on both laptop and SDSC environments.  

For the second model, we applied an unsupervised learning approach. We reduced feature dimensionality with PCA and then clustered trips using KMeans. This revealed groups of trips with similar characteristics, such as short intra-borough rides versus longer cross-city routes. We then re-framed the regression problem into a binary classification task of predicting whether fares fell above or below the median. By evaluating confusion matrices, we were able to measure false positives and false negatives alongside regression accuracy, giving a fuller picture of model performance.  

The combination of regression and clustering allowed us to both predict fares and uncover hidden structures in rider and trip behavior.

## Results

The final regression model achieved an RMSE of around $4.02 across the full dataset, significantly improving upon earlier versions. The underfitting/overfitting diagnostic showed that deeper trees reduced training error but only marginally improved test error, suggesting diminishing returns after depth six. The fitting curve is shown below:

<img width="655" height="468" alt="Screenshot 2025-09-03 at 11 41 51 PM" src="https://github.com/user-attachments/assets/bc4c215a-c9e8-4e1a-9d92-ed4ed303e199" />


This curve confirmed that our chosen parameters struck a balance between accuracy and generalization. In classification-style evaluation, the model showed balanced performance, though some errors were made in predicting above-median fares, reflecting natural volatility in Uber’s dynamic pricing.  

From a data perspective, we uncovered several insights. Driver pay often failed to scale linearly with passenger fares, showing that commission rates and booking fees significantly alter payouts. In some trips, drivers received nearly the full fare, while in others their take-home was substantially lower. Adding weather revealed that rainy and snowy conditions coincided with higher base fares, consistent with expected demand surges. Holidays like Thanksgiving and New Year’s Eve showed similar spikes, validating that external context plays a real role in Uber’s pricing system.  

## Discussion

Our process revealed both strengths and limitations. Tree-based models proved well-suited to this task, handling large, non-linear datasets efficiently while offering interpretable breakpoints. The integration of holidays and weather data improved accuracy, reducing residual error and highlighting meaningful external drivers of fare dynamics. However, Uber’s proprietary pricing introduces noise that cannot be fully captured with TLC-reported variables alone. Driver compensation patterns in the data often contradicted the expected commission structure, reflecting a lack of transparency in the platform’s financial flows.  

The results are believable, but not perfect. Fare prediction is constrained by the complexity of Uber’s internal algorithms and dynamic multipliers, which remain inaccessible. Some shortcomings included missing or incomplete weather station coverage, borough-level centroid approximations for zones, and the exclusion of certain passenger fees classified as proprietary by TLC. Future work could expand to heteroscedastic models, quantile regression, or blending tree ensembles with neural methods. Despite these limitations, our project successfully demonstrated that fare dynamics can be modeled transparently and meaningfully with public TLC data.

## Conclusion

Our project showed that predicting base passenger fare from TLC’s high-volume trip data is feasible with strong accuracy when combining tree-based regression, contextual features, and unsupervised clustering. The final model achieved an RMSE of just over $4, meaning predictions were on average within a few dollars of the actual base fare. Adding holidays and weather proved critical to capturing external demand spikes. More broadly, the analysis provided useful insights into fare transparency, the impact of external events, and discrepancies in driver pay. Future improvements could involve leveraging GPU training at scale, incorporating additional external datasets, or reframing the problem into multi-task learning with both fare and driver pay as targets.  

## Authors

- Fawaz Al-Senayin  
- Anna Liu
- Ethan Gross 

## Statement of Collaboration

This project was a collaborative effort between all group members. We additionally used ChatGPT as a tool to help debug code, and refine written explanations. All outputs were verified, adapted, and critically assessed by the team before inclusion in the final submission. All members rotated taking turns coding and doing the write-up.

## Repository Links

- Milestone 3 notebook (Model 1): [link-to-notebook]  
- Milestone 4 notebook (Model 2): [notebooks/Model 2.ipynb]  
