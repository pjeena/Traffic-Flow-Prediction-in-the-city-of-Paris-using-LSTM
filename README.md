
# Traffic Flow forecasting and Optimization

Traffic forecasting plays a crucial role in transportation planning, urban management, and resource allocation. Accurate predictions can help optimize traffic flow, reduce congestion, and improve overall transportation efficiency. In this project, I have developed a machine learning model to forecast traffic (no of vehicles passing across a junction at a particular hour) for the next 3 hours from the current time based on historical data (app link is on the right side in the about section). I have considered 8 major junctions of Paris which as as follows :

[Paris] Amsterdam x Clichy \
[Paris] CF1 Rivoli x Sébastopol \
[Paris] CF318 Rivoli x Lobau \
[Paris] CF4 Poissonnière x Montmartre EST \
[Paris] CF4 Poissonnière x Montmartre OUEST \
[Paris] CF892 Rivoli x Bourdonnais \
[Paris] Quai de Jemmapes \
[Paris] Quai de Valmy \
[Paris] Rivoli x Nicolas Flamel 

![embed](https://github.com/pjeena/Traffic-Management-and-Optimization-using-LSTM/blob/main/resources/schema.jpeg)




## Data Pipeline

The project collects data from the following API:

[Open Data Paris API](https://opendata.paris.fr/explore/dataset/comptage-multimodal-comptages/api/?disjunctive.label&disjunctive.mode&disjunctive.voie&disjunctive.sens&disjunctive.trajectoire&sort=t)

The API is maintained by City of Paris and it provides vehicle count data by:

* Travel modes (Scooters, Scooters + Bicycles (when the distinction between these 2 travel modes is not implemented on the sensor), Bicycles, 2 motorized wheels, Light vehicles < 3.5 tonnes, Heavy vehicles > 3.5 tonnes , Buses & Coaches),

* Traffic lane types (Corona-lanes, Cycle lanes, General lanes),

* Direction of traffic .

These data are constructed by using an artificial intelligence algorithm that analyzes images from thermal cameras installed in public spaces. Images from thermal cameras do not identify faces or license plates. The data thus collected does not contain any personal or individual data. No image is transferred or stored on computer servers, the analysis being carried out as close as possible to the thermal camera. Only count data is transmitted. More info can be known from their website.

The collected data is inserted into BiqQuery, a serverless and cost-effective enterprise data warehouse.




## Preprocessing and Feature engineering

Before building the machine learning model, the collected data is preprocessed to clean and transform it into a suitable format. Since, this is a time-series data, we need to create lag features. Here, I take 24 lag features which means that the traffic from last 24 hours is being considered to forecast the traffic at the current hour and the next two hours. Alternatively, this is a mulit step time series problem. The model schema is as follows :

![embed](https://github.com/pjeena/Traffic-Management-and-Optimization-using-LSTM/blob/main/resources/model_schema.jpeg)

Suppose we want to predict the traffic(**y(T_n)**) at the current hour(**T_n**) and the next two hours, we need the data for the past 24 hrs( **f(T_(n-24))** ..... **f(T_(n-1))**. It can easily be understood in the above table.

Since the data collection is through sensors (which might not work on sometime) there were a decent amount of missing values. They were imputed by using the data from the previous day which seems to be a resonable choice. As we know that, time is inherently cyclical. To include this information, we perform a cyclic encoding for each time stamp containing hour, day, month. This eventually helps our model to learn the cyclical nature of our data.

**For simplicity, I have assumed that the traffic at one junction is independent of other which is not entirely true in real life. To incorporate dependency one might need to use Graph neural networks.**
## Exploratory data analysis

One can get a preliminary idea regarding the traffic by some simple visualization. Below we plot the amount of traffic wrt hour, day of month, day of week.

![embed](https://github.com/pjeena/Traffic-Management-and-Optimization-using-LSTM/blob/main/resources/hour.png)

![embed](https://github.com/pjeena/Traffic-Management-and-Optimization-using-LSTM/blob/main/resources/day.png)

![embed](https://github.com/pjeena/Traffic-Management-and-Optimization-using-LSTM/blob/main/resources/dayname.png)

* From all the above figures we see that Boulevard Rivoli x Sebastopol is the busiest junction in Paris which is quite true. Actually, its one of the busiest roads in the world. On the contrary, Quai de Jemappes seems quite silent.

* Majority of the traffic is around the two time slots( 7-9am and 5-6 pm )which is expected since people go to work in the  morning during these hours and return at around 5pm.

* The traffic stays pretty constant during a particular month.

* Sunday sees the lowest amount of traffic since most of the people stay at home.


## Model Building

After getting the data in the appropriate format, a LSTM was trained on test set and validated on val set. The model was evaluated using rmse since we are tring to predict the number of vehicles passing across a junction. 





## CI/CD pipeline

The API is updated daily with the latest data available from the sensors. In order to get real time forecast we need to use automation so that our model can predict the forecast at the current hour and for the next 2 hours. This project uses Github Actions to do the same. 


**Data collection**: collect data from the API and insert it into BigQuery.  frequency : **hourly**

**Preprocessing**: preprocess the collected data and prepare it for machine learning. frequency : **hourly**

**Model training**: train the machine learning model on historical data. frequency : **weekly**

**Model evaluation**: evaluating the performance of the model on the data from the past 7 days.

**Model deployment**: deploy the model on a [web-based dashboard](https://traffic-management-and-optimization-in-paris.streamlit.app/), which displays real-time traffic and historical insights on different junctions of Paris.

The pipeline is triggered automatically whenever new data is available, ensuring that the model is always up-to-date and accurate.

Note : **Github Actions is not entirely accurate to trigger the pipeline at the scheduled time. Therefore, sometimes the dashboard might not reflect the forecasts for the next 2 hours.**


## Dashboard

The inferences were visualized by projecting the traffic to google maps showing the forecasted traffic and the change in traffic from prior hour.
Link to the [dashboard](https://traffic-management-and-optimization-in-paris.streamlit.app/).


This shows the traffic in 8 major junctions of Paris. More intensity of color -> more traffic
![embed](https://github.com/pjeena/Traffic-Management-and-Optimization-using-LSTM/blob/main/resources/dashboard_1.jpeg)

Here, we see the forecasts from the last  7 days and upto the next 3 hours
![embed](https://github.com/pjeena/Traffic-Management-and-Optimization-using-LSTM/blob/main/resources/dashboard_2.jpeg)

The model performs quite well. One can play around with the lag values or can include more future forecast hours instead of 3. It heavily depends on the relevant use case. 
## Installation and Usage

1. Install the dependencies `pip install -r requirements.txt`

2. Go to the src folder and run the pipeline one after another :

    `python 01_data_fetching_between_dates.py` \
    `python 02_data_ingestion_into_bigquery.py` \
    `python 03_data_preprocessing.py` \
    `python 04_feature_build.py` \
    `python 05_model_trainer.py` \
    `python 06_predict_pipeline.py` 


    One might want to make a GCP account to access BigQuery otherwise step 2 can be skipped.

3. To get the dashboard, run this :

    `python 01_🎈_Main_page.py` 


## Conclusion


This project demonstrates how machine learning algorithms can be used to forecast traffic in real-time, using data from different sources. The project also shows how a CI/CD pipeline can be implemented to automate the data processing, model training, and deployment, improving the efficiency and reliability of the project. 
## Future work

This project provides a foundation for further development and improvement. Some possible areas for future work include:

**Integration with additional data sources** : incorporating data from other sources, such as social media feeds, weather or traffic cameras.

**User feedback and interaction**: gathering user feedback and incorporating it into the design and functionality of the dashboard could improve its usability and usefulness for the public and law enforcement agencies.

Overall, I really enjoyed working on this end to end project. I enjoyed the challenge of collecting and preprocessing data from multiple sources and building a machine learning model to predict crime rates in real-time. The implementation of the CI/CD pipeline was a great learning experience for me as well, and I am proud of the automation and efficiency it brought to the project.
