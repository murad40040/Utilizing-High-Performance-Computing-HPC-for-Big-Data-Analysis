# Utilizing-High-Performance-Computing-HPC-for-Big-Data-Analysis
Executes a data analysis pipeline on a large-scale dataset (>100 GB) using the Texas Advanced Computing Center (TACC) supercomputer.

Project Report
Utilizing High-Performance Computing for Big Data Analysis: A Flight Delay Prediction Study
Author: Md Murad Sharif

Abstract
This project demonstrates the use of a High-Performance Computing (HPC) environment at the Texas Advanced Computing Center (TACC) to analyze a large dataset of airline flight information. The primary goal was to build and compare two machine learning models for predicting flight delays. The project involved preprocessing a dataset with millions of records, training a Logistic Regression and a Random Forest model, and evaluating their performance based on accuracy and computational time. The results show that the Random Forest Classifier achieved higher predictive accuracy, while the Logistic Regression model was significantly faster to train, highlighting the classic trade-off between model complexity and computational cost in big data analysis.
________________________________________
1. Dataset
The dataset used for this project is the 2015 Flight Delays and Cancellations dataset, which contains over five million records of domestic flights in the United States.
•	Link to Dataset: The data was downloaded form Kaggle.  
https://www.kaggle.com/datasets/usdot/flight-delays?select=flights.csv
________________________________________
2. Instructions to Compile and Run Program
The analysis was conducted within a Jupyter Notebook environment on the TACC portal.
1.	Environment Setup: Launch a Jupyter Notebook session from the TACC user portal.
2.	Install Libraries: If the required libraries are not already installed in your Python environment, run the following command in a notebook cell:
Python
!pip install pandas numpy matplotlib seaborn scikit-learn
3.	Run the Program: Copy the complete Python script into a new Jupyter Notebook cell and execute it. The script automatically performs the following actions:
o	Loads the data directly from the web URL.
o	Preprocesses the data.
o	Trains both machine learning models.
o	Outputs performance reports and a final comparison table.
________________________________________
3. Preprocessing Techniques
The raw dataset required several preprocessing steps to be suitable for machine learning:
1.	Target Variable Creation: A binary target variable named IS_DELAYED was created. A flight was classified as delayed (1) if its arrival delay (arrival_delay) was greater than 15 minutes, and on-time (0) otherwise.
2.	Feature Selection: To create a manageable model for this analysis, a subset of relevant numeric features was selected: month, day_of_week, departure_time, distance, and arrival_time. Categorical features and identifiers were excluded to simplify the model.
3.	Handling Missing Values: Rows containing any missing values (NaN) in the selected features or the target variable were dropped from the dataset to ensure data quality.
________________________________________
4. Algorithm Parameter Selection
Two classification algorithms were chosen to compare a simple linear model against a more complex ensemble model.
1.	Logistic Regression:
o	Parameters: max_iter=1000, random_state=42.
o	Justification: The max_iter (maximum iterations) was set to 1000 to ensure the model had enough iterations to converge on this large dataset. The random_state was set for reproducibility. Default values for other parameters, like regularization (C=1.0), were used to establish a standard performance baseline.
2.	Random Forest Classifier:
o	Parameters: n_estimators=100, max_depth=10, n_jobs=-1, random_state=42.
o	Justification: n_estimators=100 was chosen as it provides a good balance between high accuracy and computational load. max_depth=10 was selected to prevent the model from overfitting to the training data. n_jobs=-1 is a critical parameter for HPC, as it instructs the model to use all available CPU cores on the TACC node for parallel processing, significantly speeding up training.
________________________________________

Results Summary
The performance of the two models is summarized below.
Model	Training Time (s)	Accuracy
Logistic Regression	25.15	0.8195
Random Forest Classifier	142.30	0.8250
________________________________________
6. Discussion & Comparison
The results highlight a fundamental trade-off in machine learning.
•	Performance: The Random Forest Classifier achieved a slightly higher accuracy (82.50%) compared to the Logistic Regression model (81.95%). This is expected, as Random Forest is an ensemble method capable of capturing more complex, non-linear patterns in the data.
•	Computational Cost: The Logistic Regression model was significantly faster, taking only ~25 seconds to train. In contrast, the Random Forest model took ~142 seconds, nearly six times longer, even when using all available CPU cores.
For an application where real-time prediction is not critical and accuracy is paramount, the Random Forest model is the superior choice. However, if computational resources are limited or predictions need to be generated very quickly, the Logistic Regression model provides a reasonable level of accuracy with a much lower computational footprint.
7. Conclusion
This project successfully demonstrated the end-to-end process of a big data analysis project on an HPC platform. A large dataset of flight records was processed, and two distinct classification models were trained and compared. The analysis confirmed that more complex models like Random Forest can yield higher accuracy but come at a significant computational cost, reinforcing the importance of selecting the right tool for the specific performance requirements of a data science task.

