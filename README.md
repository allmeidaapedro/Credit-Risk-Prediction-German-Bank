# German Bank Credit Risk Prediction

<img src="images/german_bank.jpg" width="800" height="450">

# 1. Description
- This is an <b>end to end machine learning project using Random Forest to predict credit risk of German Bank's customers.</b> It involves supervised learning (using a labeled training set) for <b>classification</b>, where the target is 1 if the customer represents a <b>bad risk</b> and 0 if he represents a <b>good risk</b>.
- I implemented this project following some CI/CD principles and using modular coding. First, I developed my entire analysis (from EDA to modeling) in notebooks. Then, I divided the project development into components responsible for data ingestion, transformation, and model training, following the same steps as in the notebooks. Once I had done this, I created scripts for automating the training and prediction pipelines using these components. The training pipeline executes them and obtains all the machine learning model artifacts, while the prediction pipeline makes predictions by consuming the obtained artifacts. All of this was made with good practices like exception handling, loggings and documentation. Afterward, I built a web app in Flask, integrating everything mentioned above. My objective with this was to get closer to a real data science project workflow by packaging my entire project as a package.

# 2. Business problem and project objective
<b>Predict credit risk of German Bank's customers.</b><br>

<b>Credit risk</b> refers to the potential financial loss that a lender, such as a bank, might incur if a borrower fails to repay a loan or credit obligation. It's the uncertainty about whether borrowers will honor their financial commitments.

<b>The German Bank aims to predict their customers' credit risk for several reasons:</b><br>
<b>1. Risk assessment.</b><br>
<b>2. Profitability.</b><br>
<b>3. Minimize losses.</b><br>
<b>4. Compliance.</b><br>
<b>5. Customer segmentation.</b><br>
<b>6. Strengthen trust.</b><br>

By employing <b>predictive models</b>, the <b>bank</b> can make <b>informed decisions</b> that <b>balance profit generation with prudent risk management</b>, ultimately benefiting both the institution and its customers. Thus, the <b>project objective is to build a model that is able to identify as many as possible bad risk customers and provide valuable insights about credit risk within the available features.</b> By doing this, the business problem is solved.

# 3. Solution pipeline
The following pipeline was used, based on CRISP-DM framework:

<b>1. Define the business problem.</b><br>
<b>2. Collect the data and get a general overview of it.</b><br>
<b>3. Split the data into train and test sets.</b><br>
<b>4. Explore the data (exploratory data analysis)</b><br>
<b>5. Data cleaning and preprocessing.</b><br>
<b>6. Model training, comparison, selection and tuning.</b><br>
<b>7. Final production model testing and evaluation.</b><br>
<b>8. Conclude and interpret the model results.</b><br>
<b>9. Deploy.</b><br>

# 4. Main business insights
<b>1.</b> Young customers tend to present higher credit risk. This makes sense since younger people usually don't have financial stability.<br>
<b>2.</b> Customers who take higher credit amounts from the bank tend to present higher risk. This makes total sense. The higher the credit amount taken, the harder it is to pay it.<br>
<b>3.</b> Customers whose credit services have a long duration present higher risk. The more time a customer spends using a credit service without paying it, the higher the risk of default.<br>

<img src="images/numericalvstarget.png" width="800" height="250">

<b>4.</b> Credit amount and Duration are strongly positively correlated. Credit services with longer durations generally are associated with higher credit amounts and vice-versa.<br>

<img src="images/credit_duration.png" width="800" height="250">

<b>5.</b> Customers with little saving and checking accounts tend to present higher credit risk. Particularly, almost 50% of the customers who have little checking accounts are bad risk ones. Moreover, when a customer takes credit from the bank for vacation/others and education purposes, it must be alert. Specifically, almost 50% of the customers who took credit for education are bad risk.<br>

<img src="images/categoricalvstarget.png" width="800" height="450">
<img src="images/purposevstarget.png" width="800" height="250">

# 5. Modelling
1. I chose a set of models for performance comparison, analyzing the ROC-AUC score. Accuracy is not a good metric because the target is imbalanced.
2. In order to select the best model for hyperparameter tuning and final evaluation, I trained and evaluated each of the models using stratified k-fold cross-validation, which provides a more reliable performance estimate.
3. Although Logistic Regression had the best average validation score, I chose Random Forest for the next steps because there was more room for performance improvement with it. It was overfitting, and I could apply some regularization techniques to enhance its generalization ability.

<img src="images/cross_val.png">

4. I tuned Random Forest model with Bayesian Search because it uses probabilistic models to intelligently explore the hyperparameter space, balancing exploration and exploitation. An important point here was to define a class_weight hyperparameter, such that the estimator was able to better learn the patterns in minority target class (bad risk customers).
5. I evaluated the results and looked at precision-recall trade-off. Once the project aim is to predict as many as possible bad risk customers, recall is the metric of interest. Thus, I selected a threshold that provided a 0.8 recall without significantly compromising the precision score, getting the final model with satisfying results! In fact, the precision fell down just 0.01 (from 0.49 to 0.48) while the recall increased at 0.12 (from o.68 to 0.8).

Classification report before precision-recall trade-off balancing:

<img src="images/class_report_2.jpeg" width="479" height="182">

Classification report after precision-recall trade-off balancing:

<img src="images/class_report_1.jpeg" width="479" height="182">

Finally, the results are satisfying, and the project objective was achieved. The model can predict 80% of the bad risk customers, effectively solving the business problem. In summary, to address the imbalanced target, I implemented strategies such as stratified hold-out and k-fold cross-validation, utilized class-weight hyperparameter, and balanced precision-recall trade-off.