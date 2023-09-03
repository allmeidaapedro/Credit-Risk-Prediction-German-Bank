# German Bank Credit Risk Prediction

<img src="images/german_bank.jpg" width="800" height="450">

# 1. Description
- This is a <b>machine learning project using Random Forest to predict credit risk of German Bank's customers.</b> It involves supervised learning (using a labeled training set) for <b>classification</b>, where the target is 1 if the customer represents a <b>bad risk</b> and 0 if he represents a <b>good risk</b>.
- II implemented this project following some CI/CD principles and using modular coding. First, I developed my entire analysis (from EDA to modeling) in notebooks. Then, I divided the project development into components responsible for data ingestion, transformation, and model training, following the same steps as in the notebooks. Once I had done this, I created scripts for automating the training and prediction pipelines using these components. The training pipeline executes them and obtains all the machine learning model artifacts, while the prediction pipeline makes predictions by consuming the obtained artifacts. Afterward, I built a web app in Flask, integrating everything mentioned above. My objective with this was to get closer to a real data science project workflow by packaging my entire project as a package.

# 2. Business problem and project objective.
<b>Predict credit risk of German Bank's customers.</b><br>

<b>Credit risk</b> refers to the potential financial loss that a lender, such as a bank, might incur if a borrower fails to repay a loan or credit obligation. It's the uncertainty about whether borrowers will honor their financial commitments.

<b>The German Bank aims to predict their customers' credit risk for several reasons:</b><br>
1. Risk assessment
2. Profitability
3. Minimize losses
4. Compliance
5. Customer segmentation
6. Strengthen trust

By employing <b>predictive models</b>, the <b>bank</b> can make <b>informed decisions</b> that <b>balance profit generation with prudent risk management</b>, ultimately benefiting both the institution and its customers. Thus, the <b>project objective is to build a model that is able to identify as many as possible bad risk customers and provide valuable insights about credit risk within the available features.</b> By doing this, the business problem is solved.