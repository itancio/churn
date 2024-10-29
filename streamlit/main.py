import streamlit as st
import pandas as pd
import numpy as np
import pickle
import utils as ut
import sklearn as sk
import xgboost as xgb

from openai import OpenAI

client = OpenAI()

# For debugging purposes
print('Numpy verstion: ', np.__version__)
print('Pandas verstion: ', pd.__version__)
print('Sklearn verstion: ', sk.__version__)


def load_model(filename):
  with open(filename, 'rb') as file:
    return pickle.load(file)

# Trained models
# xgboost_model = load_model('models_churn/XGBClassifier.pkl')
# naive_bayes_model = load_model('models_churn/GaussianNB.pkl')
# random_forest_model = load_model('models_churn/RandomForestClassifier.pkl')
# decision_tree_model = load_model('models_churn/DecisionTreeClassifier.pkl')
# svm_model = load_model( 'models_churn/SVC.pkl')
# knn_model = load_model('models_churn/KNeighborsClassifier.pkl')

# Train models with feature engineering
voting_classifier_model = load_model('models_churn/VotingClassifier-voting.pkl')
xgboost_SMOTE_model = load_model('models_churn/XGBClassifier-SMOTE.pkl')
xgboost_featureEngineered_model = load_model('models_churn/XGBClassifier-featureEngineer.pkl')
naive_bayes_model = load_model('models_churn/GaussianNB-engineered.pkl')
random_forest_model = load_model('models_churn/RandomForestClassifier-engineered.pkl')
decision_tree_model = load_model('models_churn/DecisionTreeClassifier-engineered.pkl')
svm_model = load_model( 'models_churn/SVC-engineered.pkl')
knn_model = load_model('models_churn/KNeighborsClassifier-engineered.pkl')

def prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary):
  input_dict = {
    'CreditScore': credit_score,
    'Geography_France': 1 if location == 'France' else 0,
    'Geography_Germany': 1 if location == 'Germany' else 0,
    'Geography_Spain': 1 if location == 'Spain' else 0,
    'Gender_Male': 1 if gender == 'Male' else 0,
    'Gender_Female': 1 if gender == 'Female' else 0,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'HasCrCard': has_credit_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary,

    # Additional features
    'TenureAgeRatio' : tenure / age if age > 0 else 0,
    'CLV' : (balance + estimated_salary) / (age + 1),
    'AgeGroup_MiddleAge' : 1 if 30 <= age < 44 else 0,
    'AgeGroup_Senior' : 1 if 45 <= age < 59 else 0,
    'AgeGroup_Elderly' : 1 if age >= 45 else 0,
  }

  input_df = pd.DataFrame([input_dict])
  return input_df, input_dict

def make_predictions(input_df, input_dict):
  # Define the expected order of features for XGBoost
  expected_order = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 
    'IsActiveMember', 'EstimatedSalary', 'Geography_France', 'Geography_Germany', 'Geography_Spain', 
    'Gender_Female', 'Gender_Male', 'CLV', 'TenureAgeRatio', 'AgeGroup_MiddleAge', 
    'AgeGroup_Senior', 'AgeGroup_Elderly'] 
  
  # Reorder the input DataFrame
  input_df = input_df[expected_order]

  # Make predictions
  nb_predict = naive_bayes_model.predict_proba(input_df)[0][1],
  rf_predict = random_forest_model.predict_proba(input_df)[0][1],
  dt_predict = decision_tree_model.predict_proba(input_df)[0][1],
  knn_predict = knn_model.predict_proba(input_df)[0][1],
  svm_predict = svm_model.predict_proba(input_df)[0][1],
  vc_predict = voting_classifier_model.predict_proba(input_df)[0][1],
  xgb_smote_predict = xgboost_SMOTE_model.predict_proba(input_df)[0][1],
  xgb_featureEngineered = xgboost_featureEngineered_model.predict_proba(input_df)[0][1]

  nb_predict = nb_predict[0]
  rf_predict = rf_predict[0]
  dt_predict = dt_predict[0]
  knn_predict = knn_predict[0]
  svm_predict = svm_predict[0]
  vc_predict = vc_predict[0]
  xgb_smote_predict = xgb_smote_predict[0]
  xgb_featureEngineered = xgb_featureEngineered

  probabilities = {}

  # Filter out predictions that are very close to zero
  min_threshold = 0.0001
  if nb_predict >= min_threshold: probabilities['Naive Bayes'] = nb_predict
  if rf_predict >= min_threshold: probabilities['Random Forest'] = rf_predict
  if dt_predict >= min_threshold: probabilities['Decision Tree'] = dt_predict
  if knn_predict >= min_threshold: probabilities['K-Nearest Neighbors'] = knn_predict
  if svm_predict >= min_threshold: probabilities['SVM'] = svm_predict
  if vc_predict >= min_threshold: probabilities['Voting Classifier'] = vc_predict
  if xgb_smote_predict >= min_threshold: probabilities['XGBoost SMOTE'] = xgb_smote_predict
  if xgb_featureEngineered >= min_threshold: probabilities['XGBoost Feature Engineered'] = xgb_featureEngineered
  
  print(probabilities)

  # Calculate the average probability
  avg_probability = np.mean(list(probabilities.values()))
  print('avg_probability: ', avg_probability)

  col1, col2 = st.columns(2)

  with col1:
    fig = ut.create_gauge_chart(avg_probability)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"The customer has a {avg_probability:.2%} probability of churning.")

    st.markdown("### Model Probabilities")
    col1_1, col1_2 = st.columns(2)
    for model, prob in probabilities.items():
      col1_1.write(f"{model}: ")
      col1_2.write(f"{prob * 100:.2f}%")

  with col2:
    fig_probs = ut.create_model_probability_chart(probabilities)
    st.plotly_chart(fig_probs, use_container_width=True)

  st.markdown(f"### Average Probability: {avg_probability * 100:.2f}%")

  return avg_probability

def explain_prediction(probability, input_dict, surname):
  systemPrompt = f"""
  You are an expert data scientist at a bank, wehre you specialize in interpreting
  and explaining predictions of machine learning models.

  Your machine learning model has predicted that a customer named {surname} has a
  {probability * 100}% probablity of churning, based on the information provided below.

  Here is the customer's information:
  {input_dict}

  Here are the machine learning model's top 10 most important features for predciting churn:


  features	          importance
  NumOfProducts	      0.323888
  IsActiveMember	    0.164146
  Age	                0.109550
  Geography_Germany	  0.091373
  Balance	            0.052786
  Geography_France	  0.046463
  Gender_Female  	    0.045283
  Geography_Spain	    0.036855
  CreditScore	        0.035005
  EstimatedSalary	    0.032655
  HasCrCard	          0.031940
  Tenure	            0.030054
  Gender_Male	        0.000000

  {pd.set_option('display.max_columns', None)}

  Here are the summary statistics for churned customers:
  {df[df['Exited'] == 1].describe()}

  Here are summary statistics for non-churned customers:
  {df[df['Exited'] == 0].describe()}

  - If the customer has over a 40% risk of churning, genearte a 3 sentence explanation of
  why they are at risk of churning.
  - If the customer has less than a 40$ risk of churning, generate a 3 sentence explanation
  of why they might not be at risk of churning.
  - Your explanation should be based on the customer's information, the summary statitics
  of churned and non-churned customers, and the feature importances provided.

  Don't mention the probability of churning, or the machine learning model,
  or say anything like "Based on the machine learning model's predictions and top 10 most
  important features, just explain the prediction.

  """

  print("EXPLANATION PROMPT", systemPrompt)

  raw_response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{
      'role': 'user',
      'content': systemPrompt
    }]
  )

  return raw_response.choices[0].message.content

def generate_email(probability, input_dict, explanation, surname):
  systemPrompt = f"""
  You are a manager at HS Bank. You are responsible for ensuring customers stay
  with the bank and are incentivized with various offers.

  You noticed a customer named {surname} has a {probability * 100}% probability of churning.
  Here is the customer's information:
  {input_dict}

  Here is the explanation of the customer's churning probability:
  {explanation}

  Generate an email to the customer based on their information,
  asking them to stay if they are at risk of churning, or offerign them incentives
  so taht they become more loyal to the bank.

  Make sure to list out a set of incentives to stay based on their information,
  in bullet point format. Don't ever mention the probability of churning, or
  the machine learning model to the customer.
  """

  raw_response = client.chat.completions. create(
    model='gpt-4o-mini',
    messages=[{
      'role': 'user',
      'content': systemPrompt
    }]
  )

  print("\n\nEMAIL PROMPT", systemPrompt)
  
  return raw_response.choices[0].message.content

tab1, tab2 = st.tabs([
  "Customer Churn Prediction",
  "Fraud Detection Predictions"
])

with tab1:
  st.title("Customer Churn Prediction")

  # Load dataset
  # path = "https://media.githubusercontent.com/media/itancio/churn/refs/heads/main/streamlit/churn.csv"
  # df = pd.read_csv(path)

  # customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

  # selected_customer_option = st.selectbox('Select a customer', customers)

  # if selected_customer_option:
  #   selected_customer_id = int(selected_customer_option.split(" - ")[0])
  #   selected_surname = selected_customer_option.split(" - ")[1]
  #   selected_customer = df.loc[df['CustomerId'] == selected_customer_id].to_dict(
  #     orient='records')

  #   customer_surname = selected_customer[0]['Surname']
  #   customer_credit_score = selected_customer[0]['CreditScore']
  #   customer_location = selected_customer[0]['Geography']
  #   customer_gender = selected_customer[0]['Gender']
  #   customer_age = selected_customer[0]['Age']
  #   customer_tenure = selected_customer[0]['Tenure']

  #   customer_balance = selected_customer[0]['Balance']
  #   customer_num_products = selected_customer[0]['NumOfProducts']
  #   customer_has_credit_card = selected_customer[0]['HasCrCard']
  #   customer_is_active_member = selected_customer[0]['IsActiveMember']
  #   customer_estimated_salary = selected_customer[0]['EstimatedSalary']

  #   col1, col2 = st.columns(2)

  #   with col1:
  #     credit_score = st.number_input(
  #       "Credit Score",
  #       min_value=300,
  #       max_value=850,
  #       value=customer_credit_score)

  #     locations = ['Spain', 'France', 'Germany']
      
  #     location = st.selectbox(
  #       "Location", locations,
  #       index=locations.index(customer_location))

  #     genders = ['Male', 'Female']
      
  #     gender = st.radio('Gender', genders,
  #                     index=0 if customer_gender=='Male' else 1)

  #     age = st.number_input(
  #       'Age',
  #       min_value=18,
  #       max_value=100,
  #       value=customer_age
  #     )

  #     tenure = st.number_input(
  #       'Tenure (years)',
  #       min_value=0,
  #       max_value=50,
  #       value=customer_tenure
  #     )

  #   with col2:
  #     balance = st.number_input(
  #       "Balance",
  #       min_value=0.0,
  #       value=customer_balance
  #     )
    
  #     estimated_salary = st.number_input(
  #       'Estimated Salary',
  #       min_value=0.0,
  #       value=customer_estimated_salary
  #     )
      
  #     num_products = st.number_input(
  #       "Number of products",
  #       min_value=0,
  #       max_value=10,
  #       value=customer_num_products
  #     )
    
  #     has_credit_card = st.checkbox(
  #       'Has Credit Card',
  #       value=customer_has_credit_card
  #     )
      
  #     is_active_member = st.checkbox(
  #       "Is Active Member",
  #       value=customer_is_active_member
  #     )

  #   input_df, input_dict = prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary)
  #   print(input_df)

  #   avg_probability = make_predictions(input_df, input_dict)
  #   print(avg_probability)

  #   st.markdown('---')
  #   st.subheader('Explanation of Prediction')
  #   explanation = explain_prediction(avg_probability, input_dict, customer_surname)
  #   st.markdown(explanation)

  #   st.markdown('---')
  #   st.subheader('Personalized Email')
  #   email = generate_email(avg_probability, input_dict, explanation, customer_surname)
  #   st.markdown(email)




################################################################################################################################

# Load models
xgboost_model = load_model('models_fraud/XGBClassifier.pkl')
naive_bayes_model = load_model('models_fraud/GaussianNB.pkl')
random_forest_model = load_model('models_fraud/RandomForestClassifier.pkl')
decision_tree_model = load_model('models_fraud/DecisionTreeClassifier.pkl')


def prepare_fraud_input(category, amount, age, gender, state, median_price):
  input_dict = {
    'Amount' : amount,
    'Age' : age,
    'Price_Ratio' : amount / median_price,

  }

  input_df = pd.DataFrame([input_dict])
  return input_df, input_dict

def make_fraud_predictions(input_df, input_dict):
  # Define the expected order of features for XGBoost
  expected_order = [
    'Gender',
    'Age',
    'State',
    'Job',
    'Merchant',
    'Category',
    'Amount',
  ] 
  
  # Reorder the input DataFrame
  input_df = input_df[expected_order]

  # Convert categorical columns to the 'category' dtype
  categorical_cols = ['State', 'Category']
  input_df['Category'] = input_df[categorical_cols].astype('category')
  print("input shape: ", input_df.shape)


  # Make predictions
  xgb_predict = xgboost_model.predict_proba(input_df)[0][1],
  nb_predict = naive_bayes_model.predict_proba(input_df)[0][1],
  rf_predict = random_forest_model.predict_proba(input_df)[0][1],
  dt_predict = decision_tree_model.predict_proba(input_df)[0][1],

  xgb_predict = xgb_predict[0]
  nb_predict = nb_predict[0]
  rf_predict = rf_predict[0]
  dt_predict = dt_predict[0]

  probabilities = {}

  # Filter out predictions that are very close to zero
  min_threshold = 0.0001
  if xgb_predict >= min_threshold: probabilities['XGBoost'] = xgb_predict
  if nb_predict >= min_threshold: probabilities['Naive Bayes'] = nb_predict
  if rf_predict >= min_threshold: probabilities['Random Forest'] = rf_predict
  if dt_predict >= min_threshold: probabilities['Decision Tree'] = dt_predict
  
  print(probabilities)

  # Calculate the average probability
  avg_probability = np.mean(list(probabilities.values()))
  print('avg_probability: ', avg_probability)

  col1, col2 = st.columns(2)

  with col1:
    fig = ut.create_gauge_chart(avg_probability)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"The customer has a {avg_probability:.2%} probability of churning.")

    st.markdown("### Model Probabilities")
    col1_1, col1_2 = st.columns(2)
    for model, prob in probabilities.items():
      col1_1.write(f"{model}: ")
      col1_2.write(f"{prob * 100:.2f}%")

  with col2:
    fig_probs = ut.create_model_probability_chart(probabilities)
    st.plotly_chart(fig_probs, use_container_width=True)

  st.markdown(f"### Average Probability: {avg_probability * 100:.2f}%")

  return avg_probability





# Load dataset
path = "https://media.githubusercontent.com/media/itancio/churn/refs/heads/main/notebook/fraud/fraudTrain.csv"
df = pd.read_csv(path, index_col=0)

# Needed for the map
df = df.rename(columns={'long' : 'lon'})

with tab2:
  st.title("Fraud Detection Predictions")

  # sample = pd.DataFrame([{
  #   'trans_date_trans_time': '2019-01-01 00:00:18', 
  #   'cc_num': 2703186189652095, 
  #   'merchant': 'fraud_Rippin, Kub and Mann', 
  #   'category': 'misc_net', 
  #   'amt': 4.97, 
  #   'first': 'Jennifer', 
  #   'last': 'Banks', 
  #   'gender': 'F', 
  #   'street': '561 Perry Cove', 
  #   'city': 'Moravian Falls', 
  #   'state': 'NC', 
  #   'zip': 28654, 
  #   'lat': 36.0788, 
  #   'lon': -81.1781, 
  #   'city_pop': 3495, 
  #   'job': 'Psychologist, counselling', 
  #   'dob': '1988-03-09', 
  #   'trans_num': '0b242abb623afc578575680df30655b9', 
  #   'unix_time': 1325376018, 
  #   'merch_lat': 36.011293, 
  #   'merch_long': -82.048315, 
  #   'is_fraud': 0
  # }])

  # transactions = sample['trans_num'] + ' - ' + sample['last']

  # Create lists
  states = sorted(list(df['state'].unique()))
  jobs = sorted(list(df['job'].unique()))
  merchants = sorted(list(merch.split('fraud_')[1] for merch in df['merchant'].unique()))
  categories = sorted(list(df['category'].unique()))
  median_price = df['amt'].median()

  print('States: ', states)

  transactions = [f"{row['last']}, {row['first']} - {row['trans_num']}" for _, row in df.iterrows()]
  selected_transaction_option = st.selectbox('Select a transaction', transactions)

  if selected_transaction_option:
    selected_transaction_id = selected_transaction_option.split(" - ")[1]
    selected_transaction = df.loc[df['trans_num'] == selected_transaction_id].to_dict(orient='records')

    customer_first = selected_transaction[0]['first']
    customer_last = selected_transaction[0]['last']
    customer_gender = selected_transaction[0]['gender']
    customer_job = selected_transaction[0]['job']
    customer_birthdate = selected_transaction[0]['dob']
    customer_street = selected_transaction[0]['street']
    customer_city = selected_transaction[0]['city']
    customer_state = selected_transaction[0]['state']
    customer_zip = str(selected_transaction[0]['zip'])
    customer_lat = selected_transaction[0]['lat']
    customer_long = selected_transaction[0]['lon']

    selected_city_pop = selected_transaction[0]['city_pop']
    selected_merchant = selected_transaction[0]['merchant'].split('fraud_')[1]
    selected_merchant_lat = selected_transaction[0]['merch_lat']
    selected_merchant_long = selected_transaction[0]['merch_long']
    selected_category = selected_transaction[0]['category']
    selected_amt = selected_transaction[0]['amt']
    selected_is_fraud = selected_transaction[0]['is_fraud']

    # Preprocess Age Feature
    selected_trans_date = selected_transaction[0]['trans_date_trans_time']
    customer_dob = selected_transaction[0]['dob'] 

    selected_trans_date = pd.to_datetime(selected_trans_date)
    customer_dob = pd.to_datetime(customer_dob)

    selected_trans_year = selected_trans_date.year
    customer_dob_year = customer_dob.year

    customer_age = selected_trans_year - customer_dob_year

    st.map(selected_transaction, latitude=customer_lat, longitude=customer_long, color="#0044ff", zoom=7.5,)

    col1, col2 = st.columns(2)

    with col1:
      st.markdown(f'''
        **Customer Details** \n
        :green[Name:] {customer_first} {customer_last} \n
        :green[Job:] {customer_job} \n
        :green[Gender:] {'Male' if customer_gender =='M' else 'Female'} \n
        :green[Birthdate:] {customer_birthdate} \n
        :green[Age:] {customer_age} \n
        :green[Address:] {customer_street}, {customer_city}, {customer_state}, {customer_zip} \n
        :green[City Population:] ~{selected_city_pop} \n
      ''')     

    with col2:
      st.markdown(f'''
        **Transaction Details** \n
        :green[Transaction ID:] {selected_transaction_id[:-6]} \n
        :green[Transaction Timestamp:] {selected_trans_date} \n
        :green[Merchant Name:] {selected_merchant} \n
        :green[Category: ] {selected_category} \n
        :green[Amount: ] $ {selected_amt} \n
      ''')

      if selected_is_fraud:
        st.markdown('**:green[Status:] :red[Detected fraudulent activity]**')
      else:
        st.markdown('**:green[Status: Clear]**')

    st.title("Prediction Parameters")
    st.markdown("Adjust the parameters to observe changes in probabilities or predictions.")

    with st.container(border=True):
      col3, col4 = st.columns(2)

      with col3:
        genders = ['Female', 'Male']
        gender = st.radio(
          'Gender', genders,
          index=1 if customer_gender=='Male' else 0
        )

        age = st.number_input(
          "Age",
          min_value=18,
          max_value=100,
          value=customer_age
        )
        
        state = st.selectbox(
          "State", states,
          index=states.index(customer_state)
        )

      with col4:
        category = st.selectbox(
          "Category", categories,
          index=categories.index(selected_category)
        )

        amount = st.number_input(
          "Transaction amount",
          min_value = 0.0,
          value = selected_amt
        )
    
    input_df, input_dict = prepare_fraud_input(category, amount, age, gender, state, median_price)
    print(input_df)

    avg_probability = make_fraud_predictions(input_df, input_dict)
    print(avg_probability)


      




