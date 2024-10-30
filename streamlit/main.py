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
  path = "https://media.githubusercontent.com/media/itancio/churn/refs/heads/main/streamlit/churn.csv"
  df = pd.read_csv(path)

  customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

  selected_customer_option = st.selectbox('Select a customer', customers)

  if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split(" - ")[0])
    selected_surname = selected_customer_option.split(" - ")[1]
    selected_customer = df.loc[df['CustomerId'] == selected_customer_id].to_dict(
      orient='records')

    customer_surname = selected_customer[0]['Surname']
    customer_credit_score = selected_customer[0]['CreditScore']
    customer_location = selected_customer[0]['Geography']
    customer_gender = selected_customer[0]['Gender']
    customer_age = selected_customer[0]['Age']
    customer_tenure = selected_customer[0]['Tenure']

    customer_balance = selected_customer[0]['Balance']
    customer_num_products = selected_customer[0]['NumOfProducts']
    customer_has_credit_card = selected_customer[0]['HasCrCard']
    customer_is_active_member = selected_customer[0]['IsActiveMember']
    customer_estimated_salary = selected_customer[0]['EstimatedSalary']

    col1, col2 = st.columns(2)

    with col1:
      credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=customer_credit_score)

      locations = ['Spain', 'France', 'Germany']
      
      location = st.selectbox(
        "Location", locations,
        index=locations.index(customer_location))

      genders = ['Male', 'Female']
      
      gender = st.radio('Gender', genders,
                      index=0 if customer_gender=='Male' else 1)

      age = st.number_input(
        'Age',
        min_value=18,
        max_value=100,
        value=customer_age
      )

      tenure = st.number_input(
        'Tenure (years)',
        min_value=0,
        max_value=50,
        value=customer_tenure
      )

    with col2:
      balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=customer_balance
      )
    
      estimated_salary = st.number_input(
        'Estimated Salary',
        min_value=0.0,
        value=customer_estimated_salary
      )
      
      num_products = st.number_input(
        "Number of products",
        min_value=0,
        max_value=10,
        value=customer_num_products
      )
    
      has_credit_card = st.checkbox(
        'Has Credit Card',
        value=customer_has_credit_card
      )
      
      is_active_member = st.checkbox(
        "Is Active Member",
        value=customer_is_active_member
      )

    input_df, input_dict = prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary)
    print(input_df)

    avg_probability = make_predictions(input_df, input_dict)
    print(avg_probability)

    st.markdown('---')
    st.subheader('Explanation of Prediction')
    explanation = explain_prediction(avg_probability, input_dict, customer_surname)
    st.markdown(explanation)

    st.markdown('---')
    st.subheader('Personalized Email')
    email = generate_email(avg_probability, input_dict, explanation, customer_surname)
    st.markdown(email)




################################################################################################################################

# Load models
xgboost_model = load_model('models_fraud/XGBClassifier-featured.pkl')
naive_bayes_model = load_model('models_fraud/GaussianNB-featured.pkl')
random_forest_model = load_model('models_fraud/RandomForestClassifier-featured.pkl')
decision_tree_model = load_model('models_fraud/DecisionTreeClassifier-featured.pkl')


def prepare_fraud_input(category, amount, age, gender, state, median_price):
  def group_age(age):
    if 0 <= age <= 31:
      return 0
    elif 32 <= age <= 47:
      return 1
    elif 48 <= age <= 75:
      return 2
    else:
      return 3

  input_dict = {
    'amt' : amount,
    'age' : age,
    'price_ratio_to_median' : amount / median_price,
    'category_codes' : int(category.split(' - ')[0]),
    'state_codes' : int(state.split(' - ')[0]),
    'ageGroup_codes' : group_age(age),
    'gender_codes' : int(gender.split(' - ')[0]),
  }

  input_df = pd.DataFrame([input_dict])
  return input_df, input_dict

def make_fraud_predictions(input_df, input_dict):
  # Make predictions
  xgb_predict = xgboost_model.predict_proba(input_df)[0][1],
  nb_predict = naive_bayes_model.predict_proba(input_df)[0][1],
  rf_predict = random_forest_model.predict_proba(input_df)[0][1],
  dt_predict = decision_tree_model.predict_proba(input_df)[0][1],

  xgb_predict = xgb_predict[0]
  nb_predict = nb_predict[0]
  rf_predict = rf_predict[0]
  dt_predict = dt_predict[0]
  print(xgb_predict,nb_predict, rf_predict,dt_predict)

  probabilities = {}

  # Filter out predictions that are very close to zero
  min_threshold = 0.0000001
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
    fig = ut.create_gauge_chart(avg_probability, 'Fraud')
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"The transaction has a {avg_probability:.2%} probability of fraudulence.")

    st.markdown("### Model Probabilities")
    col1_1, col1_2 = st.columns(2)
    for model, prob in probabilities.items():
      col1_1.write(f"{model}: ")
      col1_2.write(f"{prob * 100:.4f}%")

  with col2:
    fig_probs = ut.create_model_probability_chart(probabilities, 'Fraud')
    st.plotly_chart(fig_probs, use_container_width=True)

  st.markdown(f"### Average Probability: {avg_probability * 100:.4f}%")

  return avg_probability

def explain_fraud_prediction(probability, input_dict, surname):
  systemPrompt = f"""
  You are an expert data scientist at a bank, wehre you specialize in interpreting
  and explaining predictions of machine learning models.

  Your machine learning model has predicted that a customer named {surname} has a
  {probability * 100}% probablity of fraudulence, based on the information provided below.

  Here is the customer's information:
  {input_dict}

  Here are the machine learning model's top 10 most important features for predicting fraud:

  features	          importance
  amt	                  0.297064
  price_ratio_to_median	0.270963
  category_codes	      0.249900
  age	                  0.080951
  state_codes	          0.072448
  gender_codes	        0.016715
  ageGroup_codes	      0.011958
  

  {pd.set_option('display.max_columns', None)}

  Here are the summary statistics for fraud transactions:
  {df[df['is_fraud'] == 1].describe()}

  Here are summary statistics for non-fraud transactions:
  {df[df['is_fraud'] == 0].describe()}

  - If the transaction has over a 40% risk of fraud, generate a 3 sentence explanation of
  why they are at risk of fraud.
  - If the transaction has less than a 40$ risk of fraud, generate a 3 sentence explanation
  of why they might not be at risk of fraud.
  - Your explanation should be based on the transactions's information, the summary statitics
  of fraud and non-fraud transactions, and the feature importances provided.

  Don't mention the probability of fraud, or the machine learning model,
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

def generate_fraud_email(probability, input_dict, explanation, surname):
  systemPrompt = f"""
  You are a manager at HS Bank, responsible for ensuring the safety and security of customer transactions. 

  You have identified a potential risk with a customer named {surname}, who may be at risk of fraudulent activity or unusual transaction behavior. Here is the customer's information:
  {input_dict}

  Here is a detailed analysis of the potential risks associated with this customer:
  {explanation}

  Generate an email to the customer to inform them of potential risks, provide clear advice on how to proceed, and offer reassurance that the bank is actively working to safeguard their account. 

  The email should include:
  - A polite notification of unusual activity or potential risk.
  - Clear advice on immediate actions the customer can take to protect their account (e.g., updating passwords, reviewing recent transactions).
  - Reassurance that HS Bank has implemented measures to mitigate risks, such as monitoring the account more closely or providing additional security options.
  - Encourage the customer to contact support if they notice any unauthorized transactions.

  Make sure to use reassuring language, maintain a helpful tone, and never disclose the probability of risk detection or the use of a risk model to the customer.
  Don't ever mention the probability of fraud, or
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
  #   'is_fraud': 0,
  #   'trans_year' : 2019, 
  #   'dob_year' : 1978, 
  #   'age' : 31, 
  #   'ageGroup' : 'Early',
  #   'price_ratio_to_median': 4.97 / 4.97, 
  #   'category_codes': 8, 
  #   'state_codes': 27, 
  #   'ageGroup_codes': 0, 
  #   'gender_codes': 0
  # }])

  # transactions = sample['trans_num'] + ' - ' + sample['last']

  # Load dataset
  path = "https://media.githubusercontent.com/media/itancio/churn/refs/heads/main/notebook/fraud/modified_df.csv"
  df = pd.read_csv(path)

  # Needed for the map
  df = df.rename(columns={'long' : 'lon'})

  mappings = {
    'ageGroup': {
      0: 'Early', 
      1: 'MiddleAge', 
      2: 'Senior', 
      3: 'Elderly'},
    'category': {
      0: 'entertainment',
      1: 'food_dining',
      2: 'gas_transport',
      3: 'grocery_net',
      4: 'grocery_pos',
      5: 'health_fitness',
      6: 'home',
      7: 'kids_pets',
      8: 'misc_net',
      9: 'misc_pos',
      10: 'personal_care',
      11: 'shopping_net',
      12: 'shopping_pos',
      13: 'travel'},
    'gender': {
      0: 'F', 
      1: 'M'},
    'state': {
      0: 'AK',
      1: 'AL',
      2: 'AR',
      3: 'AZ',
      4: 'CA',
      5: 'CO',
      6: 'CT',
      7: 'DC',
      8: 'DE',
      9: 'FL',
      10: 'GA',
      11: 'HI',
      12: 'IA',
      13: 'ID',
      14: 'IL',
      15: 'IN',
      16: 'KS',
      17: 'KY',
      18: 'LA',
      19: 'MA',
      20: 'MD',
      21: 'ME',
      22: 'MI',
      23: 'MN',
      24: 'MO',
      25: 'MS',
      26: 'MT',
      27: 'NC',
      28: 'ND',
      29: 'NE',
      30: 'NH',
      31: 'NJ',
      32: 'NM',
      33: 'NV',
      34: 'NY',
      35: 'OH',
      36: 'OK',
      37: 'OR',
      38: 'PA',
      39: 'RI',
      40: 'SC',
      41: 'SD',
      42: 'TN',
      43: 'TX',
      44: 'UT',
      45: 'VA',
      46: 'VT',
      47: 'WA',
      48: 'WI',
      49: 'WV',
      50: 'WY'}
    }

  # Create lists
  jobs = sorted(list(df['job'].unique()))
  merchants = sorted(list(merch.split('fraud_')[1] for merch in df['merchant'].unique()))
  categories = [f'{k} - {v}' for k, v in mappings['category'].items()]
  ageGroups = [f'{k} - {v}' for k, v in mappings['ageGroup'].items()]
  genders = [f'{k} - {v}' for k, v in mappings['gender'].items()]
  states = [f'{k} - {v}' for k, v in mappings['state'].items()]

  median_price = df['amt'].median()


  transactions = [f"{row['last']}, {row['first']} - {row['trans_num']}" for _, row in df.iterrows()]
  selected_transaction_option = st.selectbox('Select a transaction', transactions)

  if selected_transaction_option:
    selected_transaction_id = selected_transaction_option.split(" - ")[1]
    selected_transaction = df.loc[df['trans_num'] == selected_transaction_id].to_dict(orient='records')

    customer_first = selected_transaction[0]['first']
    customer_last = selected_transaction[0]['last']
    customer_gender = selected_transaction[0]['gender']
    customer_job = selected_transaction[0]['job']
    customer_age = selected_transaction[0]['age']
    customer_birthdate = selected_transaction[0]['dob']
    customer_street = selected_transaction[0]['street']
    customer_city = selected_transaction[0]['city']
    customer_state = f"{selected_transaction[0]['state_codes']} - {selected_transaction[0]['state']}"
    customer_zip = str(selected_transaction[0]['zip'])
    customer_lat = selected_transaction[0]['lat']
    customer_long = selected_transaction[0]['lon']

    selected_city_pop = selected_transaction[0]['city_pop']
    selected_merchant = selected_transaction[0]['merchant'].split('fraud_')[1]
    selected_merchant_lat = selected_transaction[0]['merch_lat']
    selected_merchant_long = selected_transaction[0]['merch_long']
    selected_trans_date = selected_transaction[0]['trans_date_trans_time']
    selected_category = f"{selected_transaction[0]['category_codes']} - {selected_transaction[0]['category']}"
    selected_amt = selected_transaction[0]['amt']
    selected_is_fraud = selected_transaction[0]['is_fraud']


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
        gender = st.radio(
          'Gender', genders,
          index=1 if customer_gender=='1 - Male' else 0
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

    print('selected: ', gender, age, state, category, amount)
    
    fraud_input_df, fraud_input_dict = prepare_fraud_input(category, amount, age, gender, state, median_price)
    print(input_df)

    fraud_avg_probability = make_fraud_predictions(fraud_input_df, fraud_input_dict)
    print(fraud_avg_probability)

    st.markdown('---')
    st.subheader('Explanation of Prediction')
    explanation = explain_fraud_prediction(fraud_avg_probability, fraud_input_dict, customer_last)
    st.markdown(explanation)

    st.markdown('---')
    st.subheader('Personalized Email')
    email = generate_fraud_email(avg_probability, input_dict, explanation, customer_surname)
    st.markdown(email)


      




