import streamlit as st
import pandas as pd
import numpy as np
import pickle
import utils as ut
import sklearn as sk

print('Numpy verstion: ', np.__version__)
print('Pandas verstion: ', pd.__version__)
print('Sklearn verstion: ', sk.__version__)

# client = OpenAI(
#   base_url="https://api.groq.com/openaiv1",
#   api_key = os.environ.get("OPENAI_API_KEY")
# )


def load_model(filename):
  with open(filename, 'rb') as file:
    return pickle.load(file)

# Trained models
xgboost_model = load_model('XGBClassifier.pkl')
naive_bayes_model = load_model('GaussianNB.pkl')
random_forest_model = load_model('RandomForestClassifier.pkl')
decision_tree_model = load_model('DecisionTreeClassifier.pkl')
svm_model = load_model( 'SVC.pkl')
knn_model = load_model('KNeighborsClassifier.pkl')

# Train models with feature engineering
voting_classifier_model = load_model('VotingClassifier-voting.pkl')
xgboost_SMOTE_model = load_model('XGBClassifier-SMOTE.pkl')
xgboost_featureEngineered_model = load_model('XGBClassifier-featureEngineer.pkl')

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
    'EstimatedSalary': estimated_salary
  }

  # Features Engineering
  input_dict['TenureAgeRatio'] = tenure / age if age > 0 else 0
  input_dict['CLV'] = (balance + estimated_salary) / (age + 1)  # Simplified CLV calculation

  # Age groups
  input_dict['AgeGroup_MiddleAge'] = 1 if 30 <= age < 44 else 0
  input_dict['AgeGroup_Senior'] = 1 if 45 <= age < 59 else 0
  input_dict['AgeGroup_Elderly'] = 1 if age >= 45 else 0

  input_df = pd.DataFrame([input_dict])
  return input_df, input_dict

def make_predictions(input_df, input_dict):
  # Define the expected order of features for XGBoost
  expected_order = [
      'CreditScore', 'Geography_France', 'Geography_Germany', 'Geography_Spain',
      'Gender_Male', 'Gender_Female', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
      'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'TenureAgeRatio', 'CLV',
      'AgeGroup_MiddleAge', 'AgeGroup_Senior', 'AgeGroup_Elderly'
  ]
  
  # Reorder the input DataFrame
  input_df = input_df[expected_order]

  probabilities = {
    'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
    'Naive Bayes': naive_bayes_model.predict_proba(input_df)[0][1],
    'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
    'Decision Tree' : decision_tree_model.predict_proba(input_df)[0][1],
    'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1],
    # 'SVM': svm_model.predict_proba(input_df),
    # 'Voting Classifier': voting_classifier_model.predict_proba(input_df)[0][1],
    # 'XGBoost SMOTE': xgboost_SMOTE_model.predict_proba(input_df)[0][1],
    # 'XGBoost Feature Engineered': xgboost_featureEngineered_model.predict_proba(input_df)[0][1]
  }
  
  # Calculate the average probability
  avg_probability = np.mean(list(probabilities.values()))
  print('avg_probability: ', avg_probability)

  col1, col2 = st.columns(2)

  with col1:
    fig = ut.create_gauge_chart(avg_probability)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"The customer has a {avg_probability:.2%} probability of churning.")

  with col2:
    fig_probs = ut.create_model_probability_chart(probabilities)
    st.plotly_chart(fig_probs, use_container_width=True)

  st.markdown("### Model Probabilities")
  for model, prob in probabilities.items():
    st.write(f"{model}: {prob:.2f}")
    st.write(f"Average Probability: {avg_probability:.2f}")

  return avg_probability

# def explain_prediction(probability, input_dict, surname):
#   systemPrompt = f"""
#   You are an expert data scientist at a bank, wehre you specialize in interpreting
#   and explaining predictions of machine learning models.

#   Your machine learning model has predicted that a customer named {surname} has a
#   {probability * 100}% probablity of churning, based on the information provided below.

#   Here is the customer's information:
#   {input_dict}

#   Here are the machine learning model's top 10 most important features for predciting churn:


#   features	          importance
#   NumOfProducts	      0.323888
#   IsActiveMember	    0.164146
#   Age	                0.109550
#   Geography_Germany	  0.091373
#   Balance	            0.052786
#   Geography_France	  0.046463
#   Gender_Female  	    0.045283
#   Geography_Spain	    0.036855
#   CreditScore	        0.035005
#   EstimatedSalary	    0.032655
#   HasCrCard	          0.031940
#   Tenure	            0.030054
#   Gender_Male	        0.000000

#   {pd.set_option('display.max_columns', None)}

#   Here are the summary statistics for churned customers:
#   {df[df['Exited'] == 1].describe()}

#   Here are summary statistics for non-churned customers:
#   {df[df['Exited'] == 0].describe()}

#   - If the customer has over a 40% risk of churning, genearte a 3 sentence explanation of
#   why they are at risk of churning.
#   - If the customer has less than a 40$ risk of churning, generate a 3 sentence explanation
#   of why they might not be at risk of churning.
#   - Your explanation should be based on the customer's information, the summary statitics
#   of churned and non-churned customers, and the feature importances provided.

#   Don't mention the probability of churning, or the machine learning model,
#   or say anything like "Based on the machine learning model's predictions and top 10 most
#   important features, just explain the prediction.

#   """

#   print("EXPLANATION PROMPT", systemPrompt)

#   raw_response = client.chat.completions.create(
#     model='gpt-4o mini',
#     messages=[{
#       'role': 'user',
#       'content': systemPrompt
#     }]
#   )

#   return raw_response.choices[0].message.content

# def generate_email(probability, input_dict, explanation, surname):
#   systemPrompt = f"""
#   You are a manager at HS Bank. You are responsible for ensuring customers stay
#   with the bank and are incentivized with various offers.

#   You noticed a customer named {surname} has a {probability * 100}% probability of churning.
#   Here is the customer's information:
#   {input_dict}

#   Here is the explanation of the customer's churning probability:
#   {explanation}

#   Generate an email to the customer based on their information,
#   asking them to stay if they are at risk of churning, or offerign them incentives
#   so taht they become more loyal to the bank.

#   Make sure to list out a set of incentives to stay based on their information,
#   in bullet point format. Don't ever mention the probability of churning, or
#   the machine learning model to the customer.
#   """

#   raw_response = client.chat.completions. create(
#     model='gpt-4o mini',
#     messages=[{
#       'role': 'user',
#       'content': systemPrompt
#     }]
#   )

#   print("\n\nEMAIL PROMPT", systemPrompt)
  
#   return raw_response.choices[0].message.content


st.title("Customer Churn Prediction")

df = pd.read_csv("churn.csv")

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

  # explanation = explain_prediction(avg_probability, input_dict, customer_surname)
  # st.markdown('---')
  # st.subheader('Explanation of Prediction')
  # st.markdown(explanation)
  # email = generate_email(avg_probability, input_dict, explanation, customer_surname)
  # st.markdown('---')
  # st.subheader('Personalized Email')
  # st.markdown(email)