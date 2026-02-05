import pandas as pd

#read code
df = pd.read_csv('fast_food.csv')
#print first 5 rows
print(df.head())

df.info()

#Checking for missing values
print(df.isnull().sum())

#getting detailed statistics
print(df.describe())


#Removing Gender_other from the dataframe to avoid dummy variable trap

df =df[df['Gender'].isin(['Male', 'Female'])].copy()
print(df['Gender'].value_counts())

#Encoding Gender
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
print(df['Gender'].isnull().sum())

#Binary Target BMI > 25
df['Target_Variable'] = (df['BMI'] > 25).astype(int)


#Models : Logistic Regression, Decision Tree, Random Forest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Features and Target variable
#Binary Target BMI > 25
df['BMI_binary'] = (df['BMI'] > 25).astype(int)
Y = df['BMI_binary']
# Features
X = df.drop(columns=['BMI', 'BMI_binary', 'Target_Variable'])
Y = df['BMI_binary']

# Categorical and Numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist() 
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing pipelines for both numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numerical_cols),
])

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Processing
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Logistic Regression
log_reg = LogisticRegression(max_iter=3000)
log_reg.fit(X_train, Y_train)

log_reg_train_pred = log_reg.predict(X_train)
log_reg_test_pred = log_reg.predict(X_test)
log_reg_test_proba = log_reg.predict_proba(X_test)[:, 1]

print("Logistic Regression Train Accuracy:", accuracy_score(Y_train, log_reg_train_pred))
print("Logistic Regression Test Accuracy:", accuracy_score(Y_test, log_reg_test_pred))
print("Logistic Regression Test ROC-AUC:", roc_auc_score(Y_test, log_reg_test_proba))


print("\nClassification Report (TEST):")
print(classification_report(Y_test, log_reg_test_pred))

#test for errors becuase accuracy is 100%

print("FEATURES:")
print(X.columns.tolist())

suspicious = [
    c for c in X.columns
    if any(k in c.lower() for k in ["weight", "obese", "bmi", "category", "class", "risk"])
]
print("SUSPICIOUS FEATURES:", suspicious)

# Random Forrest Classification
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)
rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)
rf_test_proba = rf_model.predict_proba(X_test)[:, 1]

print("Random Forest Train Accuracy:", accuracy_score(Y_train, rf_train_pred))
print("Random Forest Test Accuracy:", accuracy_score(Y_test, rf_test_pred))
print("Random Forest Test ROC-AUC:", roc_auc_score(Y_test, rf_test_proba))

print("\nClassification Report (TEST):")
print(classification_report(Y_test, rf_test_pred))

# Decision Tree Classification
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, Y_train)  
dt_train_pred = dt_model.predict(X_train)
dt_test_pred = dt_model.predict(X_test)
dt_test_proba = dt_model.predict_proba(X_test)[:, 1]

print("Decision Tree Train Accuracy:", accuracy_score(Y_train, dt_train_pred))
print("Decision Tree Test Accuracy:", accuracy_score(Y_test, dt_test_pred))
print("Decision Tree Test ROC-AUC:", roc_auc_score(Y_test, dt_test_proba))

print("\nClassification Report (TEST):")
print(classification_report(Y_test, dt_test_pred))


# Model Performance Comparison
models = ['Random Forest', 'Logistic Regression', 'Decision Tree']

train_accuracies = [accuracy_score(Y_train, rf_train_pred),
                     accuracy_score(Y_train, log_reg_train_pred),
                        accuracy_score(Y_train, dt_train_pred)]

test_accuracies = [accuracy_score(Y_test, rf_test_pred),
                    accuracy_score(Y_test, log_reg_test_pred),
                        accuracy_score(Y_test, dt_test_pred)]

roc_aucs = [roc_auc_score(Y_test, rf_test_proba),
              roc_auc_score(Y_test, log_reg_test_proba),    
                    roc_auc_score(Y_test, dt_test_proba)]

results_df = pd.DataFrame({
    'Model': models,
    'Train Accuracy': [f"{x:.4f}" for x in train_accuracies],
    'Test Accuracy': [f"{x:.4f}" for x in test_accuracies],
    'ROC-AUC': [f"{x:.4f}" for x in roc_aucs]
})

#Table of results   
print("\nModel Performance Comparison:")
print(results_df)

# Average BMI by # of times eating fast food per week
df = pd.read_csv('fast_food.csv')
avg_bmi = df.groupby('Fast_Food_Meals_Per_Week')['BMI'].mean().reset_index()

print("\nAverage BMI by Fast Food Consumption per Week:")
print(avg_bmi)

