from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

app = Flask(__name__)

# Load and preprocess the data
data = pd.read_csv('diabetic_data.csv')
columns_to_drop = ['patient_nbr', 'encounter_id', 'weight', 'payer_code', 'medical_specialty', 'examide', 'citoglipton']
data.drop(columns=columns_to_drop, inplace=True)
data.replace('?', pd.NA, inplace=True)
data.dropna(subset=['diag_1', 'diag_2', 'diag_3'], inplace=True)

def fillna_with_random(data, column):
    non_null_values = data[column].dropna().values
    num_missing = data[column].isna().sum()
    random_choices = np.random.choice(non_null_values, num_missing)
    data.loc[data[column].isna(), column] = random_choices

fillna_with_random(data, 'race')
data['max_glu_serum'].fillna('None', inplace=True)
data['A1Cresult'].fillna('None', inplace=True)

def map_icd9_to_category(icd9_code):
    if pd.isna(icd9_code):
        return 'Missing'
    code_prefix = icd9_code.split('.')[0]
    icd9_mapping = {
        '001-139': 'Infectious And Parasitic Diseases',
        '140-239': 'Neoplasms',
        '240-279': 'Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders',
        '280-289': 'Diseases Of The Blood And Blood-Forming Organs',
        '290-319': 'Mental Disorders',
        '320-389': 'Diseases Of The Nervous System And Sense Organs',
        '390-459': 'Diseases Of The Circulatory System',
        '460-519': 'Diseases Of The Respiratory System',
        '520-579': 'Diseases Of The Digestive System',
        '580-629': 'Diseases Of The Genitourinary System',
        '630-679': 'Complications Of Pregnancy, Childbirth, And The Puerperium',
        '680-709': 'Diseases Of The Skin And Subcutaneous Tissue',
        '710-739': 'Diseases Of The Musculoskeletal System And Connective Tissue',
        '740-759': 'Congenital Anomalies',
        '760-779': 'Certain Conditions Originating In The Perinatal Period',
        '780-799': 'Symptoms, Signs, And Ill-Defined Conditions',
        '800-999': 'Injury And Poisoning',
        'V01-V91': 'Supplementary Classification Of Factors Influencing Health Status And Contact With Health Services',
        'E000-E999': 'Supplementary Classification Of External Causes Of Injury And Poisoning'
    }
    for range_, category in icd9_mapping.items():
        start, end = range_.split('-')
        if start <= code_prefix <= end:
            return category
    return 'Other'

for col in ['diag_1', 'diag_2', 'diag_3']:
    data[col] = data[col].apply(map_icd9_to_category)

medicine_columns = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
                    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
                    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
                    'insulin', 'glyburide-metformin', 'glipizide-metformin',
                    'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']

for col in medicine_columns:
    data[col] = data[col].apply(lambda x: 'Yes' if x in ['Up', 'Down'] else 'No' if x in ['Steady', 'None'] else x)

data['diabetes_Meds_count'] = data[medicine_columns].apply(lambda row: row.isin(['Yes']).sum(), axis=1)
data.drop(columns=medicine_columns, inplace=True)
data['all_med_count'] = data['num_medications'] + data['diabetes_Meds_count']
data.drop(columns=['num_medications'], inplace=True)
data['total_num_procedures'] = data['num_lab_procedures'] + data['num_procedures']
data.drop(columns=['num_lab_procedures', 'num_procedures'], inplace=True)
data['number_visit'] = data[['number_outpatient', 'number_emergency', 'number_inpatient']].sum(axis=1)
data.drop(columns=['number_outpatient', 'number_emergency', 'number_inpatient'], inplace=True)
data.rename(columns={'number_diagnoses': 'num_comorbidity'}, inplace=True)

label_encoder = LabelEncoder()
categorical_columns = ['race', 'gender', 'age', 'admission_type_id', 'max_glu_serum', 'A1Cresult', 'diag_1', 'diag_2', 'diag_3', 'change', 'diabetesMed']
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

data['readmitted'] = data['readmitted'].replace({'<30': 0, '>30': 0, 'NO': 1})
data['readmitted'] = label_encoder.fit_transform(data['readmitted'])

X = data[['all_med_count', 'time_in_hospital', 'total_num_procedures', 'diabetes_Meds_count', 'change', 
          'num_comorbidity', 'diabetesMed', 'number_visit', 'age']]
y = data['readmitted']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

random_forest_model = RandomForestClassifier()
knn_model = KNeighborsClassifier()
xgb_model = XGBClassifier()
catboost_model = CatBoostClassifier(verbose=0)

models = {
    'random_forest': random_forest_model,
    'knn': knn_model,
    'xgboost': xgb_model,
    'catboost': catboost_model
}

for model_name, model in models.items():
    model.fit(X_resampled, y_resampled)



@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [data[col] for col in ['all_med_count', 'time_in_hospital', 'total_num_procedures', 'diabetes_Meds_count', 'change', 
                                      'num_comorbidity', 'diabetesMed', 'number_visit', 'age']]
    features = np.array(features).reshape(1, -1)
    
    predictions = {}
    for model_name, model in models.items():
        prediction = model.predict(features)
        predictions[model_name] = int(prediction[0])
    
    return jsonify(predictions)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(status="UP"), 200

if __name__ == '__main__':
    app.run(debug=True)



#  curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{
#     "all_med_count": 2,
#     "time_in_hospital": 3,
#     "total_num_procedures": 5,
#     "diabetes_Meds_count": 1,
#     "change": 0,
#     "num_comorbidity": 4,
#     "diabetesMed": 1,
#     "number_visit": 2,
#     "age": 6
# }'