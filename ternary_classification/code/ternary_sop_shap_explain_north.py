"""

Description: XGBoost + SHAP Explain model
Dataset : South dataset
Input feature : 34
Classification : ternary classification
"""

import pandas as pd
import shap
import xgboost
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)

num_columns = ['Age', 'Height', 'Weight', 'BMI',
               'SBP', 'DBP', 'Heart Rate', 'FBG',
               'HbA1c', 'ALT', 'AST', 'ALP', 'GGT',
               'UA', 'TC', 'TG', 'HDL-C', 'LDL-C',
               'Ca', 'P', 'FT3', 'FT4', 'VD3', 'N-MID', 'PINP', 'β-CTX']

cat_columns = ["Gender", "Macrovascular Complications",
               "History of Hypertension", "Nephropathy",
               "Retinopathy", "Neuropathy",
               "History of Smoking", "History of Drinking"]
label_name = "OP_Group"
# =====================================================================

X = pd.read_csv("../north_train_data/feature_data.csv")
label = pd.read_csv("../north_train_data/label_data.csv")

# change the label to fit models
label[label[label_name] == 1] = 0 # Normal
label[label[label_name] == 2] = 1 # Osteopenia
label[label[label_name] == 3] = 2 # Osteporosis
y = label.to_numpy()

# SHAP for mulitclass classification
model = xgboost.XGBClassifier(objective="binary:logistic", max_depth=5, n_estimators=30).fit(X, y)

explainer = shap.Explainer(model)
shap.plots.initjs()
shap_values = shap.TreeExplainer(model).shap_values(X)

shap.summary_plot(shap_values, X,plot_type="bar",max_display=20,class_names=['Normal','Osteopenia','Osteoporosis'])
plt.xlabel("mean|SHAP Value|")
plt.xticks(fontsize=13)

plt.savefig("../images/top20_mean_shap_ternary_north.jpg",dpi=600)


