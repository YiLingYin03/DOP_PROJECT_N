"""
Description: XGBoost + SHAP Explain model
Dataset : South dataset
Input feature : the optimal feature set
Classification : ternary classification
"""
import joblib
from ternary_classification.tools.sop_tool import *
import shap
import matplotlib.pyplot as plt
from matplotlib import rcParams

np.random.seed(42)

config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "font.size": 20,
    "mathtext.fontset":'stix',
}
rcParams.update(config)

print("================================SHAP Explain=============================")
# load data
X_train_t_1 = pd.read_csv("../north_review_data/X_train_t.csv")
X_train_val_1 = pd.read_csv("../north_review_data/X_train_val.csv")
y_train_t_1 = pd.read_csv("../north_review_data/y_train_t.csv")
y_train_val_1 = pd.read_csv("../north_review_data/y_train_val.csv")

X_test_1 = pd.read_csv("../north_review_data/X_test.csv")
y_test_1 = pd.read_csv("../north_review_data/y_test.csv")

final_model_features = ['Neuropathy',
                        'History of Drinking',
                        'Age',
                        'Weight',
                        'SBP',
                        'DBP',
                        'Heart Rate',
                        'ALT',
                        'AST',
                        'ALP',
                        'UA',
                        'TC',
                        'HDL-C',
                        'LDL-C',
                        'P',
                        'FT3',
                        'FT4',
                        'VD3',
                        'N-MID',
                        'β-CTX']

label_name = "OP_Group"

# change the label to fit models
y_train_t_1[y_train_t_1[label_name] == 1] = 0 # Normal
y_train_t_1[y_train_t_1[label_name] == 2] = 1 # Osteopenia
y_train_t_1[y_train_t_1[label_name] == 3] = 2 # Osteporosis

y_train_val_1[y_train_val_1[label_name] == 1] = 0 # Normal
y_train_val_1[y_train_val_1[label_name] == 2] = 1 # Osteopenia
y_train_val_1[y_train_val_1[label_name] == 3] = 2 # Osteporosis

y_test_1[y_test_1[label_name] == 1] = 0 # Normal
y_test_1[y_test_1[label_name] == 2] = 1 # Osteopenia
y_test_1[y_test_1[label_name] == 3] = 2 # Osteporosis

# 将dataframe转换为narry
y_train_t_1 = np.array(y_train_t_1[label_name])
y_train_val_1 = np.array(y_train_val_1[label_name])
y_test_1 = np.array(y_test_1[label_name])

# save trained model
model_save_path = "../models/north_clf_model.pkl"
# load model
model = joblib.load(model_save_path)

# SHAP explain
explainer = shap.Explainer(model)
shap_values = shap.TreeExplainer(model).shap_values(X_test_1[final_model_features])
shap.summary_plot(shap_values, X_test_1[final_model_features],plot_type="bar",max_display=20,class_names=['Normal','Osteopenia','Osteoporosis'])
plt.xlabel("mean|SHAP Value|")
plt.xticks(fontsize=13)
plt.savefig("../images/shap_ternary_north_optimal.jpg",dpi=600)
print("================================END=============================")

