"""
  SHAP Explain
  South optimal feature set
"""
import joblib
from binary_classification.tools.sop_tool import *
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

final_feature_set = ['History of Hypertension',
                     'Nephropathy',
                     'Retinopathy',
                     'History of Smoking',
                     'Age',
                     'BMI',
                     'DBP',
                     'ALT',
                     'AST',
                     'ALP',
                     'GGT',
                     'UA',
                     'TG',
                     'HDL-C',
                     'LDL-C',
                     'Ca',
                     'FT3',
                     'FT4',
                     'PINP',
                     'β-CTX']

label_name = "OP_Group"

print("================================Optimal Model=============================")
# 复现模型
X_train_t_1 = pd.read_csv("../north_review_data/X_train_t.csv")
X_train_val_1 = pd.read_csv("../north_review_data/X_train_val.csv")
y_train_t_1 = pd.read_csv("../north_review_data/y_train_t.csv")
y_train_val_1 = pd.read_csv("../north_review_data/y_train_val.csv")

X_test_1 = pd.read_csv("../north_review_data/X_test.csv")
y_test_1 = pd.read_csv("../north_review_data/y_test.csv")

# 将dataframe转换为narry
y_train_t_1 = np.array(y_train_t_1[label_name])
y_train_val_1 = np.array(y_train_val_1[label_name])
y_test_1 = np.array(y_test_1[label_name])

# save trained model
model_save_path = "../models/north_clf_model.pkl"
# load model
model = joblib.load(model_save_path)

# SHAP explain
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_1[final_feature_set])
# shap.summary_plot(shap_values, X_test_1[final_feature_set], plot_type="bar")
shap.summary_plot(shap_values[1], X_test_1[final_feature_set], plot_type="dot")
# shap.summary_plot(shap_values, X_test_1[final_feature_set],plot_type="bar",max_display=20,class_names=['Normal','Osteoporosis'])

plt.xticks(fontsize=13)
plt.savefig("../images/shap_binary_north_optimal.jpg",dpi=600)


