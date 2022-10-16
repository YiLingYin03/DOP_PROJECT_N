"""
Description: Optimal model
Dataset : Northern dataset
Input feature : the optimal feature set
Classification : binary classification
"""
import joblib

from binary_classification.tools.opUtils import modelling_revised_calibration
from binary_classification.tools.sop_tool import *

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

print("================================North optimal Model=============================")
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

# ==========================
# 1.重新实例化模型（与之前使用的基础分类器类型一致）

final_model = RandomForestClassifier(n_estimators=25, criterion='gini', random_state=42)

# 得到最终模型的Accuracy，precision，recall，f1 score，auc 以及 confuse matricx
clf_model, cal_clf_model = modelling_revised_calibration(
    X_train=X_train_t_1[final_feature_set],
    X_train_val=X_train_val_1[final_feature_set],
    y_train=y_train_t_1,
    y_train_val=y_train_val_1,
    X_test=X_test_1[final_feature_set],
    y_test=y_test_1,
    model=final_model
)

# save trained model
model_save_path = "../models/north_clf_model.pkl"
joblib.dump(clf_model, model_save_path)
print("================================END=============================")
