"""
    compare with other classifiers (use the selected optimal feature set)
    optimal set (for paper) based on random forest:

"""
import sys

from ternary_classification.tools.opUtils import modelling_revised_calibration
from ternary_classification.tools.sop_tool import *


# ========================== 将consloe打印输出保存到log文件 ==========================

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# 将控制台的结果输出到XXX.log文件
t = round(time.time())
log_name = "../logs/results_comparison_full_north" + str(t) + ".log"
sys.stdout = Logger(log_name, sys.stdout)
# sys.stderr = Logger('results.log_file', sys.stderr)
# =====================================================================

print("================================Full Model review Comparison=============================")

final_feature_set = ['Age', 'Height', 'Weight', 'BMI',
               'SBP', 'DBP', 'Heart Rate', 'FBG',
               'HbA1c', 'ALT', 'AST', 'ALP', 'GGT',
               'UA', 'TC', 'TG', 'HDL-C', 'LDL-C',
               'Ca', 'P', 'FT3', 'FT4', 'VD3', 'N-MID', 'PINP', 'β-CTX',
               "Gender", "Macrovascular Complications",
               "History of Hypertension", "Nephropathy",
               "Retinopathy", "Neuropathy",
               "History of Smoking", "History of Drinking"]

label_name = "OP_Group"

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

"""
    rebuild the random forest model
"""

final_model = RandomForestClassifier(n_estimators=25,criterion='gini',random_state=42)

# 得到最终模型的Accuracy，precision，recall，f1 score，auc 以及 confuse matricx
rf_selected_columns = modelling_revised_calibration(
                                X_train=X_train_t_1[final_feature_set],
                                X_train_val=X_train_val_1[final_feature_set],
                                y_train=y_train_t_1,
                                y_train_val=y_train_val_1,
                                X_test=X_test_1[final_feature_set],
                                y_test=y_test_1,
                                model=final_model,
                                name='north'
                            )

print("===============================Multinomial Logistic regression=============================")

lr_model = LogisticRegression(random_state=0,multi_class='multinomial')

# 得到最终模型的Accuracy，precision，recall，f1 score，auc 以及 confuse matricx
selected_columns = modelling_revised_calibration(
                                X_train=X_train_t_1[final_feature_set],
                                X_train_val=X_train_val_1[final_feature_set],
                                y_train=y_train_t_1,
                                y_train_val=y_train_val_1,
                                X_test=X_test_1[final_feature_set],
                                y_test=y_test_1,
                                model=lr_model,
                                name='north'
                            )

print("===============================KNN=============================")

knn_model = KNeighborsClassifier(n_neighbors=3)

# 得到最终模型的Accuracy，precision，recall，f1 score，auc 以及 confuse matricx
knn_selected_columns = modelling_revised_calibration(
                                X_train=X_train_t_1[final_feature_set],
                                X_train_val=X_train_val_1[final_feature_set],
                                y_train=y_train_t_1,
                                y_train_val=y_train_val_1,
                                X_test=X_test_1[final_feature_set],
                                y_test=y_test_1,
                                model=knn_model,
                                name='north'
                            )

print("===============================SVM (one vs all)=============================")
svm_base_model = SVC(C=15, gamma='auto', kernel='rbf', random_state=0)
svm_model = OneVsRestClassifier(svm_base_model)

# 得到最终模型的Accuracy，precision，recall，f1 score，auc 以及 confuse matricx
svmovr_selected_columns = modelling_revised_calibration(
                                X_train=X_train_t_1[final_feature_set],
                                X_train_val=X_train_val_1[final_feature_set],
                                y_train=y_train_t_1,
                                y_train_val=y_train_val_1,
                                X_test=X_test_1[final_feature_set],
                                y_test=y_test_1,
                                model=svm_model,
                                name='north'
                            )

print("===============================DT=============================")

dt_model = DecisionTreeClassifier(random_state=0, max_depth=5)

# 得到最终模型的Accuracy，precision，recall，f1 score，auc 以及 confuse matricx
dt_selected_columns = modelling_revised_calibration(
                                X_train=X_train_t_1[final_feature_set],
                                X_train_val=X_train_val_1[final_feature_set],
                                y_train=y_train_t_1,
                                y_train_val=y_train_val_1,
                                X_test=X_test_1[final_feature_set],
                                y_test=y_test_1,
                                model=dt_model,
                                name='north'
                            )

print("==============================END=============================")
