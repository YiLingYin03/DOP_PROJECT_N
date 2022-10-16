"""
Step: 2
Description: hybrid method
Dataset : North dataset
Classification : ternary classification
"""

import sys
from ternary_classification.tools.opUtils import modelling_revised_calibration
from ternary_classification.tools.utils import *
from ternary_classification.tools.sop_tool import *

np.random.seed(42)
def ternary_sop_model_north(population_size,num_features,num_generation,drop_rate,mutation_rate,error_tolerance,model):

    # ========================== console log ==========================

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
    log_name = "../logs/results_north"+str(t)+".log"
    sys.stdout = Logger(log_name, sys.stdout)
    # sys.stderr = Logger('results.log_file', sys.stderr)
    # =====================================================================

    isPlot = True

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

    columns = np.concatenate((cat_columns,num_columns))


    # read preprocessing data
    feature_data_path = "../north_train_data/feature_data.csv"
    feature_data = pd.read_csv(feature_data_path)
    label_data_path = "../north_train_data/label_data.csv"
    label_data = pd.read_csv(label_data_path)
    # split data
    X_train, X_test, y_train, y_test = train_test_split(feature_data,label_data,test_size=0.2,random_state=42)
    X_train_t, X_train_val, y_train_t, y_train_val = train_test_split(X_train, y_train,
                                                                      test_size=0.5, random_state=42)
    # saving data
    X_train_t.to_csv("../north_review_data/X_train_t.csv", sep=',', index=False)
    X_train_val.to_csv("../north_review_data/X_train_val.csv", sep=',', index=False)
    y_train_t.to_csv("../north_review_data/y_train_t.csv", sep=',', index=False)
    y_train_val.to_csv("../north_review_data/y_train_val.csv", sep=',', index=False)
    X_test.to_csv("../north_review_data/X_test.csv", sep=',', index=False)
    y_test.to_csv("../north_review_data/y_test.csv", sep=',', index=False)

    # dataframe to narry
    y_train_t = np.array(y_train_t[label_name])
    y_train_val = np.array(y_train_val[label_name])
    y_test = np.array(y_test[label_name])

    # ==========================
    # SigCRF-NSGA-II-RS
    features, top_object, front_0_num = nsga_tool.generations_v2(
            X_train=X_train_t,
            X_train_val = X_train_val,
            y_train=y_train_t,
            y_train_val = y_train_val,
            X_test=X_test,
            y_test=y_test,
            model=model,
            pop_size=population_size,
            num_features=num_features,
            drop_rate=drop_rate,
            mutation_rate=mutation_rate,
            num_generation=num_generation,
            e_torlance=error_tolerance
    )

    # Pareto optimal solutions
    best_model_selected = nsga_tool.sort_selected_final_model(
                                        features=features,
                                        columns=columns,
                                        top_object=top_object,
                                        front_0_num=front_0_num,
                                        error_toler=error_tolerance
                                        )

    print("================================North Pareto optimal solutions=============================")
    final_model_features = []
    print('front_0_num: ', front_0_num)
    for index in range(front_0_num):
            if(top_object[index] == best_model_selected):
                print("==============⚠️ NOTICE! THIS IS THE BEST MODEL!==============")
                final_model_features = columns[features[index]]
                final_model_target_val = top_object[index]
                print('track ---best_model_selected: ', best_model_selected)
            print(f"track ========= {index} =========")
            print(f'track ---{index}-- columns[features[index]] len: ',len(columns[features[index]]))
            print(f'track ---{index}-- columns[features[index]] : ',columns[features[index]])
            print(f'track ---{index}-- top_object: ', top_object[index])

    print("==============================================END======================================")

    """
        generating gif for the iteration
    """
    # ==========================
    debug_2 = True
    # generating gif for iterations
    if debug_2:
        now = int(round(time.time() * 1000))
        now02 = time.strftime('%Y%m%d-%H%M%S', time.localtime(now / 1000))
        gif_images = []
        gif_n_generation = num_generation
        for i in range(1, gif_n_generation + 1):
            gif_images.append(imageio.imread(f'../gaImages/generation_{i}.jpg'))
        imageio.mimsave(f"../gifs/north_optimization{now02}.gif", gif_images, fps=3)
    # ==========================

    print("================================Model review=============================")
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

    final_model = RandomForestClassifier(n_estimators=25,criterion='gini',random_state=42)

    # 得到最终模型的Accuracy，precision，recall，f1 score，auc 以及 confuse matricx
    selected_columns = modelling_revised_calibration(
                                    X_train=X_train_t_1[final_model_features],
                                    X_train_val=X_train_val_1[final_model_features],
                                    y_train=y_train_t_1,
                                    y_train_val=y_train_val_1,
                                    X_test=X_test_1[final_model_features],
                                    y_test=y_test_1,
                                    model=final_model,
                                    name="north"
                                )
    # ==========================

