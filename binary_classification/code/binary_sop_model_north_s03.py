"""
使用随机森林进行多分类
"""
import pandas as pd
from sklearn.svm import LinearSVC, SVC

from binary_classification.tools.opUtils import modelling_revised_calibration
from binary_classification.tools.utils import *
from binary_classification.tools.sop_tool import *

np.random.seed(42)
def binary_sop_model_north(population_size,num_features,num_generation,drop_rate,mutation_rate,error_tolerance,model):

    # ========================== 将consloe打印输出保存到log文件 ==========================
    import sys
    import time
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
    log_name = "../logs/north_results_"+str(t)+".log"
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

    feature_data_path = "../train_data/north_feature_data.csv"
    feature_data = pd.read_csv(feature_data_path)
    label_data_path = "../train_data/north_label_data.csv"
    label_data = pd.read_csv(label_data_path)

    # print(y_array)
    X_train, X_test, y_train, y_test = train_test_split(feature_data,label_data,test_size=0.2,random_state=42)
    X_train_t, X_train_val, y_train_t, y_train_val = train_test_split(X_train, y_train,
                                                                      test_size=0.5, random_state=42)
    # 将数据进行保存（为了进行复现）
    X_train_t.to_csv("../north_review_data/X_train_t.csv", sep=',', index=False)
    X_train_val.to_csv("../north_review_data/X_train_val.csv", sep=',', index=False)
    y_train_t.to_csv("../north_review_data/y_train_t.csv", sep=',', index=False)
    y_train_val.to_csv("../north_review_data/y_train_val.csv", sep=',', index=False)

    X_test.to_csv("../north_review_data/X_test.csv", sep=',', index=False)
    y_test.to_csv("../north_review_data/y_test.csv", sep=',', index=False)


    # 将dataframe转换为narry
    y_train_t = np.array(y_train_t[label_name])
    y_train_val = np.array(y_train_val[label_name])
    y_test = np.array(y_test[label_name])

    # print("y_train",type(y_train),y_train.shape)
    # print("y_test",type(y_test),y_test.shape)
    # 选择基础分类器（此处选择的随机森林）+ 参数设置
    # ==========================

    # ==========================
    # 7.执行程序

    # NSGA-II-RF-Calibration
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
            name='north'
    )
    # ==========================
    """
    （2）自定义评分流程
    """
    # ==========================
    # 2.自定义评分（从帕累托解集中选出最佳模型）
    best_model_selected = nsga_tool.sort_selected_final_model(
                                        features=features,
                                        columns=columns,
                                        top_object=top_object,
                                        front_0_num=front_0_num,
                                        error_toler=error_tolerance
                                        )

    # ==========================
    # 3.得到最佳模型的目标值以及选择的特征变量
    final_model_features = []
    final_model_target_val = []
    print('front_0_num: ', front_0_num)
    # print(f'track ---features: ',features)
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

    print("================================END=============================")

    """
    （4）* 将每个generation得到的目标绘制为gif图 (optional)
    """
    # ==========================
    debug_2 = True
    # 将每个generation得到的目标绘制为gif图
    if debug_2:
        # 获取时间戳
        now = int(round(time.time() * 1000))
        now02 = time.strftime('%Y%m%d-%H%M%S', time.localtime(now / 1000))

        gif_images = []
        gif_n_generation = num_generation
        for i in range(1, gif_n_generation + 1):
            gif_images.append(imageio.imread(f'../north_gaImages/generation_{i}.png'))  # 读取图片
        imageio.mimsave(f"../gifs/{now02}-nsga-ii_north.gif", gif_images, fps=3)  # 转化为gif动画
    # ==========================

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
                                    model=final_model
                                )
    # ==========================

