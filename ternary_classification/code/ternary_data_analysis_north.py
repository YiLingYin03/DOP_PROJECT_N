"""
 Step 1
 Description: preprocessing North data
 Classification: ternary classifications
 labels:
    Normal: 1
    Osteopenia : 2
    Osteoporosis: 3

 Date: 2020.10.07
"""

from ternary_classification.tools.utils import *
np.random.seed(0)

path = "../data/north_bone_cleanedData.csv"
isPlot = True

# feature variables and label
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

# data description
df = data_describe(path,label_name,is_plot=True)

# one-hot encoding and normalization
X,y = feature_uniform(df=df,
                      cat_columns=cat_columns,
                      num_columns=num_columns,
                      label_name=label_name)
# k-means oversampling
feature_data,label_data = imbalance_process(X=X,
                                            y=y,
                                            cat_columns=cat_columns,
                                            num_columns=num_columns)

# save training data
feature_data_path = "../north_train_data/feature_data.csv"
feature_data.to_csv(feature_data_path, sep=',', index=False)
label_data_path = "../north_train_data/label_data.csv"
label_data.to_csv(label_data_path, sep=',', index=False)


