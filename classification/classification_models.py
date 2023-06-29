import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#using method read_csv to read data from file has csv extension
datas = pd.read_csv("car_seats.csv")

#show data's informations(column name, non-null count, data type)
print(datas.info())

#choose result which model must predict
label = "US"

#divide datas into 2 groups(features and target)
data_features = datas.drop(labels=label, axis=1)
data_target = datas[label]

#divide each group into 2 groups(train and validation)
data_features_train, data_features_valid, data_target_train, data_target_valid = train_test_split(data_features, data_target, train_size=0.8, random_state=7)
data_target_train = pd.DataFrame(data_target_train)
data_target_valid = pd.DataFrame(data_target_valid)

#transform data
##numerical data (column Sales, CompPrice, Income, Advertising, Population, Price, Age, Education)
standard_scaler = StandardScaler()
data_features_train_copy = data_features_train.copy()
numerical_column_name = ["Sales", "CompPrice", "Income", "Advertising", "Population", "Price", "Age", "Education"]
data_features_train_numerical = data_features_train_copy[numerical_column_name]
data_features_train_numerical = standard_scaler.fit_transform(data_features_train_numerical.values)
data_features_train[numerical_column_name] = data_features_train_numerical

##ordinal data (column Shelveloc)
ordinal_encoder = OrdinalEncoder()
ordinal_column_name = ["ShelveLoc"]
data_features_train_ordinal = data_features_train_copy[ordinal_column_name]
data_features_train_ordinal = ordinal_encoder.fit_transform(data_features_train_ordinal.values)
data_features_train[ordinal_column_name] = data_features_train_ordinal

##nominal data (column Urban)
onehot_encoder = OneHotEncoder()
nominal_column_name = ["Urban"]
data_features_train_nominal = data_features_train_copy[nominal_column_name]
data_features_train_nominal = onehot_encoder.fit_transform(data_features_train_nominal.values)
data_features_train_nominal = data_features_train_nominal.toarray()
data_features_train_nominal_one_column = data_features_train_nominal[:,0]
data_features_train = data_features_train.assign(Urban=data_features_train_nominal_one_column)

print(data_features_train)

###nominal data (target US)
data_target_train_copy = data_target_train.copy()
nominal_column_name = ["US"]
data_target_train_nominal = data_target_train_copy[nominal_column_name]
data_target_train_nominal = onehot_encoder.fit_transform(data_target_train_nominal.values)
data_target_train_nominal = data_target_train_nominal.toarray()
data_target_train_nominal_one_column = data_target_train_nominal[:,0]
data_target_train = data_target_train.assign(US=data_target_train_nominal_one_column)

print(data_target_train)

##variable's relationship
sns.heatmap(data_features_train.corr(),
    cmap=sns.diverging_palette(220, 10, as_cmap=True),
    vmin=-1.0, vmax=1.0,
    square=True)

#visualize dataset
##histogram chart
data_features_train.hist()
plt.show()

#build model
svm_model = svm.SVC()
svm_model.fit(data_features_train, data_target_train)

##numerical data (column Sales, CompPrice, Income, Advertising, Population, Price, Age, Education)
data_features_valid_numerical = data_features_valid[numerical_column_name]
data_features_valid_numerical = standard_scaler.transform(data_features_valid_numerical.values)
data_features_valid[numerical_column_name] = data_features_valid_numerical

##ordinal data (column Shelveloc)
data_features_valid_ordinal = data_features_valid[ordinal_column_name]
data_features_valid_ordinal = ordinal_encoder.transform(data_features_valid_ordinal.values)
data_features_valid[ordinal_column_name] = data_features_valid_ordinal

##nominal data (column Urban)
nominal_column_name = ["Urban"]
data_features_valid_nominal = data_features_valid[nominal_column_name]
data_features_valid_nominal = onehot_encoder.transform(data_features_valid_nominal.values)
data_features_valid_nominal = data_features_valid_nominal.toarray()
data_features_valid_nominal_one_column = data_features_valid_nominal[:,0]
data_features_valid = data_features_valid.assign(Urban=data_features_valid_nominal_one_column)

print(data_features_valid)

###nominal data (target US)
nominal_column_name = ["US"]
data_target_valid_nominal = data_target_valid[nominal_column_name]
data_target_valid_nominal = onehot_encoder.transform(data_target_valid_nominal.values)
data_target_valid_nominal = data_target_valid_nominal.toarray()
data_target_valid_nominal_one_column = data_target_valid_nominal[:,0]
data_target_valid = data_target_valid.assign(US=data_target_valid_nominal_one_column)

print(data_target_valid)

predict = svm_model.predict(data_features_valid)
print("Confusion matrix:\n{}".format(confusion_matrix(data_target_valid, predict)))
print("Accuracy: {:.2%}".format(accuracy_score(data_target_valid, predict)))
print("Precision: {:.2%}".format(precision_score(data_target_valid, predict)))
print("F1 score: {:.2%}".format(f1_score(data_target_valid, predict)))
print("Recall: {:.2%}".format(recall_score(data_target_valid, predict)))
