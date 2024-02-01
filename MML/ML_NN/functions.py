import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from keras.utils import to_categorical
from keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import warnings
import itertools
import copy


##################### DATA PROCESSING ##################
def splitting_into_X_Y(dataset):
    copy_of_dataset = dataset
    Y = copy_of_dataset.iloc[:, -1].to_frame()
    X = copy_of_dataset.iloc[:, :-1]
    return X, Y

def plot_histograms(dataset):
    dataset.hist()
    plt.show()

def plot_correlation(dataset):
    corr = dataset.corr()
    plt.subplots(figsize=(10,7))
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(20, 10, as_cmap=True))
    plt.show()

def VS(parameter, dataset):
    data = dataset[[parameter,'quality']]
    fig, axs = plt.subplots(ncols=1,figsize=(6,3))
    sns.barplot(x='quality', y=parameter, data=data, ax=axs)
    plt.title(f'{parameter} VS quality')

    plt.tight_layout()
    plt.show()
    plt.gcf().clear()

def vs_plots(data):
    for column in data.columns:
        if column != 'quality':
            VS(column, data)

def boxplots(wine_df):
    for label in wine_df.columns[:-1]:
        plt.boxplot([wine_df[wine_df['quality']==i][label]for i in range(1,11)])
        plt.title(label)
        plt.xlabel('quality')
        plt.ylabel(label)
        plt.show()

def mod_outlier(df):
        df1 = df.copy()
        df = df._get_numeric_data()
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 -(1.5 * iqr)
        upper_bound = q3 +(1.5 * iqr)


        for col in df.columns:
            for i in range(0,len(df[col])):
                if df[col][i] < lower_bound[col]:
                    df[col][i] = lower_bound[col]

                if df[col][i] > upper_bound[col]:
                    df[col][i] = upper_bound[col]


        for col in df.columns:
            df1[col] = df[col]

        return(df1)

def standardize(data):
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data))
    scaled_data.columns = data.columns
    mean = scaler.mean_
    std = scaler.scale_
    return scaled_data, mean, std

def normalize(data):
    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(data))
    normalized_data.columns = data.columns
    min_val = scaler.data_min_
    max_val = scaler.data_max_
    return normalized_data, min_val, max_val

def inverse_standardize(scaled_data, mean, std):
    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = std
    inverted = pd.DataFrame(scaler.inverse_transform(scaled_data))
    return inverted

def inverse_normalize(normalized_data, min_val, max_val):
    scaler = MinMaxScaler()
    scaler.data_min_ = min_val
    scaler.data_max_ = max_val
    inverted = pd.DataFrame(scaler.inverse_transform(normalized_data))
    return inverted

def split_and_normalize_scale(X,Y, apply_normalization):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    scaled_X_train, mean_X_train, std_X_train = standardize(x_train)
    scaled_X_test, mean_X_test, std_X_test = standardize(x_test)

    if apply_normalization:
        normalized_scaled_X_train, min_val_scaled_X_train, max_val_scaled_X_train = normalize(scaled_X_train)
        normalized_scaled_X_test, min_val_scaled_X_test, max_val_scaled_X_test = normalize(scaled_X_test)
        return normalized_scaled_X_train, normalized_scaled_X_test, y_train, y_test
    else:
        return scaled_X_train, scaled_X_test, y_train, y_test

def mine_train_test_split(Dataset, apply_normalization, apply_over_sampling):  
    X_, Y_ = splitting_into_X_Y(Dataset)
    Y_['quality'] = Y_['quality'].astype('category')

    if apply_over_sampling:
        unique_values = Dataset['quality'].unique()
        strategy = {value: 2300 for value in unique_values}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            oversample = SMOTE(sampling_strategy=strategy, k_neighbors=6)
            X_, Y_ = oversample.fit_resample(X_, Y_)

    x_tr, x_te, y_tr, y_te = split_and_normalize_scale(X_, Y_, apply_normalization)
    return x_tr, x_te, y_tr, y_te

def remove_least_important_features(df, plot, num_features_to_remove=3):
    X, Y = splitting_into_X_Y(df)
    model = RandomForestRegressor(random_state=1, max_depth=12)
    model.fit(X.values, Y.values.ravel())
    importances = model.feature_importances_
    feature_importances_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)
    features_to_remove = feature_importances_df.tail(num_features_to_remove)['Feature'].values
    df_reduced = df.drop(columns=features_to_remove)

    if plot:
        features = X.columns
        importances = model.feature_importances_
        indices = np.argsort(importances)[:]
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()

    return df_reduced


def adjusted_train_test_split(data, to_reduce, apply_normalization, apply_over_sampling):
    if to_reduce:
        data = remove_least_important_features(data, plot=False, num_features_to_remove=3)
    x_train, x_test, y_train, y_test = mine_train_test_split(data, apply_normalization=apply_normalization, 
                                                             apply_over_sampling=apply_over_sampling)

    unique_classes = sorted(data['quality'].unique())
    num_classes = len(unique_classes)

    replacement_dict = {cls: idx for idx, cls in enumerate(unique_classes)}

    y_train['quality'] = y_train['quality'].replace(replacement_dict)
    y_test['quality'] = y_test['quality'].replace(replacement_dict)

    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    return x_train, x_test, y_train, y_test

def map_quality(quality):
    if quality < 4:
        return 0
    elif quality >= 5 and quality <= 6:
        return 1
    else:
        return 2
    

############# Models training and evaluation #########

def create_and_compile_model(num_inputs, num_outputs):
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Input(shape=num_inputs))
    ann.add(tf.keras.layers.Dense(16, activation='relu', 
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    ann.add(tf.keras.layers.Dense(units=16, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=8, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=8, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=num_outputs, activation='softmax'))
    ann.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return ann

def calculate_sensitivity_specificity(conf_matrix, wine_dataframe):
    num_classes = len(wine_dataframe['quality'].unique())
    specificities = []
    sensitivities = []
    for i in range(num_classes):
        tp = conf_matrix[i, i]
        fn = np.sum(conf_matrix[i, :]) - tp
        fp = np.sum(conf_matrix[:, i]) - tp
        tn = np.sum(conf_matrix) - (tp + fn + fp)
        
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        specificities.append(specificity)
        sensitivities.append(sensitivity)
        
        print(f"Class {i} Specificity: {specificity:.4f}")
        print(f"Class {i} Sensitivity: {sensitivity:.4f}")

def plot_loss_and_accuracy(history):
    plt.plot(history.history['loss'], label='MAE training data')
    plt.plot(history.history['val_loss'], label='MAE validation data')
    plt.legend()
    plt.title('MAE for model')
    plt.ylabel('MAE')
    plt.xlabel('epoch')
    plt.show()
    plt.close()

    plt.plot(history.history['accuracy'], label='Accuracy training data')
    plt.plot(history.history['val_accuracy'], label='Accuracy validation data')
    plt.legend()
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.show()
    plt.close()

def evaluate_model(ann, x_test, y_test):
    y_pred = ann.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
    return conf_matrix, y_pred_classes, y_test_classes

def plot_confusion_matrix(conf_matrix, wine_dataframe):
    df_cm = pd.DataFrame(conf_matrix, index=[i for i in np.sort(wine_dataframe['quality'].unique())],
                         columns=[i for i in np.sort(wine_dataframe['quality'].unique())])
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt="d")

def generate_classification_report(y_test, y_pred):
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

def perform_cross_validation(wine_dataframe, to_reduce, apply_normalization, apply_over_sampling, epochs, batch_size):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=32)
    scores = []
    X, Y = splitting_into_X_Y(wine_dataframe)
    for train, test in kfold.split(X, Y):
        x_train, x_test, y_train, y_test = adjusted_train_test_split(wine_dataframe, to_reduce, apply_normalization, apply_over_sampling)

        ann = create_and_compile_model(x_train.shape[1], len(wine_dataframe['quality'].unique()))
        ann.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        loss, accuracy = ann.evaluate(x_test, y_test, verbose=0)
        scores.append(accuracy)

    print("Average accuracy across cross-validation:", np.round(np.mean(scores), 3))
    return np.round(np.mean(scores), 3)

def train_neural_network(wine_dataframe, to_reduce, apply_normalization, 
                         apply_over_sampling, to_decrease_num_classes, epochs, batch_size):
    if to_decrease_num_classes:
        wine_dataframe['quality'] = wine_dataframe['quality'].apply(map_quality)

    class_counts = wine_dataframe['quality'].value_counts()
    classes_to_remove = class_counts[class_counts < 10].index
    wine_dataframe = wine_dataframe[~wine_dataframe['quality'].isin(classes_to_remove)]
    x_train, x_test, y_train, y_test = adjusted_train_test_split(wine_dataframe, to_reduce, 
                                                                 apply_normalization, apply_over_sampling)
    num_inputs = x_train.shape[1]
    num_outputs = len(wine_dataframe['quality'].unique())
    ann = create_and_compile_model(num_inputs, num_outputs)
    history = ann.fit(x_train, y_train, batch_size=batch_size, 
                      epochs=epochs, validation_data=(x_test, y_test), verbose=0)
    plot_loss_and_accuracy(history)
    conf_matrix, y_pred, y_test = evaluate_model(ann, x_test, y_test)
    plot_confusion_matrix(conf_matrix, wine_dataframe)
    calculate_sensitivity_specificity(conf_matrix, wine_dataframe)
    generate_classification_report(y_test, y_pred)
    ann.summary()
    acc = perform_cross_validation(wine_dataframe, to_reduce, apply_normalization, 
                             apply_over_sampling, epochs, batch_size)
    return acc

########## Benchmarking ##########

def run_parameter_combinations(dataset, results_df):
    original_dataset = copy.deepcopy(dataset)  

    to_reduce_values = [True, False]
    apply_normalization_values = [True, False]
    apply_over_sampling_values = [True, False]
    to_decrease_num_classes_values = [True, False]
    epochs_values = [50]  
    batch_size_values = [32]  


    parameter_combinations = itertools.product(
        to_reduce_values, apply_normalization_values, apply_over_sampling_values,
        to_decrease_num_classes_values, epochs_values, batch_size_values)

    for params in parameter_combinations:
        to_reduce, apply_normalization, apply_over_sampling, to_decrease_num_classes, epochs, batch_size = params
        print("Testing with parameters:", params)
        dataset = copy.deepcopy(original_dataset)
        # For the case of white wine dataset with to_decrease_num_classes=True there is no need for oversampling since there are enough
        try:
            accuracy = train_neural_network(dataset, to_reduce, apply_normalization, apply_over_sampling, 
                                            to_decrease_num_classes, epochs, batch_size)

        except ValueError as e:
            print(f"Error in the combination {params}: {e}")
            continue  
        
        results_df = results_df.append({
            'to_reduce': to_reduce,
            'apply_normalization': apply_normalization,
            'apply_over_sampling': apply_over_sampling,
            'to_decrease_num_classes': to_decrease_num_classes,
            'epochs': epochs,
            'batch_size': batch_size,
            'cross_val_accuracy': accuracy}, ignore_index=True)

    return results_df




