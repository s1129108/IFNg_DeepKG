import h5py
import os
import pickle
from tqdm import tqdm
from time import gmtime, strftime
from datetime import datetime
import numpy as np
import pandas as pd
import math
import time
from sklearn import metrics
from sklearn.metrics import roc_curve, precision_recall_curve
import tensorflow as tf
from tensorflow.keras import Model, layers
import gc
from sklearn.model_selection import KFold
import csv
import argparse
# --- In model training ---
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import import_tests as load_data

# Argument Parsing
parser = argparse.ArgumentParser(description='Program arguments')
parser.add_argument("-maxseq", "--MAXSEQ", type=int, default=500)
parser.add_argument("-f", "--FILTER", type=int, default=256)
parser.add_argument("-w", "--WINDOW", nargs='+', type=int, default=[2, 4, 6])
parser.add_argument("-nf", "--NUM_FEATURE", type=int, required=True, help="Number of features (e.g., 1024 for ProtTrans)")
parser.add_argument("-hi", "--HIDDEN", type=int, default=1000)
parser.add_argument("-drop", "--DROPOUT", type=float, default=0.7)
parser.add_argument("-ep", "--EPOCHS", type=int, default=20)
parser.add_argument("-df", "--DATA_FEATURE", type=str, default="pt")
parser.add_argument("-imb", "--imbalance_mod", type=str, default="None", help="Imbalance method: 'SMOTE', 'ADASYN', 'RANDOM'")
parser.add_argument("-csv", "--csv_path", type=str, default="MSCNN_log.csv")
parser.add_argument("-test", "--test_path", type=str, default="IndependentTest.csv")
parser.add_argument("-k", "--KFold", type=int, default=5)
parser.add_argument("-vm", "--validation_mode", type=str, default="cross")
# parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
args = parser.parse_args()


# Constants
MAXSEQ = args.MAXSEQ
NUM_FILTER = args.FILTER
WINDOW_SIZES = args.WINDOW
csv_file_path = args.csv_path
ind_file_path = args.test_path
DATA_FEATURE = args.DATA_FEATURE
DROPOUT = args.DROPOUT
NUM_HIDDEN = args.HIDDEN
IMBALANCE = args.imbalance_mod
MODE = args.validation_mode
# learning_rate = args.learning_rate
# print(f"Learning rate: {learning_rate}")
BATCH_SIZE = 256
NUM_CLASSES = 2
CLASS_NAMES = ['Negative', 'Positive']
NUM_FEATURE = args.NUM_FEATURE
EPOCHS = args.EPOCHS
K_Fold = args.KFold

print("FEATURE:", DATA_FEATURE)
print("NUM_FILTER:", NUM_FILTER)
print("WINDOW_SIZES:", WINDOW_SIZES)
print("IMBALANCE:", IMBALANCE)

def time_log(message):
    print(message, " : ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))

def SAVEROC(fpr, tpr, AUC):
    data_to_save = {"fpr": fpr, "tpr": tpr, "AUC": AUC}
    with open(f"./PKL_REVISION/MSCNN_Mouse2_{str(EPOCHS)}_{str(MAXSEQ)}_{DATA_FEATURE}_{str(DROPOUT)}_{str(NUM_FILTER)}_{str(NUM_HIDDEN)}_{WINDOW_SIZES}_AUCROC.pkl", "wb") as file:
        pickle.dump(data_to_save, file)

def SAVEPR(precision, recall, ROCPR):
    data_to_save = {"precision": precision, "recall": recall, "AUC": ROCPR}
    with open(f"./PKL_REVISION/MSCNN_Mouse2_{str(EPOCHS)}_{str(MAXSEQ)}_{DATA_FEATURE}_{str(DROPOUT)}_{str(NUM_FILTER)}_{str(NUM_HIDDEN)}_{WINDOW_SIZES}_PRROC.pkl", "wb") as file:
        pickle.dump(data_to_save, file)

def save_csv(write_data, filename):
    header = [
        "DATA_FEATURE", "WINDOW_SIZES", "IMBALANCE", "NUM_FILTER", "DROPOUT", 
        "NUM_HIDDEN", "EPOCH", "TP", "FP", "TN", "FN", 
        "Sensitivity", "Specificity", "Accuracy", "MCC", "AUC", "F1", 
        "Precision", "Recall"
    ]
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline='') as open_csv:
        write_csv = csv.writer(open_csv)
        if not file_exists:
            write_csv.writerow(header)
        write_csv.writerow(write_data)

# Model MSCNN
class DeepScan(Model):
    def __init__(self,
                 input_shape=(1, MAXSEQ, NUM_FEATURE),
                 window_sizes=WINDOW_SIZES,
                 num_filters=NUM_FILTER,
                 num_hidden=NUM_HIDDEN):
        super().__init__()
        self.convs = []
        self.pools = []
        for w in window_sizes:
            self.convs.append(layers.SeparableConv2D(
                filters=num_filters,
                kernel_size=(1, w),
                activation='relu',
                padding='valid'
            ))
            self.pools.append(layers.MaxPooling2D(
                pool_size=(1, MAXSEQ - w + 1),
                strides=(1, MAXSEQ - w + 1)
            ))
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(0.7)
        self.dense1 = layers.Dense(num_hidden, activation='relu')
        self.dense2 = layers.Dense(NUM_CLASSES, activation='softmax',
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-3))

    def call(self, x, training=False):
        features = []
        for conv, pool in zip(self.convs, self.pools):
            h = conv(x)
            h = pool(h)
            features.append(self.flatten(h))
        x = tf.concat(features, axis=1)
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        return self.dense2(x)

# Data Generator
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.data))

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = [self.data[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]
        return np.array(batch_data), np.array(batch_labels)

# Model Test
def model_test(model, x_test, y_test):
    print("Test shape:", x_test.shape)
    pred_test = model.predict(x_test)
    fpr, tpr, thresholds = roc_curve(y_test[:, 1], pred_test[:, 1])
    AUC = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=AUC, estimator_name='mCNN')
    display.plot()
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    threshold = thresholds[ix]
    #threshold = 0.95
    print(f'Best Threshold={threshold:.4f}, G-Mean={gmeans[ix]:.4f}')
    y_pred = (pred_test[:, 1] >= threshold).astype(int)
    TN, FP, FN, TP = metrics.confusion_matrix(y_test[:, 1], y_pred).ravel()
    Sens = TP / (TP + FN) if TP + FN > 0 else 0.0
    Spec = TN / (TN + FP) if TN + FP > 0 else 0.0
    Acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) > 0 else 0.0
    F1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
    Prec = TP / (TP + FP) if TP + FP > 0 else 0.0
    Recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    precision, recall, pr_thresholds = precision_recall_curve(y_test[:, 1], pred_test[:, 1])
    PR_AUC = metrics.auc(recall, precision)
    print(f'TP={TP}, FP={FP}, TN={TN}, FN={FN}, Sens={Sens:.4f}, Spec={Spec:.4f}, '
          f'Acc={Acc:.4f}, MCC={MCC:.4f}, AUC={AUC:.4f}, F1={F1:.4f}, Prec={Prec:.4f}, '
          f'Recall={Recall:.4f}, PR_AUC={PR_AUC:.4f}\n')
     # Save ROC and PR curves
    # SAVEROC(fpr, tpr, AUC)  # Save ROC
    # SAVEPR(precision, recall, PR_AUC)  # Save Precision-Recall Curve
    return TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, F1, Prec, Recall

# Imbalance Handling
def IMBALANCE_funct(IMBALANCE, x_train, y_train):
    if IMBALANCE == "None":
        return x_train, y_train
    else:
        from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
        x_train_2d = x_train.reshape(x_train.shape[0], -1)
        print("Reshaped x_train:", x_train_2d.shape)
        print("y_train shape:", y_train.shape)
        if IMBALANCE == "SMOTE":
            imbalance = SMOTE(random_state=42)
        elif IMBALANCE == "ADASYN":
            imbalance = ADASYN(random_state=42)
        elif IMBALANCE == "RANDOM":
            imbalance = RandomOverSampler(random_state=42)
        x_train_resampled, y_train_resampled = imbalance.fit_resample(x_train_2d, y_train)
        x_train_resampled = x_train_resampled.reshape(x_train_resampled.shape[0], 1, MAXSEQ, NUM_FEATURE)
        print("Resampled x_train:", x_train_resampled.shape)
        print("Resampled y_train:", y_train_resampled.shape)
        x_train = x_train_resampled
        y_train = y_train_resampled
        del x_train_resampled, y_train_resampled, x_train_2d
        gc.collect()
        y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
        return x_train, y_train

# Load Data
x_train, y_train, x_test, y_test = load_data.MCNN_data_load(feature_type="esm2")
print("x_train shape:", x_train.shape)
print("x_train dtype:", x_train.dtype)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("x_test dtype:", x_test.dtype)
print("y_test shape:", y_test.shape)

# Store results for n runs of 5-fold CV or independent test
all_run = {
    'TP': [], 'FP': [], 'TN': [], 'FN': [], 'Sens': [], 'Spec': [], 'Acc': [], 'MCC': [], 'AUC': [],
    'F1': [], 'Prec': [], 'Recall': []
}

if MODE == 'ind':
    NUM_RUNS = 1
    for i in range(NUM_RUNS):
        time_log(f"Run {i+1}/{NUM_RUNS} - Start Model Train")
        x_train, y_train = IMBALANCE_funct(IMBALANCE, x_train, y_train)
        generator = DataGenerator(x_train, y_train, batch_size=BATCH_SIZE)
        model = DeepScan(num_filters=NUM_FILTER, num_hidden=NUM_HIDDEN, window_sizes=WINDOW_SIZES)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.build(input_shape=x_train.shape)
        model.summary()
        model.fit(generator, epochs=EPOCHS, shuffle=True, verbose=1)
        time_log(f"Run {i+1}/{NUM_RUNS} - End Model Train")
        time_log(f"Run {i+1}/{NUM_RUNS} - Start Model Test")
        TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, F1, Prec, Recall = model_test(model, x_test, y_test)
        all_run['TP'].append(TP)
        all_run['FP'].append(FP)
        all_run['TN'].append(TN)
        all_run['FN'].append(FN)
        all_run['Sens'].append(Sens)
        all_run['Spec'].append(Spec)
        all_run['Acc'].append(Acc)
        all_run['MCC'].append(MCC)
        all_run['AUC'].append(AUC)
        all_run['F1'].append(F1)
        all_run['Prec'].append(Prec)
        all_run['Recall'].append(Recall)
        time_log(f"Run {i+1}/{NUM_RUNS} - End Model Test")

        #Save the model
        # model.save_weights(f"./saved_weights/Model_MSCNN_KG_Mouse2_{MAXSEQ}_{DATA_FEATURE}_{WINDOW_SIZES}.h5")
        # print(f"Save model Model_MSCNN_KG_Mouse2_{MAXSEQ}_{DATA_FEATURE}_{WINDOW_SIZES}.h5 successfully !")
        
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    # Calculate Mean ± Std for each metric
    mean_std_results = {metric: (np.mean(values), np.std(values)) for metric, values in all_run.items()}
    
    # Print results in "Mean ± Std" format
    print("\nFinal Results (Mean ± Std):")
    for metric, (mean, std) in mean_std_results.items():
        print(f"{metric}: {mean:.4f} ± {std:.4f}")

    # Save results to CSV
    csv_data = [DATA_FEATURE, WINDOW_SIZES, IMBALANCE, NUM_FILTER, DROPOUT, NUM_HIDDEN, EPOCHS] + \
               [f"{mean:.4f} ± {std:.4f}" for mean, std in mean_std_results.values()]
    df = pd.DataFrame([csv_data],
                      columns=["DATA_FEATURE", "WINDOW_SIZES", "IMBALANCE", "NUM_FILTER", "DROPOUT", 
                               "NUM_HIDDEN", "EPOCH", "TP", "FP", "TN", "FN", 
                               "Sensitivity", "Specificity", "Accuracy", "MCC", "AUC", "F1", 
                               "Precision", "Recall"])
    df.to_csv(ind_file_path, mode='a', index=False, header=not os.path.exists(ind_file_path))
    print(f"\nResults saved to {ind_file_path}")
    time_log(f"All {NUM_RUNS} independent test run(s) completed.")

if MODE == 'cross':
    time_log("Start cross")
    num_runs = 1
    for run in range(num_runs):
        kfold = KFold(n_splits=K_Fold, shuffle=True, random_state=run)
        results = []
        for train_index, test_index in kfold.split(x_train):
            X_train, X_test = x_train[train_index], x_train[test_index]
            Y_train, Y_test = y_train[train_index], y_train[test_index]
            X_train, Y_train = IMBALANCE_funct(IMBALANCE, X_train, Y_train)
            generator = DataGenerator(X_train, Y_train, batch_size=BATCH_SIZE)
            model = DeepScan(num_filters=NUM_FILTER, num_hidden=NUM_HIDDEN, window_sizes=WINDOW_SIZES)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.build(input_shape=X_train.shape)
            model.fit(generator, epochs=EPOCHS, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)], verbose=1, shuffle=True)
            
            TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, F1, Prec, Recall = model_test(model, X_test, Y_test)
            results.append([TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, F1, Prec, Recall])
            del X_train, X_test, Y_train, Y_test
            del model
            tf.keras.backend.clear_session()
            gc.collect()
        results = np.array(results)
        all_run['TP'].append(np.mean(results[:, 0]))
        all_run['FP'].append(np.mean(results[:, 1]))
        all_run['TN'].append(np.mean(results[:, 2]))
        all_run['FN'].append(np.mean(results[:, 3]))
        all_run['Sens'].append(np.mean(results[:, 4]))
        all_run['Spec'].append(np.mean(results[:, 5]))
        all_run['Acc'].append(np.mean(results[:, 6]))
        all_run['MCC'].append(np.mean(results[:, 7]))
        all_run['AUC'].append(np.mean(results[:, 8]))
        all_run['F1'].append(np.mean(results[:, 9]))
        all_run['Prec'].append(np.mean(results[:, 10]))
        all_run['Recall'].append(np.mean(results[:, 11]))
    
    mean_std_results = {metric: (np.mean(values), np.std(values)) for metric, values in all_run.items()}
    print("\nFinal Results (Mean ± Std):")
    for metric, (mean, std) in mean_std_results.items():
        print(f"{metric}: {mean:.4f} ± {std:.4f}")
    df = pd.DataFrame([[DATA_FEATURE, WINDOW_SIZES, IMBALANCE, NUM_FILTER, DROPOUT, NUM_HIDDEN] + 
                       [f"{mean:.4f} ± {std:.4f}" for mean, std in mean_std_results.values()]],
                      columns=["DATA_FEATURE", "WINDOW", "IMBALANCE", "NUM_FILTER", "DROPOUT", "NUM_HIDDEN", 
                               "TP", "FP", "TN", "FN", "Sens", "Spec", "Acc", "MCC", "AUC", "F1", "Prec", "Recall"])
    df.to_csv(csv_file_path, mode='a', index=False, header=not os.path.exists(csv_file_path))
    print(f"\nResults saved to {csv_file_path}")