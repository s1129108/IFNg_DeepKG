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
"----------------------------------------------------------------------------------------------------"
import csv
import argparse

import import_tests_multiple as load_data
# How to use
# python MCNN_PLM.py -maxseq 1000 -f 256 -w 4 8 16 -nf 1024 -dt "A" -df "pt" -imb "None" -k 0 -csv "pred.csv"

parser = argparse.ArgumentParser(description='Program arguments')
parser.add_argument("-maxseq", "--MAXSEQ", type=int, default=500)
parser.add_argument("-f", "--FILTER", type=int, default=256)
parser.add_argument("-w", "--WINDOW", nargs='+', type=int, default=[2, 4, 6])
parser.add_argument("-nf", "--NUM_FEATURE", type=int)

parser.add_argument("-hi", "--HIDDEN", type=int, default=1000)
parser.add_argument("-drop", "--DROPOUT", type=float, default=0.7)
parser.add_argument("-ep", "--EPOCHS", type=int, default=20)

# parser.add_argument("-dt", "--DATA_TYPE", type=str, default="A") # "A" or "B" (A:ionchannels, B:iontransporters)
parser.add_argument("-df", "--DATA_FEATURE", type=str, default="pt")
parser.add_argument("-imb", "--imbalance_mod", type=str, default="None", help="the mod for imbalance 'SMOTE','ADASYN','RANDOM'")
parser.add_argument("-csv", "--csv_path", type=str, default="MCNN_log.csv")

parser.add_argument("-k", "--KFold", type=int, default=5)
parser.add_argument("-vm","--validation_mode", type=str, default="cross")
parser.add_argument("-class","--class_dataset", type=str, default="class0")
args = parser.parse_args()

MAXSEQ = args.MAXSEQ
NUM_FILTER = args.FILTER
WINDOW_SIZES = args.WINDOW
csv_file_path = args.csv_path

# DATA_TYPE = args.DATA_TYPE
DATA_FEATURE = args.DATA_FEATURE

DROPOUT = args.DROPOUT
NUM_HIDDEN = args.HIDDEN

IMBALANCE = args.imbalance_mod
MODE = args.validation_mode
BATCH_SIZE  = 256
CLASS_DATASET = args.class_dataset


NUM_CLASSES = 2
CLASS_NAMES = ['Negative','Positive']


NUM_FEATURE = args.NUM_FEATURE
EPOCHS      = args.EPOCHS

K_Fold = args.KFold

print("FEATURE:", DATA_FEATURE)
print("NUM_FILTER:", NUM_FILTER)
print("WINDOW_SIZES:", WINDOW_SIZES)
print("imb:", IMBALANCE)

import datetime

write_data=[]

# write_data.append(DATA_LABEL)
# write_data.append(DATA_TYPE)
write_data.append(BATCH_SIZE)
write_data.append(NUM_HIDDEN)
write_data.append(WINDOW_SIZES)
write_data.append(NUM_FILTER)
# write_data.append(VALIDATION_MODE)
write_data.append(IMBALANCE)
"===================================================================================================="
def time_log(message):
    print(message," : ",strftime("%Y-%m-%d %H:%M:%S", gmtime()))

"----------------------------------------------------------------------------------------------------"

def SAVEROC(fpr,tpr,AUC):
    data_to_save = {
        "fpr": fpr,
        "tpr": tpr,
        "AUC": AUC
    }
    
    with open(f"./PKL/MCNN_{CLASS_DATASET}_{str(MAXSEQ)}_{DATA_FEATURE}_{str(WINDOW_SIZES)}_{IMBALANCE}_AUCROC.pkl", "wb") as file:
        pickle.dump(data_to_save, file)
        
        
def SAVEPR(precision,recall, ROCPR):
    data_to_save = {
        "precision": precision,
        "recall": recall,
        "AUC": ROCPR
    }
    
    with open(f"./PKL/MCNN_{CLASS_DATASET}_{str(MAXSEQ)}_{DATA_FEATURE}_{str(WINDOW_SIZES)}_{IMBALANCE}_PRROC.pkl", "wb") as file:
        pickle.dump(data_to_save, file)

def save_csv(write_data):
    import csv
    # b=datetime.datetime.now()
    # write_data.append(b-a)
    open_csv=open("./independent_test.csv","a")
    write_csv=csv.writer(open_csv)
    write_csv.writerow(write_data)
    
# model MCNN
class DeepScan(Model):
	def __init__(self,
	             input_shape=(1, MAXSEQ, NUM_FEATURE),
	             window_sizes=[1024],
	             num_filters=256,
	             num_hidden=1000):
		super(DeepScan, self).__init__()
		# Add input layer
		self.input_layer = tf.keras.Input(input_shape)
		self.window_sizes = window_sizes
		self.conv2d = []
		self.maxpool = []
		self.flatten = []
		for window_size in self.window_sizes:
			self.conv2d.append(
			 layers.Conv2D(filters=num_filters,
			               kernel_size=(1, window_size),
			               activation=tf.nn.relu,
			               padding='valid',
			               bias_initializer=tf.constant_initializer(0.1),
			               kernel_initializer=tf.keras.initializers.GlorotUniform()))
			self.maxpool.append(
			 layers.MaxPooling2D(pool_size=(1, MAXSEQ - window_size + 1),
			                     strides=(1, MAXSEQ),
			                     padding='valid'))
			self.flatten.append(layers.Flatten())
		self.dropout = layers.Dropout(rate=DROPOUT)
		self.fc1 = layers.Dense(
		 num_hidden,
		 activation=tf.nn.relu,
		 bias_initializer=tf.constant_initializer(0.1),
		 kernel_initializer=tf.keras.initializers.GlorotUniform())
		self.fc2 = layers.Dense(NUM_CLASSES,
		                        activation='softmax',
		                        kernel_regularizer=tf.keras.regularizers.l2(1e-3))

		# Get output layer with `call` method
		self.out = self.call(self.input_layer)

	def call(self, x, training=False):
		_x = []
		for i in range(len(self.window_sizes)):
			x_conv = self.conv2d[i](x)
			x_maxp = self.maxpool[i](x_conv)
			x_flat = self.flatten[i](x_maxp)
			_x.append(x_flat)

		x = tf.concat(_x, 1)
		x = self.dropout(x, training=training)
		x = self.fc1(x)
		x = self.fc2(x)  #Best Threshold
		return x

"----------------------------------------------------------------------------------------------------"
# design a custom focal loss
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.8, gamma=2.0, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.alpha = alpha  # Ưu tiên lớp thiểu số (cytokine receptor)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        ce = -y_true * tf.math.log(y_pred)
        pt = tf.where(y_true == 1.0, y_pred, 1.0 - y_pred)
        focal_weight = self.alpha * tf.pow(1.0 - pt, self.gamma)
        loss = focal_weight * ce
        return tf.reduce_mean(loss)

# model fit batch funtion
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
    
"----------------------------------------------------------------------------------------------------"
# predict
def model_test(model, x_test, y_test):
    print(x_test.shape)
    pred_test = model.predict(x_test)
    
    # Compute ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test[:,1], pred_test[:,1])
    AUC = metrics.auc(fpr, tpr)
    
    # Display ROC Curve
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=AUC, estimator_name='mCNN')
    display.plot()
    
    # Compute G-Mean and Best Threshold
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    threshold = thresholds[ix]
    print(f'Best Threshold={threshold}, G-Mean={gmeans[ix]}')
    
    # Apply threshold to get predictions
    y_pred = (pred_test[:, 1] >= threshold).astype(int)

    # Compute confusion matrix values
    TN, FP, FN, TP = metrics.confusion_matrix(y_test[:,1], y_pred).ravel()

    # Compute evaluation metrics
    Sens = TP / (TP + FN) if TP + FN > 0 else 0.0
    Spec = TN / (FP + TN) if FP + TN > 0 else 0.0
    Acc = (TP + TN) / (TP + FP + TN + FN)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) if TP + FP > 0 and FP + TN > 0 and TP + FN > 0 and TN + FN > 0 else 0.0
    F1 = 2 * TP / (2 * TP + FP + FN)
    Prec = TP / (TP + FP) if TP + FP > 0 else 0.0
    Recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    
    # Compute Precision-Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_test[:,1], pred_test[:,1])
    PR_AUC = metrics.auc(recall, precision)  # Compute PR AUC
    
    print(f'TP={TP}, FP={FP}, TN={TN}, FN={FN}, Sens={Sens:.4f}, Spec={Spec:.4f}, '
          f'Acc={Acc:.4f}, MCC={MCC:.4f}, AUC={AUC:.4f}, F1={F1:.4f}, Prec={Prec:.4f}, '
          f'Recall={Recall:.4f}, PR_AUC={PR_AUC:.4f}\n')

    # Save ROC and PR curves
    SAVEROC(fpr, tpr, AUC)  # Save ROC
    SAVEPR(precision, recall, PR_AUC)  # Save Precision-Recall Curve

    return TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, F1, Prec, Recall


# transform feature by SMOTE, ADASYN, RANDOM
def IMBALANCE_funct(IMBALANCE,x_train,y_train):
    if(IMBALANCE)=="None":
        return x_train,y_train
    else:
        from imblearn.over_sampling import SMOTE,ADASYN,RandomOverSampler
    
        # 將 x_train 的形狀重新整形為二維
        x_train_2d = x_train.reshape(x_train.shape[0], -1)
        print(x_train_2d.shape)
        print(y_train.shape)
        #print(y_train.shape)
        # 創建 SMOTE 物件
        if IMBALANCE=="SMOTE":
            imbalance = SMOTE(random_state=42)
        elif IMBALANCE=="ADASYN":
            imbalance = ADASYN(random_state=42)
        elif IMBALANCE=="RANDOM":
            imbalance = RandomOverSampler(random_state=42)
        
    
        # 使用 fit_resample 進行過採樣
        x_train_resampled, y_train_resampled = imbalance.fit_resample(x_train_2d, y_train)
    
        # 將 x_train_resampled 的形狀恢復為四維
        x_train_resampled = x_train_resampled.reshape(x_train_resampled.shape[0], 1, MAXSEQ, NUM_FEATURE)
    
        print(x_train_resampled.shape)
        print(y_train_resampled.shape)
    
        x_train=x_train_resampled
        y_train=y_train_resampled
        
        del x_train_resampled
        del y_train_resampled
        del x_train_2d
        gc.collect()
    
        import tensorflow as tf
        y_train = tf.keras.utils.to_categorical(y_train,NUM_CLASSES)
        return x_train,y_train

"====================================== train data lodding ======================================"
# Example usage:
# x_train,y_train,x_test,y_test= load_data.MCNN_data_load("prottrans")
x_train, y_train, x_test, y_test = load_data.MCNN_data_load(feature_type="prottrans", class_0_types=["neg"], class_1_types=["pos"])

print(x_train.shape)
print(x_train.dtype)
print(y_train.shape)
print(x_test.shape)
print(x_test.dtype)
print(y_test.shape)

# I do this as a requirement for revision. He wants me to run 5CV 100 times and get mean, std T.T

# Store the results of 100 run-time on 5CV
all_run = {
    'TP': [],
    'FP': [],
    'TN': [],
    'FN': [],
    'Sens': [],
    'Spec': [],
    'Acc': [],
    'MCC': [],
    'AUC': [],
    'F1':[],
    'Prec':[],
    'Recall':[]
}
if MODE == 'ind':
    
    NUM_RUNS = 1  # Run independent test 3 times
    all_results = []  # Store results from all runs

    for i in range(NUM_RUNS):
        time_log(f"Run {i+1}/{NUM_RUNS} - Start Model Train")

        x_train, y_train = IMBALANCE_funct(IMBALANCE, x_train, y_train)
        generator = DataGenerator(x_train, y_train, batch_size=BATCH_SIZE)

        model = DeepScan(
            num_filters=NUM_FILTER,
            num_hidden=NUM_HIDDEN,
            window_sizes=WINDOW_SIZES
        )
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # model.compile(optimizer='adam', 
        #               loss= FocalLoss(alpha=0.85, gamma=2.0)
        #               , metrics=['accuracy'])
        model.build(input_shape=x_train.shape)
        model.summary()

        # class_weight = {0:1.14,1:8.33}
        model.fit(
            generator,
            epochs=EPOCHS,
            shuffle=True,
        )

        time_log(f"Run {i+1}/{NUM_RUNS} - End Model Train")

        # Start Model Test
        time_log(f"Run {i+1}/{NUM_RUNS} - Start Model Test")
        TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, F1, Prec, Recall = model_test(model, x_test, y_test)

        # Store results
        all_results.append([TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, F1, Prec, Recall])

        time_log(f"Run {i+1}/{NUM_RUNS} - End Model Test")

        del model  # Clear model from memory

    # Convert to numpy array for calculations
    all_results = np.array(all_results)  # Shape: (10, 9)

    # Compute mean
    mean_results = np.mean(all_results, axis=0)

    # Display results
    metrics = ["TP", "FP", "TN", "FN", "Sensitivity", "Specificity", "Accuracy", "MCC", "AUC", "F1", "Prec","Recall"]
    for i, metric in enumerate(metrics):
        print(f"{metric}: {mean_results[i]:.4f}")

    # Save results to CSV
    save_csv(mean_results)

    time_log(f"All {NUM_RUNS} independent test run(s) completed.")


if MODE == 'cross':
    time_log("Start cross")

    num_runs = 1  # Number of times to repeat 5-fold CV

    for run in range(num_runs):
        kfold = KFold(n_splits=K_Fold, shuffle=True, random_state=run)  # Changing seed per run
        results = []

        for train_index, test_index in kfold.split(x_train):
            # Get train-test data
            X_train, X_test = x_train[train_index], x_train[test_index]
            Y_train, Y_test = y_train[train_index], y_train[test_index]

            # Handle class imbalance
            X_train, Y_train = IMBALANCE_funct(IMBALANCE, X_train, Y_train)
            generator = DataGenerator(X_train, Y_train, batch_size=BATCH_SIZE)

            # Build & compile model
            model = DeepScan(num_filters=NUM_FILTER, num_hidden=NUM_HIDDEN, window_sizes=WINDOW_SIZES)
            # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            # Compile với Focal Loss
            model.compile(optimizer='adam',
                  loss=FocalLoss(alpha=0.85, gamma=2.0),
                  metrics=['accuracy'])
            model.build(input_shape=X_train.shape)

            # Train model
            model.fit(generator, epochs=EPOCHS, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)], verbose=1, shuffle=True)

            # Evaluate model
            TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, F1, Prec, Recall = model_test(model, X_test, Y_test)
            results.append([TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, F1, Prec, Recall])

            # Free memory
            del X_train, X_test, Y_train, Y_test
            del model
            tf.keras.backend.clear_session()
            gc.collect()  # Ensure memory is cleared

        # Store results for this run
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
    
    # Calculate Mean ± Std for each metric
    mean_std_results = {metric: (np.mean(values), np.std(values)) for metric, values in all_run.items()}

    # Print results in "Mean ± Std" format
    print("\nFinal Results (Mean ± Std):")
    for metric, (mean, std) in mean_std_results.items():
        print(f"{metric}: {mean:.4f} ± {std:.4f}")

    # Save results to CSV
    df = pd.DataFrame([[WINDOW_SIZES] + [IMBALANCE] + [f"{mean:.4f} ± {std:.4f}" for mean, std in mean_std_results.values()]],
                      columns=["WINDOW", "IMBALANCE", "TP", "FP", "TN", "FN", "Sens", "Spec", "Acc", "MCC", "AUC","F1","Prec","Recall"])
    df.to_csv(csv_file_path, mode='a', index=False, header=not os.path.exists(csv_file_path))

    print(f"\nResults saved to {csv_file_path}")


        
