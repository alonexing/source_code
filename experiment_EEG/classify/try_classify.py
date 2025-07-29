import os
import pickle
import numpy as np
from sklearn.decomposition import PCA, FastICA, SparsePCA, KernelPCA, IncrementalPCA
from pyemd import emd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression,BayesianRidge
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
import time
# 文件夹路径
folder_path = 'data\\data_eeg_MWL2_MW2'  # Replace with your path

model_type = 'rf'
n_splits = 5
decompose = 'kernelpca'

n_components = 3

def reduce_sample_along_dim(sample, n_components=3, method='pca'):


    # sample：(1,30,800)
    data = sample.squeeze().T # (30, 800)

    if method == 'pca':
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(data)  # (30, n_components)

        reconstructed = pca.inverse_transform(reduced)

        reduced = reduced.T  # (n_components, 30)


        # from scipy.interpolate import interp1d
        # x_old = np.linspace(0, 29, 30)
        # x_new = np.linspace(0, 29, 800)
        # f = interp1d(x_old, reduced, axis=1)
        # reduced_upsampled = f(x_new)  # (n_components, 800)
        return reconstructed   # (n_components, 800)
    elif method == 'ica':
        ica = FastICA(n_components=n_components, random_state=44)
        reduced = ica.fit_transform(data)
        mixing_matrix = ica.mixing_  # shape: (features, n_components)
        sources = reduced  # shape: (samples, n_components)
        reconstructed = sources @ mixing_matrix.T
        reduced = reduced.T
        # from scipy.interpolate import interp1d
        # x_old = np.linspace(0, 29, 30)
        # x_new = np.linspace(0, 29, 800)
        # f = interp1d(x_old, reduced, axis=1)
        # reduced_upsampled = f(x_new)
        return reconstructed
    elif method == 'sparsepca':
        spca = SparsePCA(n_components=n_components, alpha=1)
        reduced = spca.fit_transform(data)
        reconstructed = reduced @ spca.components_
        reduced = reduced.T
        return reduced
    elif method == 'kernelpca':
        kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=0.1)
        reduced = kpca.fit_transform(data)

        mean_data = np.mean(data, axis=0)
        reconstructed = np.tile(mean_data, (data.shape[0], 1))
        reduced = reduced.T
        return reconstructed

    elif method == 'emd':

        imf_features = []
        for i in range(800):
            signal = data[i, :]

            IMFs = emd(signal)
            if IMFs.shape[0] > 0:
                imf = IMFs[0]
            else:
                imf = signal
            imf_features.append(np.mean(imf))

        feature_vector = np.array(imf_features)

        return feature_vector[np.newaxis, np.newaxis, :]  # (1, 1, 800)
    else:
        raise ValueError("Unsupported dimensionality reduction method：{}".format(method))
def build_deep_learning_model(model_type='cnn', num_classes=2):
    if model_type == 'cnn':
        model = models.Sequential()
        model.add(layers.Input(shape=(1, 30, 800)))
        model.add(layers.Reshape((30, 800, 1)))
        model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(num_classes, activation='softmax'))
        return model
    elif model_type == 'mlp':
        model = models.Sequential()
        model.add(layers.Flatten(input_shape=(1, 30, 800)))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(num_classes, activation='softmax'))
        return model
    elif model_type == 'lstm':
        model = models.Sequential()
        model.add(layers.Reshape((30, 800), input_shape=(1, 30, 800)))
        model.add(layers.LSTM(64))
        model.add(layers.Dense(num_classes, activation='softmax'))
        return model
    else:
        raise ValueError("Supported deep learning model types：'cnn', 'mlp', 'lstm'")

def train_and_evaluate(X_train, y_train, X_test, y_test, model_type='cnn', num_classes=2):
    if model_type in ['cnn', 'mlp', 'lstm']:
        model = build_deep_learning_model(model_type, num_classes)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.1, verbose=0)
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, digits=4)
        return acc, f1, precision, recall, report, y_pred
    elif model_type == 'svm':
        X_train_flat = X_train.reshape((X_train.shape[0], -1))
        X_test_flat = X_test.reshape((X_test.shape[0], -1))
        clf = SVC(kernel='rbf')
        clf.fit(X_train_flat, y_train)
        y_pred = clf.predict(X_test_flat)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, digits=4)
        return acc, f1, precision, recall, report, y_pred
    elif model_type == 'rf':
        X_train_flat = X_train.reshape((X_train.shape[0], -1))
        X_test_flat = X_test.reshape((X_test.shape[0], -1))
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train_flat, y_train)
        y_pred = clf.predict(X_test_flat)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, digits=4)
        return acc, f1, precision, recall, report, y_pred
    elif model_type == 'knn':
        X_train_flat = X_train.reshape((X_train.shape[0], -1))
        X_test_flat = X_test.reshape((X_test.shape[0], -1))
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train_flat, y_train)
        y_pred = clf.predict(X_test_flat)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, digits=4)
        return acc, f1, precision, recall, report, y_pred
    elif model_type == 'xgb':

        X_train_flat = X_train.reshape((X_train.shape[0], -1))
        X_test_flat = X_test.reshape((X_test.shape[0], -1))
        clf = XGBClassifier(
            objective='multi:softmax',
            num_class=num_classes,
            eval_metric='mlogloss',
            use_label_encoder=False,
            n_estimators=100,
            random_state=44
        )
        clf.fit(X_train_flat, y_train)
        y_pred = clf.predict(X_test_flat)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, digits=4)
        return acc, f1, precision, recall, report, y_pred
    elif model_type == 'gb':
        # GradientBoosting
        X_train_flat = X_train.reshape((X_train.shape[0], -1))
        X_test_flat = X_test.reshape((X_test.shape[0], -1))
        clf = GradientBoostingClassifier(n_estimators=100, random_state=44)
        clf.fit(X_train_flat, y_train)
        y_pred = clf.predict(X_test_flat)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, digits=4)
        return acc, f1, precision, recall, report, y_pred
        pass
    elif model_type == 'hgb':
        # HistGradientBoosting
        X_train_flat = X_train.reshape((X_train.shape[0], -1))
        X_test_flat = X_test.reshape((X_test.shape[0], -1))
        clf = HistGradientBoostingClassifier(max_iter=100, random_state=44)
        clf.fit(X_train_flat, y_train)
        y_pred = clf.predict(X_test_flat)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, digits=4)
        return acc, f1, precision, recall, report, y_pred
        pass
    elif model_type == 'lgb':
        # LightGBM
        X_train_flat = X_train.reshape((X_train.shape[0], -1))
        X_test_flat = X_test.reshape((X_test.shape[0], -1))
        clf = LGBMClassifier(n_estimators=100, random_state=44)
        clf.fit(X_train_flat, y_train)
        y_pred = clf.predict(X_test_flat)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, digits=4)
        return acc, f1, precision, recall, report, y_pred
        pass
    elif model_type == 'logreg':
        X_train_flat = X_train.reshape((X_train.shape[0], -1))
        X_test_flat = X_test.reshape((X_test.shape[0], -1))
        clf = LogisticRegression(max_iter=1000, random_state=44)
        clf.fit(X_train_flat, y_train)
        y_pred = clf.predict(X_test_flat)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, digits=4)
        return acc, f1, precision, recall, report, y_pred
    elif model_type == 'Bayeslr':
        X_train_flat = X_train.reshape((X_train.shape[0], -1))
        X_test_flat = X_test.reshape((X_test.shape[0], -1))
        clf = BayesianRidge()
        clf.fit(X_train_flat, y_train)
        y_pred = clf.predict(X_test_flat,return_std=True)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, digits=4)
        return acc, f1, precision, recall, report, y_pred
    elif model_type == 'extratree':  # 极端随机树
        X_train_flat = X_train.reshape((X_train.shape[0], -1))
        X_test_flat = X_test.reshape((X_test.shape[0], -1))
        clf = ExtraTreesClassifier(n_estimators=100, random_state=44)
        clf.fit(X_train_flat, y_train)
        y_pred = clf.predict(X_test_flat)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, digits=4)
        return acc, f1, precision, recall, report, y_pred
    elif model_type == 'adaboost':  # AdaBoost
        X_train_flat = X_train.reshape((X_train.shape[0], -1))
        X_test_flat = X_test.reshape((X_test.shape[0], -1))
        clf = AdaBoostClassifier(n_estimators=100, random_state=44)
        clf.fit(X_train_flat, y_train)
        y_pred = clf.predict(X_test_flat)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, digits=4)
        return acc, f1, precision, recall, report, y_pred
    elif model_type == 'catboost':
        X_train_flat = X_train.reshape((X_train.shape[0], -1))
        X_test_flat = X_test.reshape((X_test.shape[0], -1))
        clf = CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            verbose=False,
            random_seed=44
        )
        clf.fit(X_train_flat, y_train)
        y_pred = clf.predict(X_test_flat)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, digits=4)
        return acc, f1, precision, recall, report, y_pred
    elif model_type == 'naive_bayes':
        X_train_flat = X_train.reshape((X_train.shape[0], -1))
        X_test_flat = X_test.reshape((X_test.shape[0], -1))
        clf = GaussianNB()
        clf.fit(X_train_flat, y_train)
        y_pred = clf.predict(X_test_flat)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, digits=4)
        return acc, f1, precision, recall, report, y_pred
    else:
        raise ValueError("Unsupported model type：{}".format(model_type))
last_three_chars = folder_path[-3:]

per_subject_accuracies = []
per_subject_F1 = []
per_subject_Precision = []
per_subject_Recall = []
people_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
for person_file in people_files:
    person_path = os.path.join(folder_path, person_file)
    with open(person_path, 'rb') as f:
        data_dict = pickle.load(f)

    data_list = data_dict['data']
    # data_list = np.concatenate((data_list[0], data_list[1]), axis=0)
    label_list = data_dict['label']
    # label_list  = np.concatenate((label_list[0], label_list[1]), axis=0)
    X = np.array(data_list)
    y = np.array(label_list)
    start_time = time.time()
    if decompose == 'pca':
        X = np.array([reduce_sample_along_dim(sample, n_components=n_components, method='pca') for sample in X])
    elif decompose == 'ica':
        X = np.array([reduce_sample_along_dim(sample, n_components=n_components, method='ica') for sample in X])
    elif decompose == 'emd':
        X = np.array([reduce_sample_along_dim(sample, method='emd') for sample in X])
    elif decompose == 'sparsepca':
        X = np.array([reduce_sample_along_dim(sample, n_components=n_components, method='sparsepca') for sample in X])
    elif decompose == 'kernelpca':
        X = np.array([reduce_sample_along_dim(sample, n_components=n_components, method='kernelpca') for sample in X])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2023)
    end_time = time.time()
    train_time = end_time - start_time
    print(f"Dimensionality reduction time：{train_time:.4f} seconds")
    accuracies = []
    F1= []
    Precision = []
    Recall = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        start_time = time.time()
        # 训练和评估
        acc, f1, precision, recall, report, y_pred = train_and_evaluate(X_train, y_train, X_test, y_test, model_type)
        accuracies.append(acc)
        F1.append(f1)
        Precision.append(precision)
        Recall.append(recall)
        end_time = time.time()
        train_time = end_time - start_time
        print(f"Training time：{train_time:.4f} seconds")

    mean_acc = np.mean(accuracies)
    mean_f1 = np.mean(F1)
    mean_precision = np.mean(Precision)
    mean_recall = np.mean(Recall)
    per_subject_accuracies.append(mean_acc)
    per_subject_F1.append(mean_f1)
    per_subject_Precision.append(mean_precision)
    per_subject_Recall.append(mean_recall)
    print(f"Subject：{person_file}")
    print(f"5-fold average accuracy：{mean_acc:.4f}\n")
    print(f"5-fold average F1：{mean_f1:.4f}\n")
    print(f"5-fold average precision：{mean_precision:.4f}\n")
    print(f"5-fold average recall：{mean_recall:.4f}\n")
    print("-" * 50)


overall_avg_acc = np.mean(per_subject_accuracies)
overall_avg_f1 = np.mean(per_subject_F1)
overall_avg_pre = np.mean(per_subject_Precision)
overall_avg_recall = np.mean(per_subject_Recall)
print(f"Overall average accuracy: {overall_avg_acc:.4f}")
print(f"Overall average F1: {overall_avg_f1:.4f}")
print(f"Overall average precision: {overall_avg_pre:.4f}")
print(f"Overall average recall: {overall_avg_recall:.4f}")


# with open(f'{model_type}_{decompose}_per_subject_accuracies_{last_three_chars}.txt', 'w') as f:
#     for idx, acc in enumerate(per_subject_accuracies):
#         f.write(f"{people_files[idx]}：{acc:.4f}\n")
#     f.write(f"\naverage accuracy：{overall_avg_acc:.4f}\n")
# with open(f'{model_type}_{decompose}_per_subject_f1_{last_three_chars}.txt', 'w') as f:
#     for idx, f1 in enumerate(per_subject_F1):
#         f.write(f"{people_files[idx]}：{f1:.4f}\n")
#     f.write(f"\naverage F1：{overall_avg_f1:.4f}\n")
# with open(f'{model_type}_{decompose}_per_subject_precision_{last_three_chars}.txt', 'w') as f:
#     for idx, precision in enumerate(per_subject_Precision):
#         f.write(f"{people_files[idx]}：{precision:.4f}\n")
#     f.write(f"\naverage precision：{overall_avg_pre:.4f}\n")
# with open(f'{model_type}_{decompose}_per_subject_recall_{last_three_chars}.txt', 'w') as f:
#     for idx, recall in enumerate(per_subject_Recall):
#         f.write(f"{people_files[idx]}：{recall:.4f}\n")
#     f.write(f"\naverage recall：{overall_avg_recall:.4f}\n")