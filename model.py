# Loading the libraries
from glob import glob
import numpy as np
from scipy import stats
import os
import mne
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv1D, BatchNormalization, LeakyReLU, MaxPool1D, GlobalAveragePooling1D, Dense, Dropout, AveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

# Reading the data
exclude = ('EEG 23A-23R', 'EEG 24A-24R', 'Status', 'EEG A2-A1')

def read_eeg(path):
    raw = mne.io.read_raw_edf(path, preload=True, exclude=exclude)
    raw.set_eeg_reference()
    raw.filter(l_freq=0.5, h_freq=45)
    epoch = mne.make_fixed_length_epochs(raw, duration=5, overlap=1)
    array = epoch.get_data()
    return array

healthy_path = glob('Dataset\\test\\Healthy\\*.edf')
mdd_path = glob('Dataset\\test\\MDD\\*.edf')

healthy_eeg = [read_eeg(path) for path in healthy_path]
mdd_eeg = [read_eeg(path) for path in mdd_path]
healthy_labels = [len(arr)*[0] for arr in healthy_eeg]
mdd_labels = [len(arr)*[1] for arr in mdd_eeg]
eeg = healthy_eeg + mdd_eeg
labels = healthy_labels + mdd_labels

groups = [[i]*len(j) for i, j in enumerate(eeg)]

eeg_array = np.vstack(eeg)
label_array = np.hstack(labels)
group_array = np.hstack(groups)
eeg_array = np.moveaxis(eeg_array, 1, 2)

model = Sequential()
model.add(Conv1D(filters=5, kernel_size=3, strides=1, input_shape=(1280, 19)))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPool1D(pool_size=2, strides=2))
model.add(Conv1D(filters=5, kernel_size=3, strides=1))
model.add(LeakyReLU())
model.add(MaxPool1D(pool_size=2, strides=2))
model.add(Dropout(0.5))
model.add(Conv1D(filters=5, kernel_size=3, strides=1))
model.add(LeakyReLU())
model.add(AveragePooling1D(pool_size=2, strides=2))
model.add(Dropout(0.5))
model.add(Conv1D(filters=5, kernel_size=3, strides=1))
model.add(LeakyReLU())
model.add(AveragePooling1D(pool_size=2, strides=2))
model.add(Conv1D(filters=5, kernel_size=3, strides=1))
model.add(LeakyReLU())
model.add(GlobalAveragePooling1D())
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])

model_name = 'DeepWave'
plot_model(model, to_file=f"Images/{model_name}_architecture.png", show_shapes=True, show_layer_names=True)

checkpoint = ModelCheckpoint(
    filepath=f'{model_name}.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
earlystop = EarlyStopping(
    patience=10,
    verbose=1
)

gkf = GroupKFold()
accuracy = []
for train_index, val_index in gkf.split(eeg_array, label_array, groups=group_array):
    train_features, train_labels = eeg_array[train_index], label_array[train_index]
    val_features, val_labels = eeg_array[val_index], label_array[val_index]
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features.reshape(-1, train_features.shape[-1])).reshape(train_features.shape)
    val_features = scaler.transform(val_features.reshape(-1, val_features.shape[-1])).reshape(val_features.shape)
    history = model.fit(train_features, train_labels, epochs=50, batch_size=128, validation_data=(val_features, val_labels), callbacks=[checkpoint, earlystop])
    accuracy.append(model.evaluate(val_features, val_labels)[1])

metrics = ["accuracy", "loss"]
for metric in metrics:
    plt.clf()
    plt.plot(history.history[metric], label='train')
    plt.plot(history.history[f'val_{metric}'], label='val')
    plt.legend(loc="right")
    plt.xlabel('epochs')
    plt.ylabel(metric)
    plt.title(f"{model_name} {metric.capitalize()}")
    plt.savefig(f'Images/{model_name}_{metric}.png')