from numpy.core.defchararray import title
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
import re
import numpy as np
import matplotlib.pyplot as plt

def clean_text(text):
    if text is np.nan:
        return ''
    text = text.lower().strip()
    text = ''.join([ch for ch in text if ord(ch)<128])
    return text

def clean(df):
    # ages = []
    subjects, titles, checkouts = [], [], []
    for i, (subj, title, checkout) in enumerate(zip(df['Subjects'].to_list(), df['Title'].to_list(), df['Checkouts'].to_list())):
        subjects.append(clean_text(subj))
        titles.append(clean_text(title))
        checkouts.append(int(checkout))   
        # age = 0
        # values = re.findall('\d{4}', year)
        # if len(values)>0:
        #     age = 2020 - int(values[-1])
        # ages.append(age)
    return {
        'subjects':subjects, 
        'titles':titles,
        'checkouts':checkouts
    }

df_train = pd.read_csv(r'train_file.csv')
materials = df_train['MaterialType'].to_list()

df_test = pd.read_csv(r'test_file.csv')
training_data = clean(df_train)
test_data = clean(df_test)
# clean data
classes = list(set(materials))
num_classes = len(classes)
y = [classes.index(mat) for mat in materials]
y = np.identity(num_classes)[y]

tokenizer = Tokenizer(num_words=8000)
tokenizer.fit_on_texts(training_data['subjects']+ training_data['titles'])

x_subj = tokenizer.texts_to_matrix(training_data['subjects'])
x_titles = tokenizer.texts_to_matrix(training_data['titles'])
x_checkouts = np.array(training_data['checkouts']).reshape((len(training_data['checkouts']),1))

x_checkout_max, x_checkout_min = x_checkouts.max(), x_checkouts.min()
x_checkouts = (x_checkouts - x_checkout_min)/ (x_checkout_max- x_checkout_min)

split = 25000
# x1_train, x1_val = x_subj[:split], x_subj[split:]
# x2_train, x2_val = x_titles[:split], x_titles[split:]
x = np.concatenate([x_checkouts, x_subj, x_titles], axis=-1)
x_train, x_val = x[:split], x[split:]
y_train, y_val = y[:split], y[split:]

model = models.Sequential([
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.fit(x_train, y_train, \
    batch_size = 128,\
    epochs = 4,\
    validation_data= (x_val, y_val))


# subjects, years, material = read_file(r'train_file.csv')
print('done')


