import lightgbm
import matplotlib.pyplot as plt
import neptune
from neptunecontrib.monitoring.utils import pickle_and_send_artifact
from neptunecontrib.monitoring.metrics import log_binary_classification_metrics
from neptunecontrib.versioning.data import log_data_version
import pandas as pd

plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'figure.figsize': [16, 12]})
plt.style.use('seaborn-whitegrid')

# Define parameters
PROJECT_NAME = 'neptune-ai/binary-classification-metrics'

TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
NROWS = None

MODEL_PARAMS = {'random_state': 1234,
                'learning_rate': 0.1,
                'n_estimators': 1500}

# Load data
train = pd.read_csv(TRAIN_PATH, nrows=NROWS)
test = pd.read_csv(TEST_PATH, nrows=NROWS)

feature_names = [col for col in train.columns if col not in ['isFraud']]

X_train, y_train = train[feature_names], train['isFraud']
X_test, y_test = test[feature_names], test['isFraud']

# Start experiment
neptune.init(PROJECT_NAME)
neptune.create_experiment(name='lightGBM training',
                          params=MODEL_PARAMS,
                          upload_source_files=['train.py', 'environment.yaml'])
log_data_version(TRAIN_PATH, prefix='train_')
log_data_version(TEST_PATH, prefix='test_')

# Train model
model = lightgbm.LGBMClassifier(**MODEL_PARAMS)
model.fit(X_train, y_train)

# Evaluate model
y_test_pred = model.predict_proba(X_test)

log_binary_classification_metrics(y_test, y_test_pred)
pickle_and_send_artifact((y_test, y_test_pred), 'test_predictions.pkl')

neptune.stop()
