#%%
#paths
DATA_DIR = 'data/'
RAW_DATA_DIR = 'data/raw_data/'
PROCESSED_DATA_DIR = 'data/preprocessed_data/'
MODELS_DIR = 'models/'
MODEL_HISTORIES_DIR = 'models/histories/'
MODEL_WEIGHTS_DIR = 'models/weights/'
MODEL_PARAMS_DIR = 'models/parameters/'
EVALUATION_DIR = 'evaluation/'
EVALUATION_FIGURES_DIR = 'evaluation/figures/'


#data preprocessing
RATING_THRESHOLD = 3
ADD_DISSIMILARITY = True
SPLIT_METHOD = 'user'
TEST_DAYS = 180 #only used for temporal split
TEST_USER_RATIO = 0.2 #only used for user based split
META_COLUMNS = ['no_items_rated']


#hyperparameter Tuning
SPACE_VALS = {'embedding_size': [16, 128],
              'batch_size': [8, 32],
              'epochs': [2, 10],
              'model_type': ['dot', 'concatenated', 'deep', 'hybrid'],
              'test_size': [0.1, 0.35], #fraction of the full data used as validation set
              'validation_split': [0.1, 0.35]} #fraction of the training data to be used as a validation set during the training process
MAX_EVALS  = 2 #number of hyperopt trials


#evluation
k = 10 #for map@k
MAX_k = 25 #for map@k plot
COMMON_TITLE_SUBSTRING = 'Star Wars'


#prediction
TOP_RECOMMENDATIONS = 10 #the number of top recommendations to generate for each user
#%%
    