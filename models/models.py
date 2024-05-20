#%%
import numpy as np
import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import Concatenate, Dense, Dot, Dropout, Embedding, Flatten
from tensorflow.keras.models import Model
from hyperopt import hp, tpe, Trials, fmin
from functools import partial
from utils.helpers import setup_logging
from utils.parameters import (MODEL_HISTORIES_DIR, MODEL_WEIGHTS_DIR, PROCESSED_DATA_DIR, MODEL_PARAMS_DIR, 
                              META_COLUMNS, SPACE_VALS, META_COLUMNS, MAX_EVALS)

logger = setup_logging()


class ClassificationModel(Model):
      def __init__(self, embedding_size, max_user_id, max_item_id, model_type='dot'):
         super().__init__()

         self.user_embedding = Embedding(output_dim=embedding_size,
                                          input_dim=max_user_id + 1, #common practice that adds robustsness to model: reserve one additional index for entities that weren't seen during training. If there are no new entities the last weights/index of embedding will stay constant throughout the training.
                                          input_length=1,
                                          name='user_embedding')
         self.item_embedding = Embedding(output_dim=embedding_size,
                                          input_dim=max_item_id + 1, #common practice that adds robustsness to model: reserve one additional index for entities that weren't seen during training.
                                          input_length=1,
                                          name='item_embedding')

         self.flatten = Flatten()
         self.dot = Dot(axes=1)
         self.concat = Concatenate()
         self.dropout = Dropout(0.3)
         self.dense1 = Dense(units=64, activation='relu')
         self.dense2 = Dense(units=32, activation='relu')
         self.dense3 = Dense(units=1)

         self.model_type = model_type

      def call(self, inputs, training=False):
         user_inputs = inputs[0]
         item_inputs = inputs[1]

         user_vecs = self.flatten(self.user_embedding(user_inputs))
         item_vecs = self.flatten(self.item_embedding(item_inputs))

         if self.model_type == 'dot':
               y = self.dot([user_vecs, item_vecs])

         elif self.model_type == 'concatenated':
               input_vecs = self.concat([user_vecs, item_vecs])

               y = self.dense1(input_vecs)
               y = self.dense3(y)

         elif self.model_type == 'deep':
               input_vecs = self.concat([user_vecs, item_vecs])
            
               y = self.dropout(input_vecs, training=training)
               y = self.dense1(y)
               y = self.dropout(y, training=training)
               y = self.dense3(y)

         elif self.model_type == 'hybrid':
               meta_inputs = inputs[2]
               meta_inputs = tf.expand_dims(meta_inputs, axis=-1)#so the shape is (None, 1)
               user_vecs = self.dropout(user_vecs, training=training)
               item_vecs = self.dropout(item_vecs, training=training)
               input_vecs = self.concat([user_vecs, item_vecs, meta_inputs])

               y = self.dense1(input_vecs)
               y = self.dropout(y, training=training)
               y = self.dense2(y)
               y = self.dropout(y, training=training)
               y = self.dense3(y)

         return y


def initialize_and_compile_model(model_type: str, embedding_size: int, 
                                 even_combinations: pd.DataFrame) -> ClassificationModel:
    """
    Initialize and compile the classification model.

    :param model_type: type of collaborative filtering model ('dot', 'concatenated', 'deep' or 'hybrid').
    :param embedding_size: dimensionality of the embedding vectors.
    :param even_combinations: dataframe with balanced number of positive (relevant) and negative ratings.

    :return: compiled classification model.
    """
    max_user_id = even_combinations['user_id'].max()
    max_item_id = even_combinations['item_id'].max()

    model = ClassificationModel(embedding_size=embedding_size, 
                                max_user_id=max_user_id, 
                                max_item_id=max_item_id,
                                model_type=model_type)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def prepare_inputs(data: pd.DataFrame, model_type: str, meta_columns: list = META_COLUMNS) -> list:
  """
  Prepare input data for the model based on the model type.

  :param data: dataframe containing the input data.
  :param meta_columns: list of columns containing additional metadata for hybrid model.
  :param model_type: type of collaborative filtering model ('dot', 'concatenated', 'deep', or 'hybrid').

  :return: list of prepared input arrays for the model.
  """
  if model_type == 'hybrid':
      inputs = [np.array(data["user_id"]), np.array(data["item_id"])] + [np.array(data[col]) for col in meta_columns]
  else:
      inputs = [np.array(data["user_id"]), np.array(data["item_id"])]
  return inputs


def build_and_train_model(model_type: str,
                          embedding_size: int,
                          batch_size: int, 
                          epochs: int,
                          validation_split: float,
                          test_size: float,
                          even_combinations: pd.DataFrame = None,
                          train: pd.DataFrame = None, 
                          test: pd.DataFrame = None,
                          meta_columns: list = META_COLUMNS) -> tuple[pd.DataFrame, History]:
    """
    Build and train a collaborative filtering model. The function can be used with hyperparameter tuning.
    When using hyperparameter optimization with Hyperopt, it is essential to initialize and compile the model 
    within each trial. weights, history and predictions are saved for model evaluation and item recommendaton. 
    The weights of e.g the concatenated model are a list of length 6: [user_embedding weights, item_embedding weights,
    dense1 weights, dense1 biases, dense3 weights, dense3 biases].

 
    :param model_type: type of collaborative filtering model ('dot', 'concatenated', 'deep' or 'hybrid').
    :param embedding_size: dimensionality of the embedding vectors.
    :param batch_size: number of samples per gradient update.
    :param epochs: number of epochs to train the model.
    :param validation_split: fraction of the train data to be used as validation data.
    :param even_combinations: dataframe with balanced number of positive (relevant) and negative ratings.
    :param train: dataframe containing train data.
    :param test: dataframe containing test data.
    :param meta_columns: list of columns containing additional features for hybrid model.
    :params test_size: the fraction of data used for the test data (not seen during training).
    
    :return: tuple containing the dataframe with predicted ratings and training history.
    """
    model = initialize_and_compile_model(model_type=model_type, embedding_size=embedding_size, 
                                         even_combinations=even_combinations)
    
    inputs_train = prepare_inputs(data=train, model_type=model_type, meta_columns=meta_columns)
    inputs_test = prepare_inputs(data=test, model_type=model_type, meta_columns=meta_columns)
  
    history = model.fit(inputs_train,
                         np.array(train['rating']),
                         batch_size=batch_size, 
                         epochs=epochs, 
                         validation_split=validation_split,
                         shuffle=True)
    
    test_preds = model.predict(inputs_test)
    test['rating_pred'] = test_preds

    with open(MODEL_HISTORIES_DIR+f'history_{model_type}_{embedding_size}_{epochs}_{batch_size}_{test_size:.2f}_{validation_split:.2f}.pkl', 'wb') as f:
      pickle.dump(history.history, f)
   
    with open(MODEL_WEIGHTS_DIR+f'weights_{model_type}_{embedding_size}_{epochs}_{batch_size}_{test_size:.2f}_{validation_split:.2f}.pkl', 'wb') as f:
      pickle.dump(model.get_weights(), f)

    with open(PROCESSED_DATA_DIR+f'test_out_{model_type}_{embedding_size}_{epochs}_{batch_size}_{test_size:.2f}_{validation_split:.2f}.pkl', 'wb') as f:                
      pickle.dump(test, f)

    logger.info(f'weights and history have been saved.')

    return test


def objective(params: dict, train: pd.DataFrame = None, test: pd.DataFrame = None, even_combinations: pd.DataFrame = None) -> float:
    """
    Objective function to minimize during hyperparameter tuning. 
    The reason for splitting the train data into train and validation sets is to assess the performance of the model during hyperparameter tuning.
    validation_split is not used to split the data into train and validation sets. Instead, it is used to specify the fraction of the training data to be used as a validation set during the training process.
    
    :param params: dictionary containing the hyperparameters to be optimized.
    :param even_combinations: dataframe with balanced number of positive (relevant) and negative ratings.
    :param train: dataframe containing train data.
    :param test: dataframe containing test data.
    in order to assess the performance of the model during hyperparameter tuning.
    :return: negative accuracy score to minimize.
    """
    embedding_size = round(params['embedding_size'])#round is needed as the quniform function in space generates floats but ints are needed
    batch_size = round(params['batch_size'])
    epochs = round(params['epochs']) 
    model_type = params['model_type']
    test_size = params['test_size']
    validation_split = params['validation_split']
    
    if train is None or test is None or even_combinations is None:
      with open(PROCESSED_DATA_DIR+'train_test_even_combinations.pkl', 'rb') as f:
            train, test, even_combinations = pickle.load(f)

    train_data, val_data = train_test_split(train, test_size=test_size, random_state=42)
    
    test = build_and_train_model(
        even_combinations=even_combinations,
        train=train_data,
        test=val_data,
        embedding_size=embedding_size,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        model_type=model_type,
        test_size=test_size
    )
    
    accuracy = accuracy_score(val_data['rating'], (test['rating_pred'] >= 0.5).astype(int))
    return -accuracy  #minimize the negative of accuracy


def get_best_parameters(space_vals: dict = SPACE_VALS, max_evals: int = 10, train: pd.DataFrame = None, 
                        test: pd.DataFrame = None, even_combinations: pd.DataFrame = None) -> dict:
    """
    Perform hyperparameter tuning using Hyperopt and save the best parameters and trials.
    
    :param space_vals: dictionary defining the search values for the hyperparameters.
    :param max_evals: maximum number of evaluations for hyperparameter tuning.
    :param even_combinations: dataframe with balanced number of positive (relevant) and negative ratings.
    :param train: dataframe containing train data.
    :param test: dataframe containing test data.
    :return: none.
    """
    space = {'embedding_size': hp.quniform('embedding_size', space_vals['embedding_size'][0], space_vals['embedding_size'][1], 1),
             'batch_size': hp.quniform('batch_size', space_vals['batch_size'][0], space_vals['batch_size'][1], 1),
             'epochs': hp.quniform('epochs', space_vals['epochs'][0], space_vals['epochs'][1], 1),
             'model_type': hp.choice('model_type', space_vals['model_type']),
             'test_size': hp.uniform('test_size', space_vals['test_size'][0], space_vals['test_size'][1]),
             'validation_split': hp.uniform('validation_split', space_vals['validation_split'][0], space_vals['validation_split'][1])
             }
    
    #load previous trials if available
    try:
        with open(MODEL_PARAMS_DIR+'trials.json', 'rb') as f:
            trials = pickle.load(f)
    except FileNotFoundError:
        trials = Trials()

    partial_objective = partial(objective, train=train, test=test, even_combinations=even_combinations)
    best = fmin(fn=partial_objective,
                space=space,   
                algo=tpe.suggest, 
                max_evals=max_evals, 
                trials=trials)

    best = {k: (v if k != 'model_type' else space_vals[k][v]) for k, v in best.items()}

    print("Best hyperparameters:", best)
    logger.info("Best hyperparameters:", best)

    #save trials and best parameters
    with open(MODEL_PARAMS_DIR+'trials.json', 'wb') as f:
        pickle.dump(trials, f)

    with open(MODEL_PARAMS_DIR+'best_params.json', 'w') as json_file:
       json.dump(best, json_file)


if __name__ == "__main__":     
      #hyper parameter tuning
      get_best_parameters(space_vals=SPACE_VALS, max_evals=MAX_EVALS, train=None, 
                         test=None, even_combinations=None)

      #build a specific model
      with open(PROCESSED_DATA_DIR+'train_test_even_combinations.pkl', 'rb') as f:
        train, test, even_combinations = pickle.load(f)

      test_size=0.3
      validation_split=0.3
      embedding_size=23
      batch_size=200
      epochs=5
      train_data, val_data = train_test_split(train, test_size=test_size, random_state=42)
      
      _ = build_and_train_model(model_type='hybrid',
                          embedding_size=embedding_size,
                          batch_size=batch_size, 
                          epochs=epochs,
                          even_combinations=even_combinations,
                          train=train_data, 
                          test=val_data,
                          validation_split=validation_split,
                          meta_columns=META_COLUMNS,
                          test_size=test_size) 


#%%
    