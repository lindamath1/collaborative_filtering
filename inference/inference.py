#%%
import pandas as pd
import numpy as np
import pickle
from models.models import initialize_and_compile_model, ClassificationModel
from utils.parameters import TOP_RECOMMENDATIONS, META_COLUMNS, MODEL_WEIGHTS_DIR
from utils.helpers import setup_logging


logger = setup_logging()


def load_trained_model(model_type: str, embedding_size: int, batch_size: int, epochs: int, validation_split: float, 
                       test_size: float, weights: list, even_combinations: pd.DataFrame = None) -> ClassificationModel:
    """
    Function to load the trained model.

    :param model_type: type of collaborative filtering model ('dot', 'concatenated', 'deep' or 'hybrid').
    :param embedding_size: dimensionality of the embedding vectors.
    :param batch_size: number of samples per gradient update.
    :param epochs: number of epochs to train the model.
    :param validation_split: fraction of the train data to be used as validation data.
    :params test_size: the fraction of data used for the test data (not seen during training).
    :param even_combinations: dataframe with balanced number of positive and negative ratings.
    :param weights: list containing user and item embeddings, and optionally, embeddings for meta data (if model_type is).
    :return: the trained model.
    """
    with open(MODEL_WEIGHTS_DIR + f'weights_{model_type}_{embedding_size}_{epochs}_{batch_size}_{test_size:.2f}_{validation_split:.2f}.pkl', 'rb') as f:
        weights = pickle.load(f)
    
    model = initialize_and_compile_model(model_type=model_type, embedding_size=embedding_size, 
                                         even_combinations=even_combinations)

    #explicitly build the model with the expected input shape, s.t the weights can be set
    if model_type == 'hybrid':
        model.build([(None, 1), (None, 1), (None, len(meta_columns))])
    else:
        model.build([(None, 1), (None, 1)])

    model.set_weights(weights)

    return model


def generate_recommendations(model: ClassificationModel, even_combinations: pd.DataFrame, model_type: str, 
                             meta_columns: list = META_COLUMNS, top_recommendations: int = TOP_RECOMMENDATIONS) -> dict:
    """
    Generate top N recommendations for each user using the trained model. Also, make sure the already existing
    user-item pairs are not considered for recommendation by setting their score to -inf.

    :param model: trained model used for generating recommendations.
    :param even_combinations: dataframe with balanced number of positive and negative ratings.
    :param model_type: type of collaborative filtering model ('dot', 'concatenated', 'deep', or 'hybrid').
    :param meta_columns: list of columns containing additional features for hybrid model.
    :param top_recommendations: number of top recommendations to generate for each user.

    :return: dictionary with user IDs as keys and lists of recommended item IDs as values.
    """
    max_user_id = even_combinations['user_id'].max()
    max_item_id = even_combinations['item_id'].max()
    
    all_users = np.array(list(range(max_user_id + 1)))
    all_items = np.array(list(range(max_item_id + 1)))
    
    user_ids, item_ids = np.meshgrid(all_users, all_items, indexing='ij')
    user_ids = user_ids.flatten()
    item_ids = item_ids.flatten()

    if model_type == 'hybrid' and meta_columns:
        meta_data = even_combinations[meta_columns].values
        meta_data_expanded = np.tile(meta_data, (max_user_id + 1, 1))
        predictions = model.predict([user_ids, item_ids] + [meta_data_expanded])
    else:
        predictions = model.predict([user_ids, item_ids])

    #reshape predictions to user-item matrix
    prediction_matrix = predictions.reshape((max_user_id + 1, max_item_id + 1))

    #create a mask for already existing user-item pairs
    existing_mask = even_combinations.pivot(index='user_id', columns='item_id', values='existing_user_item_pair').fillna(False).values
    prediction_matrix[existing_mask] = float('-inf')

    #get top n recommendations for each user
    top_recommendations_indices = np.argpartition(-prediction_matrix, top_recommendations, axis=1)[:, :top_recommendations]
    top_recommendations_scores = np.take_along_axis(prediction_matrix, top_recommendations_indices, axis=1)
    sorted_top_recommendations_indices = np.argsort(-top_recommendations_scores, axis=1)
    top_recommendations_sorted_items = np.take_along_axis(top_recommendations_indices, sorted_top_recommendations_indices, axis=1)

    recommendations = {user_id: top_recommendations_sorted_items[user_id].tolist() for user_id in all_users}



    return recommendations


if __name__ == "__main__":
    from evaluation.evaluation import load_model_data
    from utils.parameters import PROCESSED_DATA_DIR

    with open(PROCESSED_DATA_DIR+'train_test_even_combinations.pkl', 'rb') as f:
        _, _, even_combinations = pickle.load(f)

    test, weights, model_type, embedding_size, epochs, batch_size, test_size, validation_split = load_model_data(model_params=None)

    model = load_trained_model(model_type=model_type, embedding_size=embedding_size, batch_size=batch_size, epochs=epochs, 
                      validation_split=validation_split, test_size=test_size, weights=weights, 
                      even_combinations=even_combinations)

    recommendations = generate_recommendations(model=model, even_combinations=even_combinations, 
                                               model_type=model_type, meta_columns=META_COLUMNS, 
                                               top_recommendations=TOP_RECOMMENDATIONS) 
 




#%%
    