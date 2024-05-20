#%%
from data.preprocess import get_datasets
from models.models import get_best_parameters
from evaluation.evaluation import (visualize_validation_loss_and_accuracy_of_tried_models, load_model_data,
evaluate_model)
from inference.inference import load_trained_model, generate_recommendations
from utils.helpers import setup_logging, get_directory_paths, initialize_directories
from utils.parameters import (RATING_THRESHOLD, ADD_DISSIMILARITY, SPLIT_METHOD, TEST_DAYS, TEST_USER_RATIO, MAX_k,
                              META_COLUMNS, COMMON_TITLE_SUBSTRING, MAX_EVALS, SPACE_VALS, TOP_RECOMMENDATIONS)

logger = setup_logging()


def main(rating_threshold: int = RATING_THRESHOLD,
         add_dissimilarity: bool = ADD_DISSIMILARITY,
         split_method: str = SPLIT_METHOD,
         test_days: int = TEST_DAYS,
         test_user_ratio: float = TEST_USER_RATIO,
         meta_columns: list = META_COLUMNS,
         space_vals: dict = SPACE_VALS,
         max_evals: int = MAX_EVALS,
         max_k: int = MAX_k,
         common_title_substring: str = COMMON_TITLE_SUBSTRING,
         top_recommendations: int = TOP_RECOMMENDATIONS) -> None:
     """
     Main function to run the recommender system pipeline.

     :param rating_threshold: threshold for considering a rating as positive/relevant.
     :param add_dissimilarity: ensures that negative samples have dissimilar genres compared to any of the genres 
     associated with the positive samples.
     :param split_method: method to split the dataset ('user' or 'temporal').
     :param test_days: number of test days (for the temporal split).
     :param test_user_ratio: ratio of test users (for user-based split).
     :param meta_columns: list of columns containing additional features for hybrid models.
     :param space_vals: dictionary defining the search values for the hyperparameters.
     :param max_evals: maximum number of evaluations for hyperparameter tuning.
     :param max_k: the max number of k to be used to plot the map@k for.
     :param common_title_substring: partial title string in common.
     :param top_recommendations: number of top recommendations to generate for each user.
     :return: dictionary with user IDs as keys and lists of recommended item IDs as values.
     """ 
     initialize_directories(directories=get_directory_paths())

     train, test, even_combinations = get_datasets(rating_threshold=rating_threshold,
                                                    add_dissimilarity=add_dissimilarity,
                                                    split_method=split_method,
                                                    test_days=test_days,
                                                    test_user_ratio=test_user_ratio,
                                                    meta_columns=meta_columns)

     get_best_parameters(space_vals=space_vals, 
                         max_evals=max_evals, 
                         train=train, 
                         test=test, 
                         even_combinations=even_combinations)

     visualize_validation_loss_and_accuracy_of_tried_models()  

     test, weights, model_type, embedding_size, epochs, batch_size, test_size, validation_split = load_model_data(model_params=None)

     evaluate_model(test=test, 
                    weights=weights, 
                    model_type=model_type, 
                    embedding_size=embedding_size, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    test_size=test_size, 
                    validation_split=validation_split, 
                    even_combinations=even_combinations,
                    max_k=MAX_k,
                    common_title_substring=common_title_substring)  

     model = load_trained_model(model_type=model_type, 
                                embedding_size=embedding_size, 
                                batch_size=batch_size, 
                                epochs=epochs, 
                                validation_split=validation_split,
                                test_size=test_size, 
                                weights=weights, 
                                even_combinations=even_combinations)

     recommendations = generate_recommendations(model=model, 
                                                even_combinations=even_combinations, 
                                                model_type=model_type, 
                                                meta_columns=meta_columns, 
                                                top_recommendations=top_recommendations) 

     return recommendations
 
     
if __name__ == "__main__":
     recommendations = main(rating_threshold=RATING_THRESHOLD,
                            add_dissimilarity=ADD_DISSIMILARITY,
                            split_method=SPLIT_METHOD,
                            test_days=TEST_DAYS,
                            test_user_ratio=TEST_USER_RATIO,
                            meta_columns=META_COLUMNS,
                            space_vals=SPACE_VALS,
                            max_evals=MAX_EVALS,
                            max_k=MAX_k,
                            common_title_substring=COMMON_TITLE_SUBSTRING,
                            top_recommendations=TOP_RECOMMENDATIONS)

#%%
    