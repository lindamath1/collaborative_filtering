#%%
from datetime import timedelta
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import QuantileTransformer
from joblib import Parallel, delayed
import logging
from utils.helpers import setup_logging, timing_decorator
from utils.parameters import (PROCESSED_DATA_DIR, RAW_DATA_DIR,
                              RATING_THRESHOLD, ADD_DISSIMILARITY, SPLIT_METHOD, TEST_DAYS, TEST_USER_RATIO, META_COLUMNS)


logger = setup_logging()


def read_in_data() -> tuple[pd.DataFrame, pd.DataFrame]:
     """
     Read in user rating and item data.

     :return: A tuple containing two pandas dataframes (ratings and items).
     """
     column_names = ['user_id', 'item_id', 'rating', 'timestamp']
     ratings = pd.read_csv(RAW_DATA_DIR+'ratings.dat',sep='::', names=column_names, engine='python')
    
     column_names = ['item_id', 'title', 'genres']
     items = pd.read_csv(RAW_DATA_DIR+"movies.dat", sep = "::", names = column_names, encoding='latin-1', engine='python')
            
     logger.info('datasets were read in.')
     return ratings, items


def preprocess_data(ratings: pd.DataFrame = None, items: pd.DataFrame = None, 
                    rating_treshold: float = RATING_THRESHOLD) -> pd.DataFrame:
     """
     Preprocess data. Also set a threshold for what is considered a positive/relevant rating
     in order to have a binary classification problem for the collaborative filtering model.
     Also, add a column no_items_rated in order to use it later for the hybrid model.

    :param ratings: dataframe containing user ratings.
    :param items: dataframe containing item information.
    :param rating_threshold: threshold for considering a rating as positive/relevant.
    :return: preprocessed dataframe for collaborative filtering with binary classification and
    item_genre_map which is a dictionary mapping item ids to their genres (this is used to generate 
    the neagtive samples)
     """
     item_genre_map = (items.assign(genres=items.genres.str.split('|').fillna('').fillna(''))
                              .set_index('item_id')[['genres']]
                              .to_dict()['genres'])
     
     items = items.drop(columns={'genres'})

     ratings = ratings.assign(timestamp=pd.to_datetime(ratings.timestamp, unit='s'),
                              rating=(ratings.rating >= rating_treshold).astype(int),
                              no_items_rated=ratings.groupby('user_id').rating.transform('count'),
                              no_ratings=ratings.groupby('item_id').rating.transform('count'),
                              existing_user_item_pair=True)

     ratings_items = (ratings.merge(items, on='item_id', how='left')
                           .sample(100000)
                           .sort_values('user_id', ascending=True))
     
     return ratings_items, item_genre_map


@timing_decorator
def get_even_neg_pos_samples(existing_combinations: pd.DataFrame = None, 
                             add_dissimilarity: bool = ADD_DISSIMILARITY, item_genre_map: dict = {}) -> pd.DataFrame:
    """
    Function to balance the number of positive (relevant) and negative ratings in a collaborative 
    filtering dataset. Also redefine the item/user ids so that they go from 1 to nunique.

    :param existing_combinations: dataFrame containing user-item combinations with ratings.
    :param add_dissimilarity: ensures that negative samples have dissimilar genres compared to any of the genres 
    associated with the positive samples.
    :param item_genre_map: dictionary mapping item IDs to their genres.
    :return: DataFrame with an equal number of positive and negative ratings.
    """
    no_positive_ratings = len(existing_combinations[existing_combinations.rating==1])
    no_negative_ratings = len(existing_combinations[existing_combinations.rating==0])

    if no_positive_ratings < no_negative_ratings:
      logger.info(f'no additional negative samples have to be generated in order to balance the positive and negative records.')
      even_combinations = pd.concat([existing_combinations[existing_combinations.rating==1],
                                     existing_combinations[existing_combinations.rating==0].sample(no_positive_ratings)])
      
    if no_positive_ratings > no_negative_ratings:
      logger.info(f'additional negative samples have to be generated in order to balance the positive and negative records.')
      negative_combinations = get_negative_sample(no_negative_ratings_to_be_created=no_positive_ratings-no_negative_ratings, 
                                                  existing_combinations=existing_combinations, add_dissimilarity=add_dissimilarity, 
                                                  item_genre_map=item_genre_map)
    
      even_combinations = pd.concat([negative_combinations, 
                                     existing_combinations], axis=0)
   
    items_map = {id_before: id_after for id_before, id_after in zip(even_combinations.item_id.unique(), np.arange(1, even_combinations.item_id.nunique()+1))}
    users_map = {id_before: id_after for id_before, id_after in zip(even_combinations.user_id.unique(), np.arange(1, even_combinations.user_id.nunique()+1))}
    even_combinations = even_combinations.assign(item_id=even_combinations.item_id.map(items_map),
                                                 user_id=even_combinations.user_id.map(users_map))
 
    return even_combinations


def get_negative_sample(no_negative_ratings_to_be_created: int, existing_combinations: pd.DataFrame = None,
                        add_dissimilarity: bool = ADD_DISSIMILARITY, item_genre_map: dict = {}) -> pd.DataFrame:
    """
    Function to generate a sample of negative user/item combinations for the same length 
    as existing_combinations. Parallelised as it is slow otherwise.

    :param no_negative_ratings_to_be_created: number of positive/relevant samples.
    :param existing_combinations: dataframe with existing user/item combinations.
    :param add_dissimilarity: ensures that negative samples have dissimilar genres compared to any of the genres 
    associated with the positive samples.
    :param item_genre_map: dictionary mapping item IDs to their genres.
    :return: dataframe with negative user/item combinations.
    """
    unique_users = existing_combinations['user_id'].unique()
    unique_items = existing_combinations['item_id'].unique()

    # Parallelize the generation of negative samples across unique users
    negative_samples_list = Parallel(n_jobs=-1)(
        delayed(generate_negative_sample)(user=user, existing_combinations=existing_combinations.rename(columns={'item_id':'item_id_pos'}),
                                          unique_items=unique_items, add_dissimilarity=add_dissimilarity, item_genre_map=item_genre_map)
        for user in unique_users
    )

    # Concatenate the results into a single DataFrame
    negative_combinations = pd.concat(negative_samples_list, ignore_index=True)

    negative_combinations = (negative_combinations.sample(no_negative_ratings_to_be_created)
                             .assign(timestamp=lambda df: np.random.choice(existing_combinations.timestamp, 
                                     size=len(df), replace=True),
                                     rating=0)
                            .merge(existing_combinations[['user_id', 'no_items_rated']].drop_duplicates(), on='user_id', how='left')
                            .merge(existing_combinations[['item_id', 'no_ratings', 'title']].drop_duplicates(), on='item_id', how='left')
                            )
    
    return negative_combinations


def generate_negative_sample(user: int, unique_items: int, existing_combinations: pd.DataFrame = None,
                             add_dissimilarity: bool = ADD_DISSIMILARITY, item_genre_map: dict = {}) -> pd.DataFrame:
    """
    Function to generate negative samples for a specific user by creating combinations of user/items that are not 
    present in existing_combinations. If add_dissimilarity is True negative samples are selected based on 
    dissimilar genres compared to the positive samples (per user). Create a separate NullHandler logger instance. 
    This way, the logger instance will be picklable and can be used while parallel processing.

    :param user: user id for which negative samples are generated.
    :param unique_items: unique items in total.
    :param existing_combinations: dataframe containing existing user/item combinations.
    :param add_dissimilarity: ensures that negative samples have dissimilar genres compared to any of the genres 
    associated with the positive samples of a user.
    :param item_genre_map: dictionary mapping item IDs to their genres.
    :return: dataframe with negative user/item combinations for the specified user.
    """
    null_logger = logging.getLogger(__name__)
    null_logger.addHandler(logging.NullHandler())

    all_user_item_combinations = pd.DataFrame(np.array(np.meshgrid([user], unique_items)).T.reshape(-1, 2), 
                                              columns=['user_id', 'item_id'])

    negative_combinations = (pd.merge(all_user_item_combinations, existing_combinations[['user_id', 'item_id_pos']], 
                            right_on=['user_id', 'item_id_pos'], 
                            left_on=['user_id', 'item_id'], how='left')
                  [lambda df: df['item_id_pos'].isnull()]
                  .drop(columns=['item_id_pos'])
                  )
    
    if add_dissimilarity:
        null_logger.info('select negative samples based on dissmiliar genres.')
        positive_items = existing_combinations[existing_combinations['user_id'] == user]['item_id_pos']
        positive_genres = set()
        for item in positive_items:
            if item in item_genre_map:
                positive_genres.update(item_genre_map[item])
        #check if any of the genres associated with a negative item are similar to any of the genres of the positive items
        dissimilar_negative_combinations = negative_combinations[~negative_combinations['item_id'].map(lambda x: 
                                                        any(genre in item_genre_map.get(x, []) for genre in positive_genres))]
        
        # If all available items have similar genres to positive samples, sample randomly
        if len(dissimilar_negative_combinations) == 0:
            try:
                negative_combinations = negative_combinations.sample(min(len(positive_items), len(unique_items)), replace=False)
            except ValueError:
                null_logger.error('Insufficient negative samples available -> use sampling with replacement.')
                negative_combinations = negative_combinations.sample(min(len(positive_items), len(unique_items)), replace=True)
        else:
            negative_combinations = dissimilar_negative_combinations

    return negative_combinations
   

def temporal_split(even_combinations: pd.DataFrame = None, test_days: int = TEST_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split item_id dataset using timestamps.

    :param even_combinations: dataframe with balanced number of positive (relevant) and negative ratings.
    :param test_days: the number of days that is tested.
    :return: train and test dataframes.
    """
    time_threshold = even_combinations.timestamp.max() - timedelta(days=test_days)
    logger.info(f'the time_threshold for the temporal is {time_threshold}.')
    train = even_combinations[even_combinations['timestamp'] < time_threshold]
    test = even_combinations[even_combinations['timestamp'] >= time_threshold]

    return train, test


def user_split(even_combinations: pd.DataFrame = None, test_user_ratio: int = TEST_USER_RATIO) -> pd.DataFrame:
    """
    Split data into train and test sets based on users.

    :param even_combinations: dataframe with balanced number of positive (relevant) and negative ratings.
    :param test_user_ratio: ratio of users to include in the test set.
    :return: train and test dataframes.
    """
    unique_user_ids = even_combinations['user_id'].unique()
    test_user_ids = np.random.choice(unique_user_ids, 
                                     size=int(round(test_user_ratio * len(unique_user_ids))), 
                                     replace=False)

    train = even_combinations[~even_combinations.user_id.isin(test_user_ids)]
    test = even_combinations[even_combinations.user_id.isin(test_user_ids)]

    return train, test


def remove_records_that_are_not_in_train(train: pd.DataFrame = None, 
                                         test: pd.DataFrame = None,
                                         split_method: str = SPLIT_METHOD) -> pd.DataFrame:
    """
    Remove items or users from the test set that are not in the train set. If the split is
    user based then only item ids are excluded from the test set (that are not in train), if
    it is a temporal split then both, item ids and user ids, are excluded (that are not in train).

    :param train: train set.
    :param test: test set.
    :param split_method: method used for data split ('user' for user split, 
    'temporal' for temporal split).
    :return: cleaned test set.
    """
    if split_method == 'user':
        entity_cols = ['item_id']
    else: 
        entity_cols = ['user_id', 'item_id']
        
    for col in entity_cols:
        train_records = train[col].unique()
        test = test[test[col].isin(train_records)]
        
    return test


def scale_metadata(train: pd.DataFrame = None, 
                   test: pd.DataFrame = None, 
                   meta_columns: list = []) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale metadata columns using QuantileTransformer in order to use it with the hybrid collaborative
    filtering model.

    :param train: train dataset.
    :param test: test dataset.
    :param meta_columns: list of metadata columns to be scaled.
    :return: tuple of scaled train and test datasets.
    """
    train_metadata = train[meta_columns]
    test_metadata = test[meta_columns]

    scaler = QuantileTransformer()

    scaled_train_metadata = scaler.fit_transform(train_metadata)
    scaled_test_metadata = scaler.transform(test_metadata)

    train = train.assign(**{col: train[col].astype('float') for col in meta_columns})
    test = test.assign(**{col: test[col].astype('float') for col in meta_columns})

    train.loc[:, meta_columns] = scaled_train_metadata
    test.loc[:, meta_columns] = scaled_test_metadata

    return train, test


def get_datasets(rating_threshold: int = RATING_THRESHOLD,
                 add_dissimilarity: bool = ADD_DISSIMILARITY,
                 split_method: str = SPLIT_METHOD,
                 test_days: int = TEST_DAYS,
                 test_user_ratio: float = TEST_USER_RATIO,
                 meta_columns: list[str] = META_COLUMNS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Get preprocessed and split datasets for training and testing.

    :param rating_threshold: threshold for considering a rating as positive/relevant.
    :param add_dissimilarity: ensures that negative samples have dissimilar genres compared to any of the genres 
    associated with the positive samples.
    :param split_method: method to split the dataset ('user' or 'temporal').
    :param test_days: number of test days (for the temporal split).
    :param test_user_ratio: ratio of test users (for user-based split).
    :param meta_columns: list of columns containing additional features for hybrid models.

    :return: tuple containing train, test and even_combinations dataframes.
    """
    ratings, items = read_in_data()
    existing_combinations, item_genre_map = preprocess_data(ratings=ratings, items=items, rating_treshold=rating_threshold)
  
    even_combinations = get_even_neg_pos_samples(existing_combinations=existing_combinations, add_dissimilarity=add_dissimilarity,
                                                  item_genre_map=item_genre_map)

    if split_method == 'user':
        logger.info('Using user-based split into train and test sets.')
        train, test = user_split(even_combinations=even_combinations, test_user_ratio=test_user_ratio)
    else:
        logger.info('Using temporal split into train and test sets.')
        train, test = temporal_split(even_combinations=even_combinations, test_days=test_days)

    test = remove_records_that_are_not_in_train(train=train, test=test, split_method=split_method)

    train, test = scale_metadata(train=train, test=test, meta_columns=meta_columns)

    return train, test, even_combinations


if __name__ == "__main__":
     from utils.helpers import get_directory_paths, initialize_directories

     train, test, even_combinations = get_datasets(rating_threshold=RATING_THRESHOLD,
                                                    add_dissimilarity=ADD_DISSIMILARITY,
                                                    split_method=SPLIT_METHOD,
                                                    test_days=TEST_DAYS,
                                                    test_user_ratio=TEST_USER_RATIO,
                                                    meta_columns=META_COLUMNS)
     
     #save the data
     directories = get_directory_paths()
     initialize_directories(directories=directories)

     with open(PROCESSED_DATA_DIR+'train_test_even_combinations.pkl', 'wb') as f:
        pickle.dump((train, test, even_combinations), f)


#%%
    