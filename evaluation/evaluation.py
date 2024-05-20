#%%
import os
import numpy as np
import pandas as pd
import json
import pickle
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from utils.helpers import setup_logging
from utils.parameters import (PROCESSED_DATA_DIR, MODEL_HISTORIES_DIR, MODEL_WEIGHTS_DIR, MODEL_PARAMS_DIR, EVALUATION_DIR, EVALUATION_FIGURES_DIR, 
                              k, MAX_k, COMMON_TITLE_SUBSTRING)

logger = setup_logging()


def get_map_at_k(k: int = k, test: pd.DataFrame = None) -> float: 
    """
    Calculate Mean Average Precision at k (MAP@k) for a given test dataframe. 

    :param k: number of top predictions to consider.
    :param test: dataframe containing the test data. 
    :return: mean average precision at k.
    """
    threshold = 1 #implicit ratings {0, 1}
    test = (test
               .assign(relevant_items=np.isin(test.index, np.where(test['rating'] >= threshold)[0]))
               .sort_values(by=['user_id', 'rating_pred'], ascending=False)
               .groupby('user_id').head(k)
               .assign(ratings_cumcount=lambda df: df.groupby('user_id').cumcount() + 1,
                       precision_at_k=lambda df: df.groupby('user_id')['relevant_items'].cumsum() / df.ratings_cumcount)
           )
    avg_precision_at_k = (test['precision_at_k'] * test['relevant_items']).groupby(test['user_id']).sum() / test.groupby('user_id')['relevant_items'].sum()
    
    #mean over all user_ids. fillna for all users where the sum of relevant items = 0 (division by 0 -> nan)
    map_at_k = avg_precision_at_k.fillna(0).mean()

    print(f'MAP@{k} Score on Test Set: {map_at_k}')
    logger.info(f'MAP@{k} Score on Test Set: {map_at_k}')

    return map_at_k


def get_map_at_k_sklearn(k: int, test: pd.DataFrame) -> float:
    """
    Calculate Mean Average Precision at k (MAP@k) for a given test dataframe using sklearn.metrics.average_precision_score.

    :param k: number of top predictions to consider.
    :param test: dataframe containing the test data.
    :return: mean average precision at k.
    """
    threshold = 1 #implicit ratings {0, 1}

    test = test.assign(relevant_items=np.isin(test.index, np.where(test['rating'] >= threshold)[0]))
    test = test.sort_values(by=['user_id', 'rating_pred'], ascending=False).groupby('user_id').head(k)

    user_map_scores = []
    for user_id, group in test.groupby('user_id'):
        relevant_items = group['relevant_items'].values
        rating_preds = group['rating_pred'].values
        if any(relevant_items):  
            user_map_scores.append(average_precision_score(relevant_items, rating_preds))
        else:
            user_map_scores.append(0)
            logger.warning(f"No relevant items found for user {user_id}. Appending 0 to average precision scores.")
       
    map_at_k = np.mean(user_map_scores)
    print(f'MAP@{k} Score on Test Set (sklearn): {map_at_k}')
    logger.info(f'MAP@{k} Score on Test Set (sklearn): {map_at_k}')

    return map_at_k


def get_cosine_similarity_of_similar_items(item_embeddings: np.ndarray, common_title_substring: str, 
                                           even_combinations: pd.DataFrame) -> float:
         """
         Compute the mean absolute cosine similarity of items with similar titles (excluding the diagonal of the
         cosine similarity matrix).

         :param item_embeddings: embeddings of all items.
         :param common_title_substring: a partial string shared by titles of items.
         :param even_combinations: dataframe with balanced number of positive (relevant) and negative ratings.
         :return: mean absolut similarity of items with similar titles.
         """
         items_to_compare = (even_combinations[even_combinations.title.str.contains(common_title_substring)]
                                    .drop_duplicates(subset=['title']).item_id.unique())
         
         item_embeddings_to_compare = item_embeddings[items_to_compare]

         cos_sim_matrix = cosine_similarity(item_embeddings_to_compare)

         np.fill_diagonal(cos_sim_matrix, np.nan)
         mean_abs_similarity = np.nanmean(np.abs(cos_sim_matrix))
         print(f'mean_abs_similarity of {common_title_substring} items: ', mean_abs_similarity)
         logger.info(f'the mean cosine similarity (excluding the diagonal) of items with '
                     f'{common_title_substring} in the title is {mean_abs_similarity}')
         
         return mean_abs_similarity
         

def visualize_embeddings(ax1: plt.Axes, ax2: plt.Axes, item_embeddings: np.ndarray, even_combinations: pd.DataFrame, 
                         common_title_substring: str = COMMON_TITLE_SUBSTRING, mean_abs_cos_similarity: float = 0) -> None:
         """
         Visualize embeddings using t-sne.

         :param ax1: axes object to plot first embeddings visualization on.
         :param ax2: axes object to plot second embeddings visualization on.
         :param item_embeddings: embeddings of all items.
         :param common_title_substring: a partial string shared by titles of items (only used for first visualisation).
         :param even_combinations: dataframe with balanced number of positive (relevant) and negative ratings.
         :param mean_abs_cos_similarity: mean absolut similarity of items with similar titles.
         :return: none.
         """
         item_tsne = TSNE(perplexity=30).fit_transform(item_embeddings)
         #user_tsne = TSNE(perplexity=30).fit_transform(user_embeddings)

         #visualise common_title_substring in all embeddings
         items_to_compare = (even_combinations[even_combinations.title.str.contains(common_title_substring)]
                                          .drop_duplicates(subset=['title']).item_id.unique())
         
         to_compare_x = item_tsne[items_to_compare, 0]
         to_compare_y = item_tsne[items_to_compare, 1]
         
         ax1.scatter(item_tsne[:, 0], item_tsne[:, 1], alpha=0.1)
         ax1.scatter(to_compare_x, to_compare_y, alpha=1, color='r')
         ax1.set_title(f'{common_title_substring} items')
         ax1.text(0.97, 0.03, f'mean abs cosine similarity: {mean_abs_cos_similarity:.2f}', 
                  transform=ax1.transAxes, horizontalalignment='right', verticalalignment='bottom',
                  bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round,pad=0.5'))
         

         #visualise the number of ratings per embedding/item
         items = (even_combinations[['item_id', 'no_ratings']]
                  .drop_duplicates()
                  .sort_values(by='item_id'))
         
         items['item_tsne_x'] = item_tsne[:-1, 0] #the last index is just a placeholder therefore remove
         items['item_tsne_y'] = item_tsne[:-1, 1]

         scatter = ax2.scatter(items['item_tsne_x'], items['item_tsne_y'], 
                           s=10,
                           c=np.log(items['no_ratings']), #apply logarithmic scaling
                           cmap=plt.cm.get_cmap('Blues'))
         ax2.set_title('item popularity (number of ratings)')
         plt.colorbar(scatter, ax=ax2)
        

def plot_map_at_k(ax: plt.Axes, test: pd.DataFrame = None, max_k: int = MAX_k) -> None:
    """
    Plot Mean Average Precision at k (MAP@k) for various values of k. Ideally, the MAP@k curve increases
    first and then plateaus as k increases. If the curve flattens out after a certain point, it suggests 
    that the algorithm's recommendations become less relevant as more items are considered which again 
    suggests a limit to the algorithm's effectiveness beyond a certain number of recommendations.
 
    :param ax: axes object to plot on.
    :param test: dataframe containing the test data with predicted ratings.
    :param max_k: maximum value of k to consider.
    :return: none.
    """
    k_values = np.arange(1, max_k + 1)
    map_at_k_values = [get_map_at_k(k, test) for k in k_values]

    ax.set_xlabel('k')
    ax.set_ylabel('MAP@k')
    ax.set_title('Mean Average Precision at k')
    ax.grid(True)
    ax.plot(k_values, map_at_k_values)


def plot_roc_curve(ax: plt.Axes, test: pd.DataFrame = None) -> None:
    """
    Plot the ROC curve and compute the AUC.

    :param ax: axes object to plot on.
    :param test: dataframe containing the test data.
    :return: none.
    """
    test = test.assign(relevant_items=test['rating'] >= 1)
    fpr, tpr, _ = roc_curve(test['relevant_items'], test['rating_pred'])
    roc_auc = auc(fpr, tpr)
    logger.info('ROC curve (area = %0.2f)' % roc_auc)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.grid(True)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.legend(loc='lower right')


def plot_precision_recall_curve(ax: plt.Axes, test: pd.DataFrame = None) -> None:
    """
    Plot the precision-recall curve.

    :param ax: axes object to plot on.
    :param test: dataframe containing the test data.
    :return: none.
    """
    test = test.assign(relevant_items=test['rating'] >= 1)
    precision, recall, _ = precision_recall_curve(test['relevant_items'], test['rating_pred'])
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall')
    ax.grid(True)
    ax.plot(recall, precision, marker='.')


def plot_confusion_matrix(ax: plt.Axes, test: pd.DataFrame = None) -> None:
    """
    Plot the confusion matrix.

    :param ax: axes object to plot on.
    :param test: dataframe containing the test data.
    :return: none.
    """
    test = test.assign(relevant_items=test['rating'] >= 1)
    cm = confusion_matrix(test['relevant_items'], test['rating_pred'] >= 0.5)

    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax)

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')


def visualize_validation_loss_and_accuracy_of_tried_models() -> None:
    """
    Plot validation loss and accuracy for the models tested and saved by hyperopt.

    :retrun: none.
    """
    model_colors = {'dot': 'blue', 'concatenated': 'orange', 'deep': 'green', 'hybrid': 'red'}
    model_type_embedding_j = {'dot': 1, 'concatenated': 3, 'deep': 5, 'hybrid': 7}

    fig, axes = plt.subplots(9, 2, figsize=(16, 30))
    fig.suptitle("Hyperopt's tried models \n\n",  fontsize=16, fontweight='bold')

    model_type_appeared = {} 
    embedding_sizes = []  
    embedding_size_color_map = {} 
    batch_sizes = [] 
    batch_size_color_map = {}  

    for model_file in os.listdir(MODEL_HISTORIES_DIR):
        if (model_file.endswith('.pkl')):
            model_params = model_file.replace('.pkl','').split('_')
            model_type = model_params[1]
            embedding_size = int(model_params[2])
            epochs = int(model_params[3])
            batch_size = int(model_params[4])
           

            x = np.arange(1, epochs+1, 1)
            

            with open(MODEL_HISTORIES_DIR+model_file, 'rb') as f:
                history = pickle.load(f)

            for metric, i in zip(['loss', 'accuracy'], [0, 1]):
                  if model_type not in model_type_appeared:
                     axes[0, i].plot(x, history[f'val_{metric}'], label=model_type, color=model_colors[model_type])
                     model_type_appeared[model_type] = True
                  else:
                     axes[0, i].plot(x, history[f'val_{metric}'], color=model_colors[model_type])
                  axes[0, i].set_title(f'Validation {metric} (by model type)')
                  axes[0, i].legend()

            #embedding sizes
            if embedding_size not in embedding_sizes:
                embedding_sizes.append(embedding_size)
                color = cm.viridis(embedding_sizes.index(embedding_size) / len(embedding_sizes))
                embedding_size_color_map[embedding_size] = color
                
            for metric, i in zip(['loss', 'accuracy'], [0, 1]):
                axes[model_type_embedding_j[model_type], i].plot(x, history[f'val_{metric}'], color=embedding_size_color_map[embedding_size])
                axes[model_type_embedding_j[model_type], i].set_title(f'{model_type}: validation {metric} (by embedding size)')


            #batch sizes
            if batch_size not in batch_sizes:
                batch_sizes.append(batch_size)
                color = cm.viridis(batch_sizes.index(batch_size) / len(batch_sizes))
                batch_size_color_map[batch_size] = color 
                
            for metric, i in zip(['loss', 'accuracy'], [0, 1]):
                axes[model_type_embedding_j[model_type]+1, i].plot(x, history[f'val_{metric}'], color=batch_size_color_map[batch_size])
                axes[model_type_embedding_j[model_type]+1, i].set_title(f'{model_type}: validation {metric} (by batch size)')

    #add legend with color gradient
    cmap = cm.viridis
    for plot_type in [embedding_sizes, batch_sizes]:
        norm = plt.Normalize(min(plot_type), max(plot_type))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        for j in model_type_embedding_j.values():
            for i in [0,1]:
                if plot_type==embedding_sizes:
                    label = 'Embedding size'
                    plt.colorbar(sm, ax=axes[j, i], label=label, location='right')
                else:
                    label = 'Batch size'
                    plt.colorbar(sm, ax=axes[j+1, i], label=label, location='right')
    

    fig.tight_layout()
    fig.savefig(EVALUATION_FIGURES_DIR+'hyperopt_model_comparison.png')
    fig.show()


def load_model_data(model_params: dict = None) -> None:
    """
    Load data related to a specific model from json and pickle files.

    :param model_params: parameters of the model to load. If None, it loads the best-performing model.
    :return: tuple with test data (containing predictions made by the model), embedding weights (of the model)
    and the model's parameters.
    """
    if model_params is None:
        with open(MODEL_PARAMS_DIR + 'best_params.json', 'r') as json_file:
            model_params = json.load(json_file)

    model_type, embedding_size, epochs, batch_size, test_size, validation_split = (model_params[key] for key in 
                        ['model_type', 'embedding_size', 'epochs', 'batch_size', 'test_size', 'validation_split'])
    embedding_size = int(embedding_size)
    epochs = int(epochs)
    batch_size = int(batch_size)

    test_file = f'test_out_{model_type}_{embedding_size}_{epochs}_{batch_size}_{test_size:.2f}_{validation_split:.2f}.pkl'
    weights_file = f'weights_{model_type}_{embedding_size}_{epochs}_{batch_size}_{test_size:.2f}_{validation_split:.2f}.pkl'

    # Load test data and weights
    with open(PROCESSED_DATA_DIR + test_file, 'rb') as f:
        test = pickle.load(f)

    with open(MODEL_WEIGHTS_DIR + weights_file, 'rb') as f:
        weights = pickle.load(f)

    return test, weights, model_type, embedding_size, epochs, batch_size, test_size, validation_split


def evaluate_model(test: pd.DataFrame, weights: list, model_type: str, embedding_size: int, epochs: int, 
                   batch_size: int, test_size: float, validation_split: float, even_combinations: pd.DataFrame,
                   max_k: int = MAX_k, common_title_substring: str = COMMON_TITLE_SUBSTRING) -> None:
    """
    Visualize the performance of the best model. Subplots of showing map@k, roc curve, precision recall curve, 
    confusion matrix, embedding visualizations and cosine similarity are created.

    :param test: dataframe containing the test data.
    :param weights: list containing user and item embeddings, and optionally, embeddings for meta data (if model_type is).
    :param model_type: type of collaborative filtering model ('dot', 'concatenated', 'deep' or 'hybrid').
    :param embedding_size: dimensionality of the embedding vectors.
    :param batch_size: number of samples per gradient update.
    :param epochs: number of epochs to train the model.
    :param validation_split: fraction of the train data to be used as validation data.
    :params test_size: the fraction of data used for the test data (not seen during training).
    :param even_combinations: dataframe with balanced number of positive and negative ratings.
    :param max_k: the max number of k to be used to plot the map@k for.
    :param common_title_substring: partial title string in common.
    """  
    user_embeddings = weights[0]/np.linalg.norm(weights[0], axis = 1).reshape((-1, 1))
    item_embeddings = weights[1]/np.linalg.norm(weights[1], axis = 1).reshape((-1, 1))
     
    mean_abs_cos_similarity = get_cosine_similarity_of_similar_items(item_embeddings=item_embeddings, 
                                                                     common_title_substring=common_title_substring, 
                                                                     even_combinations=even_combinations)
     
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle(f'Model Performance (model type: {model_type}, embedding size: {embedding_size}, epochs: {epochs}, batch size: {batch_size}) \n\n', 
             fontsize=16, fontweight='bold')

    plot_map_at_k(ax=axes[0,0], test=test, max_k=max_k)
    plot_roc_curve(ax=axes[0,1], test=test)
    plot_precision_recall_curve(ax=axes[1,0], test=test)
    plot_confusion_matrix(ax=axes[1,1], test=test)
    visualize_embeddings(ax1=axes[2,0], ax2=axes[2,1], item_embeddings=item_embeddings, 
                         common_title_substring=common_title_substring, 
                         even_combinations=even_combinations, mean_abs_cos_similarity=mean_abs_cos_similarity)

    fig.tight_layout()
    fig.savefig(EVALUATION_FIGURES_DIR+f'performance_{model_type}_{embedding_size}_{epochs}_{batch_size}_{test_size:.2f}_{validation_split:.2f}.png')
    fig.show()


if __name__ == "__main__":

    with open(PROCESSED_DATA_DIR+'train_test_even_combinations.pkl', 'rb') as f:
        _, _, even_combinations = pickle.load(f)

    results = []
    for model_file in os.listdir(MODEL_HISTORIES_DIR):
        if (model_file.endswith('.pkl')):

            with open(PROCESSED_DATA_DIR+f"test_out_{model_file.replace('history_','')}", 'rb') as f:
                test = pickle.load(f)
            
            model_type = model_file.split('.')[0].split('_')[1]
            embedding_size = model_file.split('.')[0].split('_')[2]

            print(f"model_type: {model_type} and embedding_size: {embedding_size}")
            logger.info(f"model_type: {model_type} and embedding_size: {embedding_size}")

            map_at_k = get_map_at_k(k=k, test=test)

            results.append({'Model_File': model_file, 'MAP@K': map_at_k})
    pd.DataFrame(results).sort_values('MAP@K').to_csv(EVALUATION_DIR+'map_at_k_for_all_tired_models.csv')


    visualize_validation_loss_and_accuracy_of_tried_models()
    test, weights, model_type, embedding_size, epochs, batch_size, test_size, validation_split = load_model_data(model_params=None)  #{'model_type':'dot', 'embedding_size':38, 'epochs':5, 'batch_size':25, 'test_size':0.32, 'validation_split':0.29}
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
                   common_title_substring=COMMON_TITLE_SUBSTRING)   

    

    
#%%
    