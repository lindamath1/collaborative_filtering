#%%
import logging
import os
import time
import utils.parameters as params

def setup_logging(log_file='app.log'):
    if not os.path.exists(log_file):
        open(log_file, 'w').close()

    logging.basicConfig(filename=log_file, level=logging.DEBUG,
                        format='%(asctime)s [%(levelname)s]: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    return logging.getLogger()


logger = setup_logging()

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Function '{func.__name__}' took {execution_time:.2f} seconds to execute.")
        return result
    return wrapper


def get_directory_paths() -> list:
    """
    Get a list of all directory paths defined in the parameters module.

    :return: list of directory paths.
    """
    return [
        params.DATA_DIR,
        params.RAW_DATA_DIR,
        params.PROCESSED_DATA_DIR,
        params.MODELS_DIR,
        params.MODEL_HISTORIES_DIR,
        params.MODEL_WEIGHTS_DIR,
        params.MODEL_PARAMS_DIR,
        params.EVALUATION_DIR,
        params.EVALUATION_FIGURES_DIR
    ]



def initialize_directories(directories: list) -> None:
    """
    Initialize necessary directories if they do not exist.

    :param directories: list of directory paths to be created.
    """
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Directory {directory} created.")
        else:
            logger.info(f"Directory {directory} already exists.")


#%%
    