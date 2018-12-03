import numpy as np
import time

# load the whole embedding into memory
def load_glove_embeddings(filepath:str, printevery:int=None)->dict:
    """
    Function which will read the GloVe pretrained word embeddings and return a dictionary mapping word to embedding vector.

    Parameters
    ----------
    filepath : str
        Absolute filepath to the GloVe txt file.
    printevery : int
        (Optional) will print time every line if desired.

    """
    embeddings_index = dict()
    line_count = 0
    start_time = time.time()

    # open the file
    with open(filepath) as f:

        # iterate through the lines in the file
        for line in f:

            # if 'printevery' is activated
            if printevery is not None:
                # increase count
                line_count += 1
                # if count is multiple of printevery
                if line_count % printevery == 0:
                    # print time
                    print ('{}\t lines:\t{}s'.format(line_count, time.time() - start_time))
                    # reset 'start_time'
                    start_time = time.time()

            # the file is space-seperated value, with the 'word' at the first index
            values = line.split()

            # extract the word
            word = values[0]

            # create array from the remaining values (the actual embedding weights)
            coefs = np.asarray(values[1:], dtype=np.float16)  # using float16 to save space!

            # set key(word) to value(weights array)
            embeddings_index[word] = coefs

    # print the stats
    print('Loaded {} words.'.format(len(embeddings_index)))

    # return the dictionary
    return embeddings_index


# Function to convert dictionary to numpy array
def dict_to_np_array(dictionary: dict)->np.array:
    """
    Function to convert dictionary to numpy array.
    (Used mostly to create a matrix of weights to be used for layer initialization in deep learning).

    :param dictionary:
        Dictionary containing ints as keys and numpy arrays as values.
    :return:
        2D numpy array, where each row corresponds to the numpy array at the corresponding index within the dictionary.
    """
    # get arbitrary array from dict for sizing
    arb_array = next(iter(dictionary.values())).flatten()
    
    num_rows: int = len(dictionary)
    num_columns: int = arb_array.shape[0]

    # create empty array of zeros
    array_to_return = np.zeros((num_rows, num_columns))

    # iter through dictionary and update rows in array_to_return
    for index, array in dictionary.items():
        array_to_return[index] = array

    # return the array
    return array_to_return
