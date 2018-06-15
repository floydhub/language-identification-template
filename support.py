import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

############
# Alphabet #
############

# we will use alphabet for text cleaning and letter counting
def define_alphabet():
    base_en = 'abcdefghijklmnopqrstuvwxyz'
    special_chars = ' !?¿¡'
    german = 'äöüß'
    italian = 'àèéìíòóùú'
    french = 'àâæçéèêêîïôœùûüÿ'
    spanish = 'áéíóúüñ'
    czech = 'áčďéěíjňóřšťúůýž'
    slovak = 'áäčďdzdžéíĺľňóôŕšťúýž'
    all_lang_chars = base_en + german +  italian + french + spanish + czech + slovak
    small_chars = list(set(list(all_lang_chars)))
    small_chars.sort() 
    big_chars = list(set(list(all_lang_chars.upper())))
    big_chars.sort()
    small_chars += special_chars
    letters_string = ''
    letters = small_chars + big_chars
    for letter in letters:
        letters_string += letter
    return small_chars,big_chars,letters_string

########
# Plot #
########

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
       
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
       
    FROM: https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

###################################
# Data cleaning utility functions #
###################################

# we will create here several text-cleaning procedures. 
# These procedure will help us to clean the data we have for training, 
# but also will be useful in cleaning the text we want to classify, before the classification by trained DNN

# remove XML tags procedure
# for example, Wikipedia Extractor creates tags like this below, we need to remove them
# <doc id="12" url="https://en.wikipedia.org/wiki?curid=12" title="Anarchism"> ... </doc>
def remove_xml(text):
    return re.sub(r'<[^<]+?>', '', text)

# remove new lines - we need dense data
def remove_newlines(text):
    return text.replace('\n', ' ') 
    
# replace many spaces in text with one space - too many spaces is unnecesary
# we want to keep single spaces between words
# as this can tell DNN about average length of the word and this may be useful feature
def remove_manyspaces(text):
    return re.sub(r'\s+', ' ', text)

# and here the whole procedure together
def clean_text(text):
    text = remove_xml(text)
    text = remove_newlines(text)
    text = remove_manyspaces(text)
    return text

#################
# Preprocessing #
#################

# this function will get sample of texh from each cleaned language file. 
# It will try to preserve complete words - if word is to be sliced, sample will be shortened to full word
def get_sample_text(file_content,start_index,sample_size):
    # we want to start from full first word
    # if the firts character is not space, move to next ones
    while not (file_content[start_index].isspace()):
        start_index += 1
    #now we look for first non-space character - beginning of any word
    while file_content[start_index].isspace():
        start_index += 1
    end_index = start_index+sample_size 
    # we also want full words at the end
    while not (file_content[end_index].isspace()):
        end_index -= 1
    return file_content[start_index:end_index]

# we need only alpha characters and some (very limited) special characters
# exactly the ones defined in the alphabet
# no numbers, most of special characters also bring no value for our classification task
# (like dot or comma - they are the same in all of our languages so does not bring additional informational value)

# count number of chars in text based on given alphabet
def count_chars(text, alphabet):
    alphabet_counts = []
    for letter in alphabet:
        count = text.count(letter)
        alphabet_counts.append(count)
    return alphabet_counts

# process text and return sample input row for DNN
# note that we are counting separatey:
# a) counts of all letters regardless of their size (whole text turned to lowercase letter)
# b) counts of big letters only
# this is because German uses big letters for beginning of nouns so this feature is meaningful
def get_input_row(content,start_index,sample_size, alphabet):
    sample_text = get_sample_text(content,start_index,sample_size)
    counted_chars_all = count_chars(sample_text.lower(), alphabet[0])
    counted_chars_big = count_chars(sample_text, alphabet[1])
    all_parts = counted_chars_all + counted_chars_big
    return all_parts
