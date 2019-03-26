# senna-replica
A tensorflow implementation of the senna named entity recognizer (NER).

## Requirements:
1. tensorflow v1.13.1
3. matplotlib
4. IPython.display (for tensorboard visualization in a jupyter notebook)
5. nltk (for sentence and word tokenizing only)

## Usage:
This repository has the following files and tree structure:
```
    root
     |---constants.py
     |---README.md
     |---senna_ner.ipynb
     |---senna_ner.py
     |---utils
     |      |
     |      |---utils.py
     |
     |---"project_name"
            |
            |---corpus
            |      |
            |      |---train.txt
            |      |---testa.txt
            |      |---testb.txt
            |
            |---wdvecs
                   |
                   |---embeddings.txt
                   |---words_list.txt
                    
    ```
(You should replace "project_name" with your own name in this tree and in the appropriate cell in `senna_ner.ipynb`.)

All files in this tree can be used as they are except the following:

1. The files `train.txt`, `testa.txt`, and `testb.txt` supplied here are incomplete and are included only to demonstrate their required contents and structure. As their names imply, these are one text file for training and two text files for testing. Each file must have the following structure: every line corresponds one-to-one with a single token in the original corpus, and from left-to-right, every line reads `"token" "tag"`, where "tag" is the named entity tag for the token. The tokens are ordered as they appear in the corpus, and a single empty line seperates each pair of consecutive sentences in the corpus.  The tag must be formatted as described here: `https://www.clips.uantwerpen.be/conll2003/ner/`. For now, the "type" part of the tag (after the dash) must be one of 'LOC' (for "location"), 'PER' (for "person"), 'ORG' (for "organization"), and 'MISC' (for "miscellaneous"), but this constraint will be removed in a future version. Also, the "position" part of the tag (before the dash) can be in either IOB format or IOBES format, as explained here: `https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)`. (At present, IOB tags are converted to IOBES tags for training, and only IOBES tags are returned during inference.) For example, you can use the Reuter's corpus (https://trec.nist.gov/data/reuters/reuters.html) from the CoNLL-2003 NER challenge (https://www.clips.uantwerpen.be/conll2003/ner/) for your text files. Indeed, after removing all lines `-DOCSTART- -X- -X- O` and removing the second and third tags on each line (as they are not named entity tags) from the `train`, `testa`, and `testb` files of the Reuter's corpus, we obtain files formatted as described above, which we may use for `train.txt`, `testa.txt`, and `testb.txt` respectively.

2. The files `words_list.txt` and `embeddings.txt` supplied here are incomplete and are included only to demonstrate their required contents and structure. These are files for the word embedding model. The first file `words_list.txt` contains a single token (word, contraction, punctuation, etc.) on each line, and the file `embeddings.txt` contains a single n-dimensional vector on each line. The vector is represented as n floats pairwise consecutively separated by a single blank space, and n is constant throughout the text file. For all natural numbers x, line x of `embeddings.txt` if it exists is the embedding vector of the token on line x of `words_list.txt`, and vice versa. `word_list.txt` must have two special tokens `PADDING` AND `UNKNOWN`, with corresponding vectors in `embeddings.txt`. `PADDING` refers to "blank" tokens attached to the beginning and end of each sentence and needed to construct the full "context vector" for the first and last tokens of each sentence. Tokens encountered in the corpus that are not in `words_list.txt` are replaced by `UNKNOWN` during training and inference. You can find an example of such a pair of files `words_list.txt` and `embeddings.txt` (by different names), for example, from here: `https://ronan.collobert.com/senna/.` Alternatively, you can use `gensim.models.Word2Vec` for your word embeddings, in which case the directory `wd_vecs` is replaced by the word embedding model itself. See `https://radimrehurek.com/gensim/models/word2vec.html`. (If you do use `gensim.models.Word2Vec`, then you may need to find a way to insert the special `PADDING` AND `UNKNOWN` tokens into the model.)
    
Once these files are in place, run the enclosed `senna_ner.ipynb` notebook in a tensorflow environment.

## Tensorflow graph:

From the enclosed `senna_ner.ipynb` notebook, you can run tensorboard to see the entire computational graph.

A look at the entire tensorflow graph:&nbsp;

![tensorflow graph](other/images/tensorflow_graph.png?raw=true "tensorflow_graph")

A look at the fully connected neural network:&nbsp;

![neural network](other/images/neural_network.png?raw=true "neural_network")
