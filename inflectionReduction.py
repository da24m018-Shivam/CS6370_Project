# Add your import statements here
import nltk
from nltk.stem import PorterStemmer

class InflectionReduction:

    def __init__(self):
        # Initialize Porter Stemmer
        self.porter_stemmer = PorterStemmer()
        # Ensure necessary NLTK resources are available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def reduce(self, text):
        """
        Perform stemming on the input text using Porter's Stemmer.

        Parameters
        ----------
        text : list
            A list of lists where each sub-list is a sequence of tokens representing a sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of stemmed tokens
        """
        reduced_text = []
        
        for sentence in text:
            if not sentence:
                reduced_text.append([])
                continue
                
            # Apply Porter stemmer to each token in the sentence
            stemmed_sentence = [self.porter_stemmer.stem(token) for token in sentence]
            reduced_text.append(stemmed_sentence)
                
        return reduced_text