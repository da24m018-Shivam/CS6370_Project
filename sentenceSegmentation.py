from util import *

# Add your import statements here
import re
import nltk
from nltk.tokenize import sent_tokenize

class SentenceSegmentation():

    def naive(self, text):
        """
        Sentence Segmentation using a Naive Approach

        Parameters
        ----------
        arg1 : str
            A string (a bunch of sentences)

        Returns
        -------
        list
            A list of strings where each string is a single sentence
        """
        # Simple top-down approach using punctuation marks
        if not text:
            return []
            
        # Split on sentence-ending punctuation (., !, ?)
        segmentedText = re.split('[.!?]', text.strip())
        
        # Remove empty strings that might result from split
        segmentedText = [sentence.strip() for sentence in segmentedText if sentence.strip()]

        return segmentedText

    def punkt(self, text):
        """
        Sentence Segmentation using the Punkt Tokenizer

        Parameters
        ----------
        arg1 : str
            A string (a bunch of sentences)

        Returns
        -------
        list
            A list of strings where each string is a single sentence
        """
        # Make sure NLTK resources are available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        if not text:
            return []
            
        # Use NLTK's pre-trained Punkt tokenizer
        segmentedText = sent_tokenize(text.strip())
        
        return segmentedText