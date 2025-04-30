# Add your import statements here
import re
import nltk
from nltk.tokenize import TreebankWordTokenizer

class Tokenization():

    def naive(self, text):
        """
        Tokenization using a Naive Approach

        Parameters
        ----------
        arg1 : list
            A list of strings where each string is a single sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
        """
        tokenizedText = []
        
        for sentence in text:
            if not sentence:
                tokenizedText.append([])
                continue
                
            # Split on whitespace and punctuation
            tokens = re.split(r'[\s,.!?:;"\'\(\)\[\]\{\}\\/|@#$%^&*<>_\+=`~-]+', sentence)
            # Remove empty tokens
            tokens = [token for token in tokens if token]
            tokenizedText.append(tokens)

        return tokenizedText

    def pennTreeBank(self, text):
        """
        Tokenization using the Penn Tree Bank Tokenizer

        Parameters
        ----------
        arg1 : list
            A list of strings where each string is a single sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
        """
        # Make sure NLTK resources are available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        tokenizer = TreebankWordTokenizer()
        tokenizedText = []
        
        for sentence in text:
            if not sentence:
                tokenizedText.append([])
                continue
                
            tokens = tokenizer.tokenize(sentence)
            tokenizedText.append(tokens)
            
        return tokenizedText