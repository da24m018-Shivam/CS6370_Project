from util import *

# Add your import statements here
import nltk
from nltk.corpus import stopwords
import math
from collections import Counter

class StopwordRemoval:

    def __init__(self):
        # Ensure stopwords are downloaded
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
            
        self.custom_stopwords = set()
        self.is_tfidf_computed = False

    def compute_tfidf_stopwords(self, corpus, threshold=0.1):
        """
        Compute stopwords based on TF-IDF scores.
        
        Parameters
        ----------
        corpus : list of lists
            A list where each sub-list contains tokens from a document
        threshold : float
            Words with TF-IDF below this threshold will be considered stopwords
            
        Returns
        -------
        set
            A set of custom stopwords identified by TF-IDF
        """
        # Flatten all documents to get term frequencies across the corpus
        all_terms = [token for doc in corpus for sentence in doc for token in sentence]
        term_freq = Counter(all_terms)
        
        # Calculate document frequencies
        doc_count = len(corpus)
        doc_freq = {}
        
        # Consider each document
        for doc in corpus:
            # Get unique terms in this document (across all sentences)
            unique_terms = set([token for sentence in doc for token in sentence])
            # Increment document frequency for each term
            for term in unique_terms:
                doc_freq[term] = doc_freq.get(term, 0) + 1
        
        # Calculate TF-IDF
        tfidf = {}
        for term, freq in term_freq.items():
            # Term frequency normalized by document length
            tf = freq / len(all_terms)
            # Inverse document frequency
            idf = math.log(doc_count / (1 + doc_freq.get(term, 0)))
            # TF-IDF score
            tfidf[term] = tf * idf
        
        # Identify words with high frequency but low information gain
        self.custom_stopwords = {term for term, score in tfidf.items() if score < threshold}
        self.is_tfidf_computed = True
        
        # Compare with NLTK stopwords
        nltk_only = self.stop_words - self.custom_stopwords
        custom_only = self.custom_stopwords - self.stop_words
        common = self.stop_words.intersection(self.custom_stopwords)
        
        print(f"NLTK stopwords only: {len(nltk_only)}")
        print(f"Custom TF-IDF stopwords only: {len(custom_only)}")
        print(f"Common stopwords: {len(common)}")
        
        return self.custom_stopwords

    def fromList(self, text):
        """
        Remove stopwords from tokenized documents.

        Parameters
        ----------
        text : list of lists
            A list where each sub-list contains tokens representing a sentence.

        Returns
        -------
        list of lists
            The same structure as input, but with stopwords removed.
        """
        if text is None or not isinstance(text, list):
            return []  # Safe return for invalid input
        
        # Combine NLTK stopwords with custom computed stopwords if available
        all_stopwords = self.stop_words.union(self.custom_stopwords)
        
        stopword_removed_text = []
        for sentence in text:
            if not sentence:
                stopword_removed_text.append([])
                continue
                
            # Filter out stopwords (case insensitive)
            filtered_sentence = [token for token in sentence if token.lower() not in all_stopwords]
            stopword_removed_text.append(filtered_sentence)
                
        return stopword_removed_text