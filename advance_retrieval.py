import numpy as np
from gensim.models import Word2Vec

class Word2VecRetrieval:
    """
    Word2Vec-based retrieval implementation for information retrieval.
    """
    
    def __init__(self, vector_size=100, window=5, min_count=1):
        """
        Initialize the Word2Vec retrieval system.
        
        Parameters
        ----------
        vector_size : int
            Dimensionality of the word vectors
        window : int
            Maximum distance between the current and predicted word within a sentence
        min_count : int
            Ignores all words with total frequency lower than this
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None
        self.doc_vectors = None
        self.doc_IDs = None
        
    def build(self, docs, docIDs):
        """
        Build the Word2Vec model and document vectors.
        
        Parameters
        ----------
        docs : list
            A list of lists of lists where each sub-list is a document and
            each sub-sub-list is a sentence of the document
        docIDs : list
            A list of integers denoting IDs of the documents
        """
        self.doc_IDs = docIDs
        
        # Prepare sentences for training Word2Vec
        sentences = []
        for doc in docs:
            for sentence in doc:
                if sentence:  # Skip empty sentences
                    sentences.append(sentence)
        
        # Train Word2Vec model
        self.model = Word2Vec(sentences, vector_size=self.vector_size, 
                             window=self.window, min_count=self.min_count, workers=4)
        
        # Create document vectors by averaging word vectors
        self.doc_vectors = []
        for doc in docs:
            # Flatten the document
            flat_doc = [token for sentence in doc for token in sentence]
            
            # Calculate document vector as average of word vectors
            doc_vector = np.zeros(self.vector_size)
            word_count = 0
            
            for word in flat_doc:
                if word in self.model.wv:
                    doc_vector += self.model.wv[word]
                    word_count += 1
            
            # Normalize by word count
            if word_count > 0:
                doc_vector /= word_count
                
            self.doc_vectors.append(doc_vector)
            
        # Convert to numpy array for efficient computation
        self.doc_vectors = np.array(self.doc_vectors)