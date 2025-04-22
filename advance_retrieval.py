from util import *
import numpy as np
from collections import defaultdict, Counter
import math
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import wordnet as wn
import scipy.sparse as sp
import networkx as nx
from collections import defaultdict, Counter
import numpy as np
import math

class LSARetrieval:
    """
    Latent Semantic Analysis (LSA) implementation for information retrieval.
    Uses SVD to reduce dimensionality and capture latent semantic structure.
    """
    
    def __init__(self, n_components=100):
        """
        Initialize the LSA model with the specified number of components.
        
        Parameters
        ----------
        n_components : int
            Number of dimensions in the latent semantic space
        """
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.vectorizer = TfidfVectorizer(use_idf=True)
        self.doc_vectors = None
        self.doc_IDs = None
        
    def buildIndex(self, docs, docIDs):
        """
        Build the LSA index for the document collection.
        
        Parameters
        ----------
        docs : list
            A list of lists of lists where each sub-list is a document and 
            each sub-sub-list is a sentence of the document
        docIDs : list
            A list of integers denoting IDs of the documents
        """
        self.doc_IDs = docIDs
        
        # Flatten documents for vectorization
        flattened_docs = [' '.join([' '.join(sentence) for sentence in doc]) for doc in docs]
        
        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(flattened_docs)
        
        # Apply SVD to reduce dimensions
        self.doc_vectors = self.svd.fit_transform(tfidf_matrix)
        
    def rank(self, queries):
        """
        Rank documents according to relevance for each query using LSA.
        
        Parameters
        ----------
        queries : list
            A list of lists of lists where each sub-list is a query and
            each sub-sub-list is a sentence of the query
            
        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        """
        doc_IDs_ordered = []
        
        # Flatten queries for vectorization
        flattened_queries = [' '.join([' '.join(sentence) for sentence in query]) for query in queries]
        
        # Transform queries to TF-IDF space
        query_tfidf = self.vectorizer.transform(flattened_queries)
        
        # Project queries to LSA space
        query_vectors = self.svd.transform(query_tfidf)
        
        # For each query, compute similarity with all documents
        for query_vec in query_vectors:
            # Compute cosine similarity
            similarities = np.dot(self.doc_vectors, query_vec) / (
                np.linalg.norm(self.doc_vectors, axis=1) * np.linalg.norm(query_vec)
            )
            
            # Create (doc_id, similarity) pairs and sort by similarity
            doc_scores = [(doc_id, sim) for doc_id, sim in zip(self.doc_IDs, similarities)]
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Extract sorted document IDs
            ranked_doc_ids = [doc_id for doc_id, _ in doc_scores]
            doc_IDs_ordered.append(ranked_doc_ids)
            
        return doc_IDs_ordered


class ESARetrieval:
    """
    Explicit Semantic Analysis (ESA) implementation for information retrieval.
    Uses external knowledge base (like Wikipedia) to create concept-based representations.
    """
    
    def __init__(self, concept_docs=None, n_concepts=1000):
        """
        Initialize the ESA model.
        
        Parameters
        ----------
        concept_docs : list, optional
            List of documents representing concepts (e.g., Wikipedia articles)
        n_concepts : int
            Number of top concepts to use for representation
        """
        self.n_concepts = n_concepts
        self.concept_vectorizer = TfidfVectorizer(use_idf=True)
        self.doc_concept_vectors = None
        self.doc_IDs = None
        
        # If concept documents are provided, build the concept space
        if concept_docs:
            self.concept_matrix = self.concept_vectorizer.fit_transform(concept_docs)
        else:
            self.concept_matrix = None
            
    def set_concept_docs(self, concept_docs):
        """
        Set or update the concept documents.
        
        Parameters
        ----------
        concept_docs : list
            List of documents representing concepts
        """
        self.concept_matrix = self.concept_vectorizer.fit_transform(concept_docs)
        
    def buildIndex(self, docs, docIDs):
        """
        Build the ESA index for the document collection.
        
        Parameters
        ----------
        docs : list
            A list of lists of lists where each sub-list is a document and 
            each sub-sub-list is a sentence of the document
        docIDs : list
            A list of integers denoting IDs of the documents
        """
        self.doc_IDs = docIDs
        
        # Ensure concept matrix is built
        if self.concept_matrix is None:
            raise ValueError("Concept matrix not initialized. Call set_concept_docs first.")
        
        # Flatten documents for vectorization
        flattened_docs = [' '.join([' '.join(sentence) for sentence in doc]) for doc in docs]
        
        # Create document-term matrix
        doc_term_matrix = self.concept_vectorizer.transform(flattened_docs)
        
        # Project documents into concept space
        self.doc_concept_vectors = doc_term_matrix.dot(self.concept_matrix.T)
        
        # Keep only top n_concepts for each document
        if self.n_concepts < self.doc_concept_vectors.shape[1]:
            # For each document, keep only the top n_concepts
            for i in range(self.doc_concept_vectors.shape[0]):
                row = self.doc_concept_vectors[i].toarray().flatten()
                threshold = np.sort(row)[-self.n_concepts]
                row[row < threshold] = 0
                self.doc_concept_vectors[i] = sp.csr_matrix(row)
        
    def rank(self, queries):
        """
        Rank documents according to relevance for each query using ESA.
        
        Parameters
        ----------
        queries : list
            A list of lists of lists where each sub-list is a query and
            each sub-sub-list is a sentence of the query
            
        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        """
        doc_IDs_ordered = []
        
        # Flatten queries for vectorization
        flattened_queries = [' '.join([' '.join(sentence) for sentence in query]) for query in queries]
        
        # Transform queries to term space
        query_term_matrix = self.concept_vectorizer.transform(flattened_queries)
        
        # Project queries to concept space
        query_concept_vectors = query_term_matrix.dot(self.concept_matrix.T)
        
        # For each query, compute similarity with all documents
        for i, query_vec in enumerate(query_concept_vectors):
            similarities = []
            
            # Compute cosine similarity with each document
            for j, doc_vec in enumerate(self.doc_concept_vectors):
                # Normalize vectors
                query_norm = np.sqrt(query_vec.multiply(query_vec).sum())
                doc_norm = np.sqrt(doc_vec.multiply(doc_vec).sum())
                
                if query_norm > 0 and doc_norm > 0:
                    # Compute dot product
                    dot_product = query_vec.multiply(doc_vec).sum()
                    sim = dot_product / (query_norm * doc_norm)
                else:
                    sim = 0.0
                    
                similarities.append((self.doc_IDs[j], sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Extract sorted document IDs
            ranked_doc_ids = [doc_id for doc_id, _ in similarities]
            doc_IDs_ordered.append(ranked_doc_ids)
            
        return doc_IDs_ordered