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
    
class WordNetRetrieval:
    """
    WordNet-enhanced information retrieval system.
    Uses WordNet to expand queries with synonyms and related terms.
    """
    
    def __init__(self):
        """
        Initialize the WordNet-enhanced retrieval system.
        """
        self.index = None
        self.doc_freqs = None
        self.doc_lengths = None
        self.idf = None
        self.doc_IDs = None
        
        # Download WordNet if not already available
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            
    def buildIndex(self, docs, docIDs):
        """
        Build the document index with WordNet enhancements.
        
        Parameters
        ----------
        docs : list
            A list of lists of lists where each sub-list is a document and 
            each sub-sub-list is a sentence of the document
        docIDs : list
            A list of integers denoting IDs of the documents
        """
        # Store document IDs
        self.doc_IDs = docIDs
        
        # Initialize index as a dictionary: term -> {doc_id -> term frequency}
        index = defaultdict(lambda: defaultdict(int))
        
        # Initialize document frequencies: term -> number of documents containing the term
        self.doc_freqs = defaultdict(int)
        
        # Initialize document lengths for normalization
        self.doc_lengths = defaultdict(float)
        
        # Build the index
        for i, doc in enumerate(docs):
            doc_id = docIDs[i]
            
            # Flatten the document (list of lists of tokens) into a single list
            flat_doc = [token for sentence in doc for token in sentence]
            
            # Count term frequencies in the document
            term_counts = Counter(flat_doc)
            
            # Update the index and document frequencies
            for term, count in term_counts.items():
                index[term][doc_id] = count
                self.doc_freqs[term] += 1
                self.doc_lengths[doc_id] += count ** 2
        
        # Calculate IDF for each term
        num_docs = len(docIDs)
        self.idf = {term: math.log10(num_docs / freq) for term, freq in self.doc_freqs.items()}
        
        # Normalize document lengths (square root for Euclidean norm)
        for doc_id in self.doc_lengths:
            self.doc_lengths[doc_id] = math.sqrt(self.doc_lengths[doc_id])
        
        self.index = dict(index)
        
    def expand_query_with_wordnet(self, query_terms, max_synonyms=3):
        """
        Expand query terms with WordNet synonyms.
        
        Parameters
        ----------
        query_terms : list
            List of query terms to expand
        max_synonyms : int
            Maximum number of synonyms to add per term
            
        Returns
        -------
        list
            Expanded list of query terms
        """
        expanded_terms = list(query_terms)  # Start with original terms
        
        for term in query_terms:
            # Get synsets for the term
            synsets = wn.synsets(term)
            
            # Get lemma names (synonyms) from synsets
            synonyms = []
            for synset in synsets[:2]:  # Limit to first 2 synsets to avoid noise
                synonyms.extend(lemma.name() for lemma in synset.lemmas())
            
            # Remove duplicates and the original term
            synonyms = [s.replace('_', ' ') for s in synonyms if s != term]
            
            # Add top synonyms to expanded terms
            expanded_terms.extend(synonyms[:max_synonyms])
            
        return expanded_terms
        
    def rank(self, queries):
        """
        Rank documents according to relevance for each query using WordNet expansion.
        
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
        
        # Process each query
        for query in queries:
            # Flatten the query (list of lists of tokens) into a single list
            flat_query = [token for sentence in query for token in sentence]
            
            # Expand query with WordNet synonyms
            expanded_query = self.expand_query_with_wordnet(flat_query)
            
            # Count term frequencies in the expanded query
            query_term_counts = Counter(expanded_query)
            
            # Calculate query vector length for normalization
            query_length = math.sqrt(sum(count ** 2 for count in query_term_counts.values()))
            
            # Calculate TF-IDF scores for each document
            scores = defaultdict(float)
            
            for term, query_tf in query_term_counts.items():
                if term in self.index:
                    # Get the IDF value for this term
                    idf = self.idf.get(term, 0)
                    
                    # Calculate the query TF-IDF weight
                    query_weight = query_tf * idf
                    
                    # For each document containing this term
                    for doc_id, doc_tf in self.index[term].items():
                        # Calculate the document TF-IDF weight
                        doc_weight = doc_tf * idf
                        
                        # Add to the cosine similarity score
                        scores[doc_id] += query_weight * doc_weight
            
            # Normalize scores by document length
            for doc_id in scores:
                if self.doc_lengths[doc_id] > 0 and query_length > 0:
                    scores[doc_id] /= (self.doc_lengths[doc_id] * query_length)
            
            # Sort documents by score in descending order
            ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            # Extract document IDs in ranked order
            ranked_doc_ids = [doc_id for doc_id, score in ranked_docs]
            
            # Add any remaining documents (with score 0) in arbitrary order
            remaining_docs = [doc_id for doc_id in self.doc_IDs if doc_id not in scores]
            ranked_doc_ids.extend(remaining_docs)
            
            doc_IDs_ordered.append(ranked_doc_ids)
    
        return doc_IDs_ordered