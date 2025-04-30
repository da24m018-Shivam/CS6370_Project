# Add your import statements here
import numpy as np
from collections import defaultdict, Counter
import math


class InformationRetrieval():

	def __init__(self):
		self.index = None
		self.doc_freqs = None
		self.doc_lengths = None
		self.idf = None
		self.doc_IDs = None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
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


	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
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
			
			# Count term frequencies in the query
			query_term_counts = Counter(flat_query)
			
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




