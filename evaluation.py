# Import necessary libraries
from util import *
import numpy as np
import math
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Evaluation():
    """
    Class for evaluating information retrieval system performance
    using various metrics like precision, recall, F-score, NDCG, and MAP
    """

    def queryPrecision(self, retrieved_docs_ordered, query_identifier, relevant_docs, cutoff_k):
        """
        Calculate precision at k for a single query
        Precision measures the fraction of retrieved documents that are relevant

        Parameters
        ----------
        retrieved_docs_ordered : list
            Documents retrieved by the system in ranked order
        query_identifier : int
            Identifier for the current query
        relevant_docs : list
            Ground truth list of relevant documents for this query
        cutoff_k : int
            Number of top documents to consider

        Returns
        -------
        float
            Precision value between 0 and 1
        """
        # Count relevant documents in top-k results
        relevant_count = 0
        for doc_identifier in retrieved_docs_ordered[:cutoff_k]:
            if doc_identifier in relevant_docs:
                relevant_count += 1
                
        # Calculate precision as relevant/retrieved
        return relevant_count / cutoff_k if cutoff_k > 0 else 0


    def meanPrecision(self, all_retrieved_docs, all_query_ids, relevance_judgments, cutoff_k):
        """
        Calculate average precision across all queries at cutoff k

        Parameters
        ----------
        all_retrieved_docs : list
            Lists of document IDs for each query in ranked order
        all_query_ids : list
            List of query identifiers
        relevance_judgments : list
            Relevance judgments from ground truth
        cutoff_k : int
            Number of top documents to consider

        Returns
        -------
        float
            Mean precision value across all queries
        """
        # Create mapping from query ID to relevant documents
        query_to_relevant_docs = defaultdict(set)
        for judgment in relevance_judgments:
            # Consider only highly relevant documents (position <= 4)
            if int(judgment["position"]) <= 4:
                query_to_relevant_docs[judgment["query_num"].strip()].add(judgment["id"].strip())
        
        # Calculate precision for each query and average
        total_precision = 0
        for idx, query_id in enumerate(all_query_ids):
            total_precision += self.queryPrecision(
                all_retrieved_docs[idx], 
                query_id, 
                query_to_relevant_docs[str(query_id).strip()], 
                cutoff_k
            )
            
        # Return average precision across all queries
        return total_precision / len(all_query_ids)

    
    def queryRecall(self, retrieved_docs_ordered, query_identifier, relevant_docs, cutoff_k):
        """
        Calculate recall at k for a single query
        Recall measures the fraction of relevant documents that are retrieved

        Parameters
        ----------
        retrieved_docs_ordered : list
            Documents retrieved by the system in ranked order
        query_identifier : int
            Identifier for the current query
        relevant_docs : list
            Ground truth list of relevant documents for this query
        cutoff_k : int
            Number of top documents to consider

        Returns
        -------
        float
            Recall value between 0 and 1
        """
        # Get top-k retrieved documents
        top_k_docs = retrieved_docs_ordered[:cutoff_k]
        
        # Count relevant documents in the retrieved set
        relevant_retrieved = sum(1 for doc_id in top_k_docs if doc_id in relevant_docs)
        
        # Calculate recall as relevant_retrieved/total_relevant
        return relevant_retrieved / len(relevant_docs) if relevant_docs else 0


    def meanRecall(self, all_retrieved_docs, all_query_ids, relevance_judgments, cutoff_k):
        """
        Calculate average recall across all queries at cutoff k

        Parameters
        ----------
        all_retrieved_docs : list
            Lists of document IDs for each query in ranked order
        all_query_ids : list
            List of query identifiers
        relevance_judgments : list
            Relevance judgments from ground truth
        cutoff_k : int
            Number of top documents to consider

        Returns
        -------
        float
            Mean recall value across all queries
        """
        # Create mapping from query ID to relevant documents
        query_to_relevant_docs = defaultdict(set)
        for judgment in relevance_judgments:
            # Consider only highly relevant documents (position <= 4)
            if int(judgment["position"]) <= 4:
                query_to_relevant_docs[str(judgment["query_num"]).strip()].add(judgment["id"].strip())    
        
        # Calculate recall for each query and average
        total_recall = 0
        for idx, query_id in enumerate(all_query_ids):
            total_recall += self.queryRecall(
                all_retrieved_docs[idx], 
                query_id, 
                query_to_relevant_docs[str(query_id).strip()], 
                cutoff_k
            )
                    
        # Return average recall across all queries
        return total_recall / len(all_query_ids)


    def queryFscore(self, retrieved_docs_ordered, query_identifier, relevant_docs, cutoff_k):
        """
        Calculate F-score at k for a single query
        F-score is the harmonic mean of precision and recall

        Parameters
        ----------
        retrieved_docs_ordered : list
            Documents retrieved by the system in ranked order
        query_identifier : int
            Identifier for the current query
        relevant_docs : list
            Ground truth list of relevant documents for this query
        cutoff_k : int
            Number of top documents to consider

        Returns
        -------
        float
            F-score value between 0 and 1
        """
        # Calculate precision and recall
        precision_val = self.queryPrecision(retrieved_docs_ordered, query_identifier, relevant_docs, cutoff_k)
        recall_val = self.queryRecall(retrieved_docs_ordered, query_identifier, relevant_docs, cutoff_k)
        
        # Calculate F-score as harmonic mean of precision and recall
        denominator = precision_val + recall_val
        return (2 * precision_val * recall_val) / denominator if denominator > 0 else 0


    def meanFscore(self, all_retrieved_docs, all_query_ids, relevance_judgments, cutoff_k):
        """
        Calculate average F-score across all queries at cutoff k

        Parameters
        ----------
        all_retrieved_docs : list
            Lists of document IDs for each query in ranked order
        all_query_ids : list
            List of query identifiers
        relevance_judgments : list
            Relevance judgments from ground truth
        cutoff_k : int
            Number of top documents to consider
        
        Returns
        -------
        float
            Mean F-score value across all queries
        """
        # Create mapping from query ID to relevant documents
        query_to_relevant_docs = defaultdict(set)
        for judgment in relevance_judgments:
            # Consider only highly relevant documents (position <= 4)
            if int(judgment["position"]) <= 4:
                query_to_relevant_docs[judgment["query_num"].strip()].add(judgment["id"].strip())
        
        # Calculate F-score for each query and average
        total_fscore = 0
        for idx, query_id in enumerate(all_query_ids):
            total_fscore += self.queryFscore(
                all_retrieved_docs[idx], 
                query_id, 
                query_to_relevant_docs[str(query_id).strip()], 
                cutoff_k
            )
            
        # Return average F-score across all queries
        return total_fscore / len(all_query_ids)
    

    def queryNDCG(self, retrieved_docs_ordered, query_identifier, relevant_docs, cutoff_k):
        """
        Calculate Normalized Discounted Cumulative Gain (nDCG) at k for a single query
        nDCG measures the ranking quality of the retrieved documents

        Parameters
        ----------
        retrieved_docs_ordered : list
            Documents retrieved by the system in ranked order
        query_identifier : int
            Identifier for the current query
        relevant_docs : list
            Ground truth list of relevant documents for this query
        cutoff_k : int
            Number of top documents to consider

        Returns
        -------
        float
            nDCG value between 0 and 1
        """
        # Create relevance scores for retrieved documents (binary relevance)
        relevance_scores = [1 if doc_id in relevant_docs else 0 for doc_id in retrieved_docs_ordered[:cutoff_k]]
        
        # Calculate Discounted Cumulative Gain (DCG)
        dcg_value = sum([rel_score / math.log2(pos + 2) for pos, rel_score in enumerate(relevance_scores)])
        
        # Calculate Ideal DCG (IDCG) - best possible ranking
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg_value = sum([rel_score / math.log2(pos + 2) for pos, rel_score in enumerate(ideal_relevance)])
        
        # Calculate nDCG as DCG/IDCG
        return dcg_value / idcg_value if idcg_value > 0 else 0


    def meanNDCG(self, all_retrieved_docs, all_query_ids, relevance_judgments, cutoff_k):
        """
        Calculate average nDCG across all queries at cutoff k

        Parameters
        ----------
        all_retrieved_docs : list
            Lists of document IDs for each query in ranked order
        all_query_ids : list
            List of query identifiers
        relevance_judgments : list
            Relevance judgments from ground truth
        cutoff_k : int
            Number of top documents to consider

        Returns
        -------
        float
            Mean nDCG value across all queries
        """
        # Create mapping from query ID to relevant documents
        query_to_relevant_docs = defaultdict(set)
        for judgment in relevance_judgments:
            # Consider only highly relevant documents (position <= 4)
            if int(judgment["position"]) <= 4:
                query_to_relevant_docs[judgment["query_num"].strip()].add(judgment["id"].strip())
        
        # Calculate nDCG for each query and average
        total_ndcg = 0
        for idx, query_id in enumerate(all_query_ids):
            total_ndcg += self.queryNDCG(
                all_retrieved_docs[idx], 
                query_id, 
                query_to_relevant_docs[str(query_id).strip()], 
                cutoff_k
            )
            
        # Return average nDCG across all queries
        return total_ndcg / len(all_query_ids)


    def queryAveragePrecision(self, retrieved_docs_ordered, query_identifier, relevant_docs, cutoff_k):
        """
        Calculate Average Precision (AP) at k for a single query
        AP is the average of precision values at positions where relevant documents are found

        Parameters
        ----------
        retrieved_docs_ordered : list
            Documents retrieved by the system in ranked order
        query_identifier : int
            Identifier for the current query
        relevant_docs : list
            Ground truth list of relevant documents for this query
        cutoff_k : int
            Number of top documents to consider

        Returns
        -------
        float
            Average Precision value between 0 and 1
        """
        # Initialize counters
        relevant_found = 0
        sum_precision = 0.0
        
        # Calculate precision at each relevant document position
        for pos, doc_id in enumerate(retrieved_docs_ordered[:cutoff_k]):
            if doc_id in relevant_docs:
                relevant_found += 1
                # Add precision at this position
                sum_precision += relevant_found / (pos + 1)
        
        # Calculate average precision
        denominator = min(len(relevant_docs), cutoff_k)
        return sum_precision / denominator if denominator > 0 else 0


    def meanAveragePrecision(self, all_retrieved_docs, all_query_ids, relevance_data, cutoff_k):
        """
        Calculate Mean Average Precision (MAP) across all queries at cutoff k

        Parameters
        ----------
        all_retrieved_docs : list
            Lists of document IDs for each query in ranked order
        all_query_ids : list
            List of query identifiers
        relevance_data : list
            Relevance judgments from ground truth
        cutoff_k : int
            Number of top documents to consider

        Returns
        -------
        float
            MAP value across all queries
        """
        # Create mapping from query ID to relevant documents
        query_to_relevant_docs = defaultdict(set)
        for judgment in relevance_data:
            # Consider only highly relevant documents (position <= 4)
            if int(judgment["position"]) <= 4:
                query_to_relevant_docs[judgment["query_num"].strip()].add(judgment["id"].strip())
        
        # Calculate average precision for each query and average
        total_ap = 0
        for idx, query_id in enumerate(all_query_ids):
            total_ap += self.queryAveragePrecision(
                all_retrieved_docs[idx], 
                query_id, 
                query_to_relevant_docs[str(query_id).strip()], 
                cutoff_k
            )
            
        # Return mean average precision across all queries
        return total_ap / len(all_query_ids)
