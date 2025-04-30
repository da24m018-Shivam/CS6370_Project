# Add your import statements here
from main import SearchEngine
from advance_retrieval import *
import argparse
import json
import matplotlib.pyplot as plt
import time
import os  # Add this import



# Add any utility functions here

# Utility functions and classes for the search engine

# Utility functions for NLP tasks

def print_progress(message):
    """
    Print progress message with formatting.
    
    Parameters
    ----------
    message : str
        The message to display
    """
    print(f"[INFO] {message}")

def flatten_list(nested_list):
    """
    Flatten a list of lists into a single list.
    
    Parameters
    ----------
    nested_list : list
        A list of lists
        
    Returns
    -------
    list
        A flattened list
    """
    return [item for sublist in nested_list for item in sublist]

def compare_outputs(output1, output2, name1="Approach 1", name2="Approach 2"):
    """
    Compare outputs from two different approaches and print differences.
    
    Parameters
    ----------
    output1 : list
        Output from first approach
    output2 : list
        Output from second approach
    name1 : str
        Name of first approach
    name2 : str
        Name of second approach
    """
    if isinstance(output1[0], list) and isinstance(output2[0], list):
        # For list of lists (like tokenized text)
        total_items1 = sum(len(sublist) for sublist in output1)
        total_items2 = sum(len(sublist) for sublist in output2)
        print(f"{name1} produced {total_items1} tokens")
        print(f"{name2} produced {total_items2} tokens")
    else:
        # For simple lists (like segmented sentences)
        print(f"{name1} produced {len(output1)} items")
        print(f"{name2} produced {len(output2)} items")
        
    # Show examples of differences if possible
    if len(output1) > 0 and len(output2) > 0:
        print(f"\nExample from {name1}: {output1[0]}")
        print(f"Example from {name2}: {output2[0]}")


class CustomSearchEngine(SearchEngine):
    def __init__(self, args):
        # Call the parent class's __init__ method
        super().__init__(args)
        self.args = args
        # Override the information retriever with your custom class
        if self.args.retrieval == "esa":
            self.informationRetriever = ESARetrieval(concept_docs=self.args.concepts, n_concepts=self.args.n_concepts)
        elif self.args.retrieval == "lsa":
            self.informationRetriever = LSARetrieval(n_components=self.args.n_components)
        elif self.args.retrieval == "wordnet":
            self.informationRetriever = WordNetRetrieval()
        elif self.args.retrieval == "word2vec":
            self.informationRetriever = Word2VecRetrieval(vector_size=self.args.vec_size, window=self.args.window, min_count=self.args.min_count)
        elif self.args.retrieval == "bm25":
            self.informationRetriever = BM25Retrieval(k1=self.args.k1, b=self.args.b)
        else:
            print("Invalid retrieval method specified. Choose from esa, lsa, wordnet, word2vec, or bm25.")
            print("Defaulting to bm25.")
            self.informationRetriever = BM25Retrieval(k1=self.args.k1, b=self.args.b)
    
    def evaluateDataset(self):
        """
		- preprocesses the queries and documents, stores in output folder
		- invokes the IR system
		- evaluates precision, recall, fscore, nDCG and MAP 
		  for all queries in the Cranfield dataset
		- produces graphs of the evaluation metrics in the output folder
		"""

		# Read queries
        queries_json = json.load(open(self.args.dataset + "cran_queries.json", 'r'))[:]
        query_ids, queries = [str(item["query number"]).strip() for item in queries_json], \
								[item["query"] for item in queries_json]
		# Process queries
        processedQueries = self.preprocessQueries(queries)

		# Read documents
        docs_json = json.load(open(self.args.dataset + "cran_docs.json", 'r'))[:]
        doc_ids, docs = [str(item["id"]).strip() for item in docs_json], \
								[item["body"] for item in docs_json]

		# Process documents
        processedDocs = self.preprocessDocs(docs)

		# Build document index
        self.informationRetriever.buildIndex(processedDocs, doc_ids)
		# Rank the documents for each query
        doc_IDs_ordered = self.informationRetriever.rank(processedQueries)

		# Read relevance judements
        qrels = json.load(open(self.args.dataset + "cran_qrels.json", 'r'))[:]

		# Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
        precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
        for k in range(1, 11):
            precision = self.evaluator.meanPrecision(
                doc_IDs_ordered, query_ids, qrels, k)
            precisions.append(precision)
            recall = self.evaluator.meanRecall(
                doc_IDs_ordered, query_ids, qrels, k)
            recalls.append(recall)
            fscore = self.evaluator.meanFscore(
                doc_IDs_ordered, query_ids, qrels, k)
            fscores.append(fscore)
            print("Precision, Recall and F-score @ " +  
                str(k) + " : " + str(precision) + ", " + str(recall) + 
                ", " + str(fscore))
            MAP = self.evaluator.meanAveragePrecision(
                doc_IDs_ordered, query_ids, qrels, k)
            MAPs.append(MAP)
            nDCG = self.evaluator.meanNDCG(
                doc_IDs_ordered, query_ids, qrels, k)
            nDCGs.append(nDCG)
            print("MAP, nDCG @ " +  
                str(k) + " : " + str(MAP) + ", " + str(nDCG))

		# Plot the metrics and save plot 
        plt.plot(range(1, 11), precisions, label="Precision")
        plt.plot(range(1, 11), recalls, label="Recall")
        plt.plot(range(1, 11), fscores, label="F-Score")
        plt.plot(range(1, 11), MAPs, label="MAP")
        plt.plot(range(1, 11), nDCGs, label="nDCG")
        plt.legend()
        plt.title("Evaluation Metrics - Cranfield Dataset")
        plt.xlabel("k")
        plt.savefig(self.args.out_folder + "eval_plot.png")

		
    def handleCustomQuery(self):
        """
        Take a custom query as input and return top five relevant documents
        """

        #Get query
        print("Enter query below")
        query = input()
        # Process documents
        processedQuery = self.preprocessQueries([query])[0]

        # Read documents
        docs_json = json.load(open(self.args.dataset + "cran_docs.json", 'r'))[:]
        doc_ids, docs = [item["id"] for item in docs_json], \
                            [item["body"] for item in docs_json]
        # Process documents
        processedDocs = self.preprocessDocs(docs)

        # Build document index
        self.informationRetriever.buildIndex(processedDocs, doc_ids)
        # Rank the documents for the query
        doc_IDs_ordered = self.informationRetriever.rank([processedQuery])[0]

        # Print the IDs of first five documents
        print("\nTop five document IDs : ")
        for id_ in doc_IDs_ordered[:5]:
            print(id_)

# Usage example
if __name__ == "__main__":    
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Custom Search Engine from util.py')
    
    # Add the same arguments as in the original main.py
    parser.add_argument('-dataset', default="cranfield/", 
                        help="Path to the dataset folder")
    parser.add_argument('-out_folder', default="results/", 
                        help="Path to output folder")
    parser.add_argument('-segmenter', default="punkt",
                        help="Sentence Segmenter Type [naive|punkt]")
    parser.add_argument('-tokenizer', default="ptb",
                        help="Tokenizer Type [naive|ptb]")
    parser.add_argument('-custom', action="store_true", 
                        help="Take custom query as input")
    
    parser.add_argument('-retrieval', default="bm25",
                        help="Retrieval method [lsa|esa|wordnet|word2vec|bm25]")
    parser.add_argument('-concepts', default="wikipedia_concepts.txt", 
                        help="Path to concept document as input")
    parser.add_argument('-n_components', type=int, default=100,
                        help="Number of components for lsa")
    parser.add_argument('-n_concepts', type=int, default=1000,
                        help="Number of concepts for esa")
    parser.add_argument('-vec_size', type=int, default=100,
                        help="Vector size for word2vec")
    parser.add_argument('-window', type=int, default=5,
                        help="window size for word2vec")
    parser.add_argument('-min_count', type=int, default=1,
                        help="min count for word2vec")
    parser.add_argument('-k1', type=float, default=1.5,
                        help='k1 parameter for bm25')
    parser.add_argument('-b', type=float, default=0.75,
                        help='b parameter for bm25')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Create an instance of the Custom Search Engine
    custom_search_engine = CustomSearchEngine(args)
    
    # Use it just like the original SearchEngine
    if args.custom:
        custom_search_engine.handleCustomQuery()
    else:
        custom_search_engine.evaluateDataset()