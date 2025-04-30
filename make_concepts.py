import os
import re
import json
import time
import random
import argparse


def preprocess_document(content):
    """
    Process a document for use as a concept
    
    Parameters
    ----------
    content : str
        Raw document content
        
    Returns
    -------
    str
        Processed document content
    """
    # Remove special characters
    content = re.sub(r'[^\w\s]', ' ', content)
    # Convert to lowercase
    content = content.lower()
    # Remove extra whitespace
    content = re.sub(r'\s+', ' ', content).strip()
    return content

def create_concepts_from_cranfield_json(docs_file="cran_docs.json", output_path="cranfield_concepts.txt"):
    """
    Create concept files from Cranfield JSON documents
    
    Parameters
    ----------
    docs_file : str
        Path to Cranfield documents JSON file
    output_path : str
        Path to save the processed concept files
        
    Returns
    -------
    list
        List of processed concept documents
    """
    print(f"Creating concept files from Cranfield JSON documents: {docs_file}")
    
    concept_docs = []
    
    # Load documents from JSON file
    if not os.path.exists(docs_file):
        print(f"Document file not found: {docs_file}")
        return concept_docs
        
    try:
        with open(docs_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        print(f"Loaded {len(documents)} documents from JSON file")
        
        for i, doc in enumerate(documents, 1):
            try:
                # Extract title and body text
                title = doc.get("title", "").strip()
                body = doc.get("body", "").strip()
                
                # Combine title and body for concept
                full_text = f"{title} {body}".strip()
                if full_text:
                    processed_content = preprocess_document(full_text)
                    concept_docs.append(processed_content)
                    
                    if i % 100 == 0:
                        print(f"Processed {i} documents")
            except Exception as e:
                print(f"Error processing document {doc.get('id', i)}: {e}")
        
        # Save to file
        if output_path and concept_docs:
            with open(output_path, 'w', encoding='utf-8') as f:
                for doc in concept_docs:
                    f.write(doc + "\n---\n")  # Use separator between concepts
        
        print(f"Created {len(concept_docs)} concept documents from Cranfield JSON")
        
    except Exception as e:
        print(f"Error loading JSON documents: {e}")
    
    return concept_docs

def create_concepts_from_wikipedia(num_articles=500, output_path="wikipedia_concepts.txt"):
    """
    Create concept files from Wikipedia articles
    
    Parameters
    ----------
    num_articles : int
        Number of Wikipedia articles to use
    output_path : str
        Path to save the processed concept files
        
    Returns
    -------
    list
        List of processed concept documents
    """
    try:
        import wikipedia
    except ImportError:
        print("Please install wikipedia package: pip install wikipedia")
        return []

    # Define search terms relevant to information retrieval and aerospace (for Cranfield relevance)
    search_terms = [
        # Aerospace/engineering terms (relevant to Cranfield collection)
        "aerospace engineering", "aerodynamics", "fluid dynamics", "aircraft design",
        "supersonic flow", "boundary layer", "shock wave", "heat transfer", 
        "structural engineering", "materials science", "thermodynamics", "propulsion",
        "computational fluid dynamics", "wind tunnel", "aircraft structure", "turbulence",
        
        # General academic/scientific terms
        "scientific method", "research methodology", "academic publishing", "experiment design",
        "scientific theory", "laboratory research", "data analysis", "statistical methods"
    ]
    
    concept_docs = []
    articles_per_term = max(5, num_articles // len(search_terms))
    
    print(f"Fetching Wikipedia articles for {len(search_terms)} search terms...")
    
    for term in search_terms:
        if len(concept_docs) >= num_articles:
            break
            
        try:
            print(f"Searching for articles related to: {term}")
            # Search for articles related to this term
            search_results = wikipedia.search(term, results=articles_per_term)
            
            for title in search_results:
                if len(concept_docs) >= num_articles:
                    break
                    
                try:
                    # Get article content
                    page = wikipedia.page(title, auto_suggest=False)
                    
                    # Process content
                    processed_content = preprocess_document(page.content)
                    concept_docs.append(processed_content)
                    print(f"Added concept: {title}")
                    
                    # Add a short delay to avoid hitting rate limits
                    time.sleep(0.5 + random.random())
                    
                except wikipedia.exceptions.DisambiguationError as e:
                    # Try the first option in disambiguation
                    try:
                        if e.options:
                            page = wikipedia.page(e.options[0], auto_suggest=False)
                            processed_content = preprocess_document(page.content)
                            concept_docs.append(processed_content)
                            print(f"Added concept from disambiguation: {e.options[0]}")
                            time.sleep(0.5 + random.random())
                    except Exception as inner_e:
                        print(f"Error handling disambiguation for {title}: {inner_e}")
                        
                except Exception as e:
                    print(f"Error fetching {title}: {e}")
        except Exception as e:
            print(f"Error searching for {term}: {e}")
    
    # Save to file
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in concept_docs:
                f.write(doc + "\n---\n")  # Use separator between concepts
    
    print(f"Created {len(concept_docs)} concept documents from Wikipedia")
    return concept_docs

def combine_concept_sources(cranfield_path="cranfield/cran_docs.json", output_path="combined_concepts.txt"):
    """
    Combine concepts from multiple sources
    
    Parameters
    ----------
    cranfield_path : str
        Path to Cranfield JSON documents
    output_path : str
        Path to save the combined concept files
        
    Returns
    -------
    list
        List of processed concept documents
    """
    all_concepts = []
    
    # 1. Get concepts from Wikipedia
    wiki_concepts = create_concepts_from_wikipedia(
        num_articles=1000, 
        output_path="wikipedia_concepts.txt"
    )
    all_concepts.extend(wiki_concepts)
    
    # 2. Get concepts from Cranfield
    cranfield_concepts = create_concepts_from_cranfield_json(
        docs_file=cranfield_path, 
        output_path="cranfield_concepts.txt"
    )
    all_concepts.extend(cranfield_concepts)
    
    # Save combined concepts
    if output_path and all_concepts:
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in all_concepts:
                f.write(doc + "\n---\n")  # Use separator between concepts
    
    print(f"Combined {len(all_concepts)} concept documents")
    return all_concepts


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='make_concepts.py')
    parser.add_argument('-dataset', default = "cranfield/", 
						help = "Path to the dataset folder")
    parser.add_argument('-out_file_path', default = "wikipedia_concepts.txt", 
						help = "Path to output concept file")
    parser.add_argument('-concepts_from', default='wikipedia',
                        help="Source for concepts [wikipedia|cranfield|combined]")
    parser.add_argument('-num_articles', type=int, default=500,
                        help="number of wikipedia articles to use")
    
    args = parser.parse_args()

    # Check if JSON files exist
    data_dir = args.dataset
    docs_file = os.path.join(data_dir, "cran_docs.json")
    queries_file =  os.path.join(data_dir, "cran_queries.json")
    qrels_file =  os.path.join(data_dir, "cran_qrels.json")
    
    has_cranfield_json = os.path.exists(docs_file)
    
    if args.concepts_from == "combined":
        if not has_cranfield_json:
            print(f"Cranfield JSON not found at: {docs_file}")
            raise FileNotFoundError("Please provide a valid cranfield dataset folder")
        print(f"Found Cranfield JSON dataset")
        
        # Create concepts from both sources (Wikipedia and Cranfield JSON)
        all_concepts = combine_concept_sources(
            cranfield_path=docs_file,
            output_path=args.out_file_path
        )
        
        print(f"Generated {len(all_concepts)} total concepts")
    elif args.concepts_from == "wikipedia":
        print("Generating concepts from Wikipedia only")
        
        # Create concept files from Wikipedia only
        wiki_concepts = create_concepts_from_wikipedia(
            num_articles=args.num_articles,
            output_path=args.out_file_path
        )
        
        print(f"Generated {len(wiki_concepts)} Wikipedia concepts")
    else:
        if not has_cranfield_json:
            print(f"Cranfield JSON dataset not found")
            raise FileNotFoundError(f"Please provide a valid cranfiled dataset folder")
        print("Generating concepts from cranfield JSON only")
        
        # Create concept file from cranfield JSON only
        cranfiled_concepts = create_concepts_from_cranfield_json(
            docs_file=docs_file,
            output_path=args.out_file_path
        )
        print(f"Generated {len(cranfiled_concepts)} concepts from dataset")
    
    print("Concept generation complete!")