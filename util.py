# Add your import statements here






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