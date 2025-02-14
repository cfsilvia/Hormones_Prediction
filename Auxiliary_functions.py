import pickle

def save_part_of_dict(filename, key, dictionary):
    try:
        # Load existing data if file exists
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    except (FileNotFoundError, EOFError):
        data = {}  # Initialize empty dict if file doesn't exist or is empty
    
    
    # Update the stored dictionary with the new key-value pair
    data[key] = dictionary[key]

    # Save updated dictionary back to the file
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)  
        