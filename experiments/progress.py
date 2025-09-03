import os
import pickle

def load_progress(cachefile):
    if os.path.exists(cachefile):
        with open(cachefile, 'rb') as f:
            progress = pickle.load(f)
    else:
        progress = {}
    return progress

def save_progress(cachefile, i, concept, llm_data):
    progress = load_progress(cachefile)
    progress[(i, concept)] = llm_data
    with open(cachefile, 'wb') as f:
        pickle.dump(progress, f)
