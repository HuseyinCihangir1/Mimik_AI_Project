import json
import os

def save_class_indices(train_gen, save_path):
    """Sinif isimlerini JSON olarak kaydeder."""
    class_indices = train_gen.class_indices
    index_to_class = {str(v): k for k, v in class_indices.items()}
    
    with open(os.path.join(save_path, "class_indices.json"), "w") as f:
        json.dump(index_to_class, f, indent=4)