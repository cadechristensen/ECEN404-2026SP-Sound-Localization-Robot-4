import joblib
import os

# Set the path to match where your training script saved the file
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
models_folder = os.path.join(_PROJECT_ROOT)
features_path = os.path.join(models_folder, 'feature_names_rawApril6_angle_try2.joblib')

def print_model_features():
    print(f"Looking for feature file at:\n{features_path}\n")
    
    if not os.path.exists(features_path):
        print("ERROR: Feature file not found. Please double-check the filename and path.")
        return

    try:
        # Load the feature list saved by RFECV
        dist_features = joblib.load(features_path)
        
        print("=" * 40)
        print(f" EXPECTED MODEL FEATURES: {len(dist_features)}")
        print("=" * 40)
        
        # Print them as a clean, copy-pasteable list
        feature_list_str = "[\n"
        for feature in dist_features:
            feature_list_str += f"    '{feature}',\n"
        feature_list_str += "]"
        
        print(feature_list_str)
        print("\nCopy and paste this array back into our chat!")
        
    except Exception as e:
        print(f"Failed to load features: {e}")

if __name__ == "__main__":
    print_model_features()