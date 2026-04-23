import os
import sys

# 1. FORCE Python to look in THIS exact folder first before checking anywhere else
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import cls_feature_class
import doanet_parameters

# 2. Print out the exact file path Python is using so we can catch it in the act
print("="*60)
print(f"DEBUG: Loading cls_feature_class from:\n{cls_feature_class.__file__}")
print("="*60)

# 3. Load parameters (Using '6' based on your console output)
params = doanet_parameters.get_params('6')

# 4. Initialize the class
dev_feat_cls = cls_feature_class.FeatureClass(params)

# 5. Run the pipeline
dev_feat_cls.extract_all_feature()
dev_feat_cls.preprocess_features()
dev_feat_cls.extract_all_labels()