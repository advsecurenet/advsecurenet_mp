import os


def get_object_detector_model_names():
    # iterate over CustomODModels folder and return the names of the files that inherit from CustomODBaseModel
    print("Getting object detector model names...")
    resulting_names = []
    custom_od_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CustomODModels")
    skip_files = ["CustomODBaseModel.py", "__init__.py"]
    for file in os.listdir(custom_od_models_dir):
        if file.endswith(".py") and file not in skip_files:
            resulting_names.append(file.replace('.py', ''))
    return resulting_names
