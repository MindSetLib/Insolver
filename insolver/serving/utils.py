import pickle


def load_pickle_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
    except pickle.UnpicklingError:
        return
    return model
