class Model:
    def __init__(self):
        self.prediction_fn = None
    def train(self):
        raise NotImplementedError("not implemented")
    def predict(self, test_data):
        self.prediction_fn(test_data)
        
