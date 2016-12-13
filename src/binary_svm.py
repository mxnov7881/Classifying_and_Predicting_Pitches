from ml475types import Predictor

# Variation of PEGASOS
class BinarySVM(Predictor):
    def __init__(self):
        self._w_dict = {}
        #self._max_fv_length = 0
        
    def train(self, instances, online_training_iterations, pegasos_lambda, knn):
        Predictor.train(self, instances, online_training_iterations, pegasos_lambda, knn)
        
        # First training iteration at t=1
        t = 1
        dot_wx = 0.0
        
        # Hasn't started training yet (i.e. there have been 0 online training iterations so far)
        oti = 0
        
        for i in instances: 
            feature_dict = i._feature_vector._fv
            for k, v in feature_dict.iteritems():
                if k not in self._w_dict: self._w_dict[k] = 0.0
        
        # Begin training iterations
        while oti < online_training_iterations:
            # Look at ith instance (AKA training example) and store (\vec{x}_t^(i), y_t^(i)) = {(\vec{x}_1^(1), y_1^(1)),),..., (\vec{x}_1^(m), y_1^(m))}
            for i in instances:
                # y_t^(i) = {0,1} \equiv the ith instance's class label, as defined in "cs475_types.py"
                actual_label = i._label._class_label
                # If the ith instance's class label (#y_t^(i)) is "0," set it to "-1.0" so it can be used in math operations
                if actual_label == 0: actual_label = -1
                else: actual_label = 1
                
                # \vec{x}_t^(i) \equiv the ith instance's feature vector, as defined in "cs475_types.py"
                feature_dict = i._feature_vector._fv
                
                """First step in updating the weight vector (w): 
                    For every value in the weight vector, find its product with the value at the corresponding index in the feature vector.
                    Sum the resulting products to get \vec{w}_t \cdot \vec{x}_t^(i)
                """
                dot_wx = sum(self._w_dict[key] * feature_dict.get(key, 0.0) for key in self._w_dict)
                
                adjusted_w_scalar = (1.0 - 1.0 / t)
                
                partially_adjusted_w = {}
                # Compute adjusted weight vector: Multiply every element in the previous weight vector by (1-1/t) -> 1 as t -> \inf
                for k, v in self._w_dict.iteritems(): 
                    partially_adjusted_w[k] = adjusted_w_scalar * v
                
                regularization_vec = {}
                
                """Use the product of the ith instance's class label (either -1 or 1) and the dot product of the weight and feature vectors as
                    the condition that must be met in order to include the regularization term in w's update equation
                    (i.e. if this condition is not met, the value of the regularization term in w's update equation is 0)
                """
                condition_val = actual_label * dot_wx
                if condition_val < 1: #condition_term = 1
                    learning_rate = 1.0 / (pegasos_lambda * t)
                    regularization_scalar = learning_rate * actual_label
                        
                    for k, v in self._w_dict.iteritems():
                        if k not in feature_dict: feature_dict[k] = 0.0
                        regularization_vec[k] = regularization_scalar * feature_dict[k]
                        self._w_dict[k] = partially_adjusted_w[k] + regularization_vec[k]
                else: #condition_term = 0
                    # Compute w_{t+1}: No regularization so it just equals the same value as partially_adjusted_w
                    for k, v in self._w_dict.iteritems():
                        if k not in feature_dict: feature_dict[k] = 0.0
                        self._w_dict[k] = partially_adjusted_w[k]
                
                # w is now updated
                t += 1 # Loops correct number of time steps (default for "easy:" t = 4501)
            oti += 1
        
    def predict(self, instance):
        Predictor.predict(self, instance)
        feature_dict = instance._feature_vector._fv
        dot_wx = sum(self._w_dict[key] * feature_dict.get(key, 0.0) for key in self._w_dict)
        
        if dot_wx >= 0: return 1
        else: return 0