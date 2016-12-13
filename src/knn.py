from ml475types import Predictor
import math
from operator import itemgetter

# Standard kNN
class KNN(Predictor):
    def __init__(self):
        # Initialize global variable to keep track of number of nearest neighbors
        self._knn = 5
        
        # Initialize the training set that will be passed to the predictor
        self._training_set = []
        
        # We need to know the size of the largest fv so we can pad smaller fvs with 0s
        self._max_key = 1
        
    # Return training data in organized way
    def train(self, instances, online_training_iterations, pegasos_lambda, knn):
        Predictor.train(self, instances, online_training_iterations, pegasos_lambda, knn)
        # Number of nearest neighbors set by user at runtime
        self._knn = knn
        
        # We need to know the size of the largest fv so we can pad smaller fvs with 0s
        previous_max_key = 1
        self._max_key = previous_max_key
        
        # Loop through all instances to determine the size of the largest fv
        for i in instances:
            feature_dict = i._feature_vector._fv
            current_max_key = max(feature_dict)
            if current_max_key > previous_max_key:
                self._max_key = current_max_key
                previous_max_key = current_max_key
        
        label_key = self._max_key + 1
        # Loop through all instances, appending each instance's label to the end of its fv
        for i in instances:
            actual_label = i._label._class_label
            feature_dict = i._feature_vector
            fv = feature_dict._fv
            
            feature_dict.add(label_key, actual_label)
            self._training_set.append(fv)
        
        # Rank instances by increasing label value to break ties during nearest neighbor selection
        self._training_set = sorted(self._training_set, key = itemgetter(label_key))
            
        return self._training_set
        
    def predict(self, instance):
        Predictor.predict(self, instance)
        # The fv of the instance we are trying to classify
        test_fv = instance._feature_vector._fv
        
        # All fvs will have same length after zero-padding
        fv_length = self._max_key
        
        # Predict label of a given instance
        label = self.getLabel(test_fv, fv_length)
        
        return label
    
    # Predict label of a given test instance of given length
    def getLabel(self, test_fv, fv_length):
        # Get the k fvs most similar to the fv of a given test instance
        kNNs = self.getNeighbors(test_fv, fv_length, self._knn)
        
        # Predict label of the current test instance
        predicted_label = self.vote(kNNs)

        return predicted_label
    
    # Get the k fvs most similar to the fv of given test instance of given length
    def getNeighbors(self, test_fv, fv_length, k):
        # Get Euclidean distance of given instance's fv to that of each instance in training set
        e_dists = []
        for i in self._training_set:
            e_dist = self.getEuclideanDistance(test_fv, i, fv_length)
            e_dists.append((i, e_dist))
        
        # Rank points by increasing distance
        e_dists.sort(key = itemgetter(1))
        
        # Get k nearest neighbors (training instances with fvs most similar to test instance fv)
        neighbors = []
        for j in range(k):
            neighbors.append(e_dists[j][0])
        
        return neighbors

    # Get Euclidean distance between two given instances of a given length
    def getEuclideanDistance(self, instance1, instance2, length):
        distance = 0.0
        for x in range(1, length + 1):
            if x not in instance1: instance1[x] = 0.0
            if x not in instance2: instance2[x] = 0.0
            distance += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)
    
    def vote(self, kNNs):
        # Keep track of how many votes are given to each class
        votes = {}
        
        # Loop through fvs of kNNs
        for x in kNNs:
            # Each instance's label can be found at the end of its fv
            label_key = self._max_key + 1
            label = x.get(label_key)
            
            # If this label already has a vote, increment the count. Else, begin the count.
            if label in votes: votes[label] += 1
            else: votes[label] = 1
        
        # Rank labels by decreasing number of votes
        votes = sorted(votes.iteritems(), key = itemgetter(1), reverse = True) #[{label, # votes}, {label, # votes},...]

        # The predicted label is the label with highest ranking
        predicted_label = votes[0][0]
        
        return predicted_label