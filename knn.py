import numpy as np
from tqdm import tqdm
from collections import Counter

class KNNClassifier:
    def __init__(self, k=1):
        if k > 0:
            self.k = k
        else:
            raise ValueError('Please provide a valid value for k')
    
    def fit(self, X, y):
        assert X.shape[0] == y.shape[0], "Equal number of samples and labels expected"
        
        self.X = X
        self.y = y
    
    def predict(self, X):
        labels = []
        
        for x in tqdm(X):
            dist = [np.linalg.norm(x-x_train) for x_train in self.X]
            
            k_samples = np.argsort(dist)[:self.k]
            
            k_labels = [self.y[i][0] for i in k_samples]
            
            label = Counter(k_labels).most_common(1)

            labels.append(label[0][0])
        
        return labels
    
    def accuracy(self, X, y):
        assert X.shape[0] == y.shape[0], "Equal number of samples and labels expected"
        
        labels = self.predict(X)
        
        total = 0
        correct = 0
        
        for i in range(len(labels)):
            total += 1
            
            if labels[i] == y[i]:
                correct += 1
        
        print('\n\nAccuracy {}%'.format(correct/total*100))
        
        return correct/total*100
