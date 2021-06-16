"""
Intro to ML HW 7
Audrey Zhang
Andrew ID: youyouz
"""
import numpy as np
import sys

#%%

class GaussianNaiveBayes:
    def __init__(self, num_features = None):
        self.num_features = num_features
        self.priors = []
        self.means = []
        self.sigmas = []
        self.labels = []
    
    def train(self, train, train_labels, test, test_labels, train_out, test_out, metrics_out, num_voxels = None):
    
        self.labels = list(set(train_labels))
                
        means = np.empty((0, len(train[0])))
        sigmas = np.empty((0, len(train[0])))
                
        for l in range(len(self.labels)):
            self.priors.append(np.char.count(train_labels, self.labels[l]).sum() / len(train_labels))

            idx = np.argwhere(train_labels == self.labels[l])
            subset = train[idx]
            mu = np.mean(subset, axis=0)
            sigma = np.std(subset, axis=0)
            
            means = np.append(means, mu, axis = 0)
            sigmas = np.append(sigmas, sigma, axis = 0)
        
        if num_voxels is not None:

            diff = np.diff(means, axis = 0)
            
            top_k = np.argsort(-abs(diff))[0][:num_voxels]
            
            new_means = means[:, top_k]
            new_sigmas = sigmas[:, top_k]
                       
            self.means = new_means
            self.sigmas = new_sigmas 
            
            data = train[:, top_k]
            y_hat, y_pred = self.predict(data)
            
            test_data = test[:, top_k]
            
            test_y_hat, test_y_pred = self.predict(test_data)

            
        else:
            self.means = means
            self.sigmas = sigmas
            
            y_hat, y_pred = self.predict(train)
                        
            test_y_hat, test_y_pred = self.predict(test)
        
        err_train = self.calc_error_rate(train_labels, y_pred)
        err_test = self.calc_error_rate(test_labels, test_y_pred)
        
        
        with open(train_out, 'w') as output:
            output.write('\n'.join(list(y_pred)))
        
        with open(test_out, 'w') as output:
            output.write('\n'.join(list(test_y_pred)))
            
        with open(metrics_out, 'w') as output:
            output.write("error(train): {:f}\n".format(err_train))
            output.write("error(test): {:f}\n".format(err_test))
        
        #return y_hat, y_pred
        return err_train, err_test
    
        
    def calc_error_rate(self, true, pred):
        
        correct = (true != pred).sum()
        return correct / len(true)
    
    def predict(self, data):
        
        labels_dict = dict(enumerate(self.labels))

        probabilities = np.empty((len(data), 0))
        
        for l in range(len(self.labels)):
            proba = self.calc_proba(data, l)
            proba = np.log(proba).sum(axis = 1) + np.log(self.priors[l])

            proba = proba.reshape((len(data), 1))
            probabilities = np.append(probabilities, proba, axis = 1)        

        
        probabilities = np.array(probabilities)
        
        y_hat = np.argmax(probabilities, axis=1)
        y_pred = np.array(list(map(lambda x: labels_dict.get(x, None), y_hat)))
        
        return y_hat, y_pred
    
    
    def calc_proba(self, data, label):
        
        means = self.means[label]
        sigmas = self.sigmas[label]
        
        cons = 1 / np.sqrt(2 * np.pi * (sigmas**2))
                
        exp = np.exp(-((data - means) ** 2) / (2 * (sigmas **2)))
        
        proba = cons * exp
        
        return proba
        
        
#%%

    

def main():
    
    # sys args:
    # train_input
    # test_input
    #train_out
    #test_out
    #metrics_out
    #num_voxels
    
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_voxels = int(sys.argv[6])
    
    
    train= [] 
    test = []

    with open(train_input, 'r') as file:
        for line in file:
            train.append(line.strip().split(','))
            
    with open(test_input, 'r') as file:
        for line in file:
            test.append(line.strip().split(','))
            
    cols = train[0]
    
    train = np.asarray(train)
    test = np.asarray(test)
    

    train_labels = train[1:, -1]
    test_labels = test[1:, -1]
    
    train = train[1:, :-1]
    test = test[1:, :-1]
    
    train = train.astype(float)
    test = test.astype(float)
    
    gnb = GaussianNaiveBayes(1) 
    
    err_train, err_test = gnb.train(train, train_labels, test, test_labels, train_out, test_out, metrics_out, num_voxels = num_voxels)
    
    
if __name__=='__main__': 
    
    main()
    
    
