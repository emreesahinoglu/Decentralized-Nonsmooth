import numpy as np
import random
import itertools
import sklearn as sk
import sklearn.datasets
import scipy as sp

from sklearn.model_selection import train_test_split

class DGFMAgent:
    def __init__(self,num_param,mb_size=16,id=0,lr=1e-3):
        self.lr = lr
        self.num_param=num_param

        self.weight = np.random.randn(num_param)/np.sqrt(num_param)

        self.y_grad=np.zeros(num_param)
        self.prev_grad=np.zeros(num_param)

        self.y_grad_mix = np.zeros(num_param)
        self.weight_mix = np.zeros(num_param)
    
    def get_y_grad(self):
        return self.y_grad
    
    def get_weight(self):
        return self.weight

    def update_y_grad(self,new_grad):
        self.y_grad=self.y_grad+new_grad-self.prev_grad
        self.prev_grad=new_grad
    
    def set_y_grad(self,y_grad):
        self.y_grad=y_grad
    
    def update_weight(self):
        self.weight = self.weight-self.lr*self.y_grad
    
    def set_weight(self,w):
        self.weight=w


class SVMAgent:

    def __init__(self,num_param,id=0,lr=1e-3,D=1):
        # self.model_params = list(model_params)
        self.id =id
        self.lr =lr
        self.D = D

        self.num_param = num_param

        # self.weight = np.random.randn(num_param)/np.sqrt(num_param)
        self.weight = np.zeros(num_param)
        self.action = np.zeros(num_param)
        self.action_mix = np.zeros(num_param)
        self.weight_mix = np.zeros(num_param)
            
    def initialize_action(self):

        self.action = np.zeros(self.num_param)
        self.action_mix = np.zeros(self.num_param)

    def update_Dlr(self,coef):
        self.D=self.D*coef
        self.lr=self.lr*coef

    def get_action(self):
        return self.action

    def get_weight(self):
        return self.weight
    
    def get_grad_point(self):
        s = np.random.rand(1)
        grad_point = self.weight+s*self.action
        new_weight = self.weight + self.action
        
        return grad_point, new_weight

    def _project_action(self):
 
        action_norm = np.linalg.norm(self.action)
        if action_norm>self.D:
            # print(f'Action Projected for agent {self.id}')
            self.action = self.action*self.D/action_norm
    
    def set_action_update(self,x):
        self.action_mix = x.copy() # Check this part, Immutable

    def action_grad_update(self,grad):
        
        self.action = self.action-self.lr*grad
        self._project_action()
    
    def update_action(self):
        self.action=self.action_mix.copy() # Check this part, Immutable
    
    def set_weight(self,x):
        self.weight = x.copy()
    
    def set_action(self,x):
        self.action = x.copy()

    def set_weight_update(self,x):
        self.weight_mix = x.copy()
    
    def update_weight(self):
        self.weight = self.weight_mix.copy()

class SGDAgent:
    def __init__(self,num_param,id=0,lr=1e-3):
        self.num_param = num_param

        self.weight = np.random.randn(num_param)/np.sqrt(num_param)
        # self.weight=np.zeros(num_param)
        self.id=id
        self.lr = lr
    
    def get_weight(self):
        return self.weight
    
    def set_weight(self,x):
        self.weight = x.copy()
    
    def update_weight(self,grad):
        self.weight-=self.lr*grad
    

class SVMOracle:
    def __init__(self,alpha=2,lam=1e-5):
        self.alpha = alpha
        self.lam = lam
    
    def get_gradients(self,w,x,y):

        dz1 = -(x*y[:, np.newaxis])*(((x@w)*y)<1).astype(float)[:, np.newaxis]
        dz2 = self.lam*np.sign(w)*(np.abs(w)<self.alpha)
        dz1=dz1.mean(axis=0)

        return dz1+dz2  # consider taking mean

    def reg_term(self,w):
        return self.lam*np.clip(np.abs(w),0,self.alpha).sum()

    def get_fn_val(self,w,x,y):

        z1= np.maximum(1-(x@w)*y ,0).mean()
        z2 = self.lam*np.clip(np.abs(w),0,self.alpha).sum()
        return z1+z2
    
    def get_zo_grad(self,w,x,y,delta=1e-3):
        w1 = np.random.randn(*w.shape)
        w1 = w1/np.linalg.norm(w1)
        w2 = w+delta*w1
        w3 = w-delta*w1
        fn_diff = self.get_fn_val(w2,x,y)-self.get_fn_val(w3,x,y)
        grad_est = (w1.shape[0]*fn_diff/(2*delta))*w1
        return grad_est
    
    def get_zo_grad_given_dir(self,w,w1,x,y,delta=1e-3):

        w2 = w+delta*w1
        w3 = w-delta*w1
        fn_diff = self.get_fn_val(w2,x,y)-self.get_fn_val(w3,x,y)
        grad_est = (w1.shape[0]*fn_diff/(2*delta))*w1
        return grad_est

    def get_zo_grad_dgfmp(self,w,w1,x,y,delta=1e-3):
        w2 = w+delta*w1
        w3 = w-delta*w1
        z0 = np.maximum(1-(x*w2).sum(axis=1)*y,0) +self.reg_term(w2)
        z1 = np.maximum(1-(x*w3).sum(axis=1)*y,0) +self.reg_term(w3)
        fn_diff = (z0-z1)[:,np.newaxis]
        grad_est = (x.shape[1]*fn_diff/(2*delta))*w1
        return grad_est.mean(axis=0)

    def get_accuracy(self,w,x,y):
        
        pred =x@w
        accuracy=((pred*y)>0).mean()
        return accuracy

class ComNetwork:
    def __init__(self,W=None):
        self.W = W
        if W is not None:
            self.N = W.shape[0]
    
    def set_weight(self,W):
        self.W = W
        self.N = W.shape[0]

    def propagate_weights(self,agents):
        
        if self.W is None:
            self.set_weight(np.eye(len(agents)))
        
        weight_data = [ agent.get_weight() for agent in agents] 
        mixed_weight = self.propagate_data(weight_data)

        for k in range(self.N):
            agents[k].set_weight(mixed_weight[k]) # Control Immutable
    
    def propagate_actions(self,agents):
        if self.W is None:
            self.set_weight(np.eye(len(agents)))
        
        action_data = [ agent.get_action() for agent in agents] 
        mixed_action = self.propagate_data(action_data)

        for k in range(self.N):
            agents[k].set_action(mixed_action[k]) # Control Immutable

    def propagate_dgfm_grad(self,agents):
        if self.W is None:
            self.set_weight(np.eye(len(agents)))
        
        y_grad_data = [ agent.get_y_grad() for agent in agents] 
        mixed_y = self.propagate_data(y_grad_data)

        for k in range(self.N):
            agents[k].set_y_grad(mixed_y[k]) # Control Immutable
        
    def get_average_weight(self,agents):
        weight_data = [ agent.get_weight() for agent in agents] 
        avg_weight = self.get_average(weight_data)
        return avg_weight

    def get_average_action(self,agents):
        action_data = [ agent.get_action() for agent in agents] 
        avg_action = self.get_average(action_data)
        return avg_action
    
    def get_average(self,data):
        N = len(data)
        num_param = data[0].shape
        avg_data = np.zeros(num_param)
        for k in range(N): 
            avg_data += data[k]/N
        
        return avg_data
    
    
    def propagate_data(self,data):
        
        mixed_data = [None]*len(data)
        N = len(data)
        num_param = data[0].shape
        
        for m in range(N):
            data_dict = np.zeros(num_param) ## Be Careful
            for k in range(N):
                data_dict += self.W[m,k]*data[k]
            mixed_data[m] = data_dict
        
        return mixed_data

    def get_consensus_error(self,data):
        x_bar = self.get_average(data)
        error = 0 
        N = len(data)
        for k in range(N): 
            error += np.linalg.norm(data[k]-x_bar)
        return error


class DatasetModel:

    def __init__(self,dsname=None,X=None,y=None,num_agent=3,mb_size=1,normalize=True):
        
        self.dsname=dsname
        self.normalize=normalize
        
        if dsname==None:
            self.X = X
            self.y = y
        else:
            self.loaddataset()
        
        self.num_param=self.X.shape[1]
        self.mb_size = mb_size
        self.dssize = self.X.shape[0]

        self.num_agent = num_agent

        self.max_sample = 500000

        shuffled_idx=np.random.permutation(self.dssize)
        sample_per_agent = self.dssize//self.num_agent
        self.agent_dict={m:shuffled_idx[m*sample_per_agent:(m+1)*sample_per_agent ].tolist() for m in range(num_agent)}

        self.valid_idx = shuffled_idx[:num_agent*sample_per_agent]
    
    def loaddataset(self):
        if self.dsname=='a9a':
            train_data_file = '../datasets/a9a'
            X,y = sk.datasets.load_svmlight_file(train_data_file)
            
        elif self.dsname=='rcv':
            train_data_file = '../datasets/rcv1_train.binary'
            X,y = sk.datasets.load_svmlight_file(train_data_file)
           
        elif self.dsname=='ijcnn':
            train_data_file = '../datasets/ijcnn1'
            X,y = sk.datasets.load_svmlight_file(train_data_file)
            
        elif self.dsname=='covtype':
            train_data_file = '../datasets/covtype.libsvm.binary'
            X,y = sk.datasets.load_svmlight_file(train_data_file) 
           
        elif self.dsname=='SUSY':
            train_data_file = '../datasets/SUSY'
            X,y = sk.datasets.load_svmlight_file(train_data_file)
            
        elif self.dsname=='HIGGS_100k':
            # X = np.load('../datasets/HIGGS100k_X.npy',allow_pickle=True)
            X = sp.sparse.load_npz('../datasets/HIGGS100k_X_sp.npz')
            y = np.load('../datasets/HIGGS100k_y.npy',allow_pickle=True)
        elif self.dsname=='HIGGS_1m':
            # X = np.load('../datasets/HIGGS100k_X.npy',allow_pickle=True)
            X = sp.sparse.load_npz('../datasets/HIGGS1m_X_sp.npz')
            y = np.load('../datasets/HIGGS1m_y.npy',allow_pickle=True)
        elif self.dsname=='HIGGS_500k':
            # X = np.load('../datasets/HIGGS100k_X.npy',allow_pickle=True)
            X = sp.sparse.load_npz('../datasets/HIGGS500k_X_sp.npz')
            y = np.load('../datasets/HIGGS500k_y.npy',allow_pickle=True)
        
        elif self.dsname=='SUSY_500k':
            # X = np.load('../datasets/HIGGS100k_X.npy',allow_pickle=True)
            X = sp.sparse.load_npz('../datasets/SUSY500k_X_sp.npz')
            y = np.load('../datasets/SUSY500k_y.npy',allow_pickle=True)

            
        
        
        

        X = X.toarray()

        if self.dsname=='rcv':
                x_std = np.std(X,axis=0)
                col_to_drop = np.where(x_std<1e-2)[0]
                X=np.delete(X, col_to_drop, axis=1)
                print(X.shape)
                if X.shape[1]>1000:
                    print(X.shape)
                    x_std = np.std(X,axis=0)
                    idx = np.argsort(-x_std)[:1000]
                    X=X[:,idx]
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
        if self.normalize:
            self.x_mean = X_train.mean(axis=0)
            self.x_std = X_train.std(axis=0)
            X_train = (X_train-self.x_mean)/self.x_std
            X_test = (X_test-self.x_mean)/self.x_std
            
        self.X = X_train
        self.y = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_sample(self,agent=1):

        idx = random.sample(self.agent_dict[agent],k=self.mb_size)
        X_mb = self.X[idx,:]
        y_mb = self.y[idx]

        return X_mb,y_mb


    def get_ds(self):

        return self.X[self.valid_idx,:],self.y[self.valid_idx]

    def get_test_set(self):
        return self.X_test,self.y_test


def sample_sphere(sh):
    w1 = np.random.randn(*sh)
    w1 = w1/np.linalg.norm(w1,axis=1,keepdims=True)
    return w1
