import numpy as np
import sklearn as sk
import sklearn.datasets
import random
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import pickle
from SVMAgent import SGDAgent, SVMOracle , DatasetModel, ComNetwork

from sim_algorithms import sgd_algorithm,run_sgd_algo

def gen_sparse_dsm(k,p=1/3):
    W = (1-2*p)* np.eye(k)
    W[0,k-1]=p
    W[k-1,0]=p
    W[0,1]=p
    W[k-1,k-2]=p
    for k1 in range(1,k-1):
        W[k1,k1-1] = p
        W[k1,k1+1] = p

    return W 


def run_sim(dsname='a9a',D=1,lr=1e-4,oracle_setup='first',num_iter=50000,T_print=500):
    dataset = DatasetModel(dsname=dsname, mb_size= batch_size,num_agent=num_agent)
    num_param = dataset.num_param
    print(f'Dataset Name : {dsname} num_sample : {dataset.dssize} num_param : {num_param} ')

    lam = 1e-5 * batch_size / dataset.dssize
    print(f'Lambda : {lam:.4e}')

    agents = list()
    for k in range(num_agent):
        agents.append(SGDAgent ( num_param , id=k,lr=lr))

    W = gen_sparse_dsm(num_agent)
    oracle = SVMOracle(alpha=2,lam=lam)

    com_net = ComNetwork(W)

    agents , x_bar , gr_per,fv_per,acc_per = run_sgd_algo(agents,dataset,oracle,com_net,num_iter=T,T_print=T_print)

    return gr_per,fv_per,acc_per


def save_results(config,results):

    data = {'config' : config, 'results' : results}
    date_str = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    dsname = config['dsname']
    pickle_file_name = f'results/sgd/{dsname}_{date_str}.pkl'
    with open(pickle_file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)



T=50000
T_print=500
T_restart=500
num_agent = 20
batch_size=128
D=1e-3
lr=1e-5


dsnames=['a9a','ijcnn','covtype','HIGGS_500k','rcv','SUSY_500k']


D_values = [1e-4,1e-3,1e-2]
lr_values = [5e-4,1e-3,1e-2]


for dsname in dsnames:
    for lr in lr_values:
        
        config = {'dsname':dsname,'algo':'dsgd', 'T' : T, 'Tp':T_print,  'batch_size' : batch_size , 'lr' : lr}
        print(config)
        grad_norm, func_value,acc_per = run_sim(dsname=dsname,lr=lr,num_iter=T,T_print=T_print)
        results={'grad_norm':grad_norm, 'func_value':func_value,'acc':acc_per}
        save_results(config,results)



