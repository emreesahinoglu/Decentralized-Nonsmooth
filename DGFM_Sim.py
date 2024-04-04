import numpy as np
import datetime
import pickle
import sklearn as sk
import sklearn.datasets
import random
import seaborn as sns
import matplotlib.pyplot as plt
from SVMAgent import DGFMAgent, SVMOracle , DatasetModel, ComNetwork

from sim_algorithms import run_dgfm_algo,dgfm_algorithm,gen_sparse_dsm


def save_results(config,results):

    data = {'config' : config, 'results' : results}
    date_str = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    dsname = config['dsname']
    pickle_file_name = f'results/dgfm/{dsname}_{date_str}.pkl'
    with open(pickle_file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def run_dgfm_sim(dsname='a9a',lr=1e-4,num_iter=50000,T_restart=500,T_print=500):

    dataset = DatasetModel(dsname=dsname, mb_size= batch_size,num_agent=num_agent)
    num_param = dataset.num_param
    print(f'Dataset Name : {dsname} Size : num_sample : {dataset.dssize} num_param : {num_param}')

    lam = 1e-5 * batch_size / dataset.dssize

    agents = list()
    for k in range(num_agent):
        agents.append(DGFMAgent(num_param , mb_size=batch_size, id=k,lr=lr))
        
    W = gen_sparse_dsm(num_agent)
    oracle = SVMOracle(alpha=2,lam=lam)

    com_net = ComNetwork(W)

    agents, x_bar, grad_norm,func_value,gr_per,fv_per,acc_per = run_dgfm_algo(agents,dataset,oracle,com_net,num_iter=num_iter,T_print=T_print)

    return gr_per,fv_per,acc_per

dsnames=['a9a','ijcnn','covtype','HIGGS_500k','rcv','SUSY_500k']

T=1000
T_print=500
num_agent = 20
batch_size=128
lr_values=[1e-3,1e-2]



for dsname in dsnames:
    for lr in lr_values:

        config = {'dsname':dsname,'algorithm':'dgfm', 'T' : T, 'Tp':T_print, 'batch_size' : batch_size , 'lr' : lr}
        print(config)
        gr_per, fv_per,acc_per = run_dgfm_sim(dsname=dsname,lr=lr,num_iter=T,T_print=T_print)
        results={'grad_norm':gr_per, 'func_value':fv_per,'acc':acc_per}
        save_results(config,results)

config = {'dsname':dsname,'alogrithm':'dgfm', 'T' : T, 'Tp':T_print, 'batch_size' : batch_size , 'lr' : lr}
print(config)
results={'grad_norm':gr_per, 'func_value':fv_per,'acc':acc_per}
save_results(config,results)