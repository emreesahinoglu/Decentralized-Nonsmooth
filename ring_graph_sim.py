import numpy as np
import sklearn as sk
import sklearn.datasets
import random
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import pickle
from SVMAgent import SVMAgent, SVMOracle , DatasetModel, ComNetwork

from sim_algorithms import algorithm,run_algo,gen_random_dsm,gen_sparse_dsm,second_eig,gen_dsm_nbd


def run_sim(dsname='a9a',D=1,lr=1e-4,oracle_setup='first',num_iter=50000,T_restart=500,T_print=500,num_agent=20,W=None):
    dataset = DatasetModel(dsname=dsname, mb_size= batch_size,num_agent=num_agent)
    num_param = dataset.num_param
    print(f'Dataset Name : {dsname} Oracle : {oracle_setup} num_sample : {dataset.dssize} num_param : {num_param} batch_size : {batch_size} ')

    lam = 1e-5 * batch_size / dataset.dssize
    print(f'Lambda : {lam:.4e}')

    
    if W is None :
        W = gen_sparse_dsm(num_agent)
        rho = second_eig(W)  
        print(f'rho : {rho:.2f}') 
        agents = list()
        for k in range(num_agent):
            agents.append(SVMAgent ( num_param , id=k, D = D,lr=lr)) 
    else:
        rho = second_eig(W)     
        print(f'rho : {rho:.2f}')
        
        D_adj = D*(1-rho)
        lr_adj = lr* np.power(1-rho,1.5)
        print(f'D and lr adjusted to rho')
        agents = list()
        for k in range(num_agent):
            agents.append(SVMAgent ( num_param , id=k, D = D_adj,lr=lr_adj))
    
    oracle = SVMOracle(alpha=2,lam=lam)

    com_net = ComNetwork(W)

    agents , x_bar , grad_norm , func_value , gr_per,fv_per,acc_per = run_algo(agents,dataset,oracle,com_net,num_iter=T,T_restart=T_restart,T_print=T_print,oracle_setup=oracle_setup)

    return gr_per,fv_per,acc_per




def save_results(config,results,oracle='first',folder='rho'):

    data = {'config' : config, 'results' : results}
    date_str = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    dsname = config['dsname']
    ora = config['oracle']
    num_agent = results['num_agent']
    pickle_file_name = f'results/{folder}/{dsname}_{num_agent}_{date_str}.pkl'
    with open(pickle_file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


dsname='rcv'
oracle_setup='first'

T=30000
T_print=100
T_restart=100
num_agent = 20
batch_size=1
D=1e-3
# lr=0.1*D


dsnames=['ijcnn','rcv','SUSY_500k']

for dsname in dsnames:
    for num_nbd in range(3,7):
        W = gen_dsm_nbd(num_agent,num_nbd)
        rho = second_eig(W)
        lr = 0.1*D

        config = {'dsname':dsname, 'T' : T, 'Tp':T_print, 'Tr':T_restart , 'D' : D , 'batch_size' : batch_size , 'lr' : lr,'oracle':oracle_setup,'num_nbd':num_nbd,'rho':rho}
        print(config)
        grad_norm, func_value,acc_per = run_sim(dsname=dsname,D=D,lr=lr,oracle_setup=oracle_setup,num_iter=T,T_restart=T_restart,T_print=T_print,num_agent=num_agent,W=W)
        
        results={'grad_norm':grad_norm, 'func_value':func_value,'acc':acc_per,'num_agent':num_agent}
        save_results(config,results,oracle=oracle_setup,folder='rho')



