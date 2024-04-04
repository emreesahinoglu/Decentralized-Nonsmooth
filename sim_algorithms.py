import numpy as np
from scipy.stats import unitary_group
import networkx as nx


def algorithm(agents,dataset,oracle,com_net,oracle_setup='zero'):
    N = len(agents)
    
    for k in range(N):
        
        grad_point,new_weight = agents[k].get_grad_point()
        agents[k].set_weight(new_weight)
        
        
        
        x1,y1 = dataset.get_sample(k)
        if oracle_setup=='zero':
            grad = oracle.get_zo_grad(grad_point,x1,y1,delta=1e-2)
        else:
            grad = oracle.get_gradients(grad_point,x1,y1)

        agents[k].action_grad_update(grad)

    com_net.propagate_actions(agents)
    com_net.propagate_weights(agents)
    
    return agents

def run_algo(agents,dataset,oracle,com_net,num_iter=50000,T_restart=500,T_print=500,oracle_setup='first'):
    avg_weight = []
    gr_per = []
    fv_per = []
    acc_per = []

    print(f'Oracle : {oracle_setup} initiated')

    grad_norm = np.zeros(int(num_iter))
    func_value = np.zeros(int(num_iter))
    X1,y1 = dataset.get_ds()
    X_test,y_test =dataset.get_test_set()


    for k in range(int(num_iter)):
        agents = algorithm(agents,dataset,oracle,com_net,oracle_setup=oracle_setup)
        x_bar = com_net.get_average_weight(agents)
        avg_weight.append(x_bar)
        
        if k%T_restart==(T_restart-1):
            for agent in agents:
                agent.initialize_action()
            print(f'Online Algorithm Restarted at Iter : {k}')

        if k %T_print ==0:
            
            epoch_grad = [oracle.get_gradients(w,X1,y1) for w in avg_weight]
            gr = com_net.get_average(epoch_grad)

            loss =  oracle.get_fn_val(x_bar,X1,y1)
            gr_per_norm=np.linalg.norm(gr)
            
            print(f'Iteration {k} Grad Norm : {gr_per_norm:.4f} Loss : {loss:.4f}')
            
            test_acc = oracle.get_accuracy(x_bar,X_test,y_test)
            print(f'Test Accuracy : {test_acc:.2f}')

            acc_per.append(test_acc)
            fv_per.append(loss)
            gr_per.append(gr_per_norm)
            avg_weight = []


    return agents,x_bar, grad_norm,func_value, gr_per,fv_per,acc_per


def sgd_algorithm(agents,dataset,oracle,com_net):
    N = len(agents)
    
    for k in range(N):
        
        weight = agents[k].get_weight()
        x1,y1 = dataset.get_sample(k)

        grad = oracle.get_gradients(weight,x1,y1)
        agents[k].update_weight(grad)

    com_net.propagate_weights(agents)
    
    return agents

def run_sgd_algo(agents,dataset,oracle,com_net,num_iter=50000,T_print=500):

    avg_weight = []
    gr_per = []

    fv_per = []
    acc_per = []

    print(f'SGD Algorithm initiated')

    X1,y1 = dataset.get_ds()
    X_test,y_test =dataset.get_test_set()

    for k in range(int(num_iter)):
        agents = sgd_algorithm(agents,dataset,oracle,com_net)
        x_bar = com_net.get_average_weight(agents)
        avg_weight.append(x_bar)
 

        if k %T_print ==0:
            
            # epoch_grad = [oracle.get_gradients(w,X1,y1) for w in avg_weight]
            # gr = com_net.get_average(epoch_grad)
            x_bar = com_net.get_average(avg_weight)
            gr = oracle.get_gradients(x_bar,X1,y1)
            loss =  oracle.get_fn_val(x_bar,X1,y1)
            gr_per_norm=np.linalg.norm(gr)
            
            print(f'Iteration {k} Grad Norm : {gr_per_norm:.4f} Loss : {loss:.4f}')
            
            test_acc = oracle.get_accuracy(x_bar,X_test,y_test)
            print(f'Test Accuracy : {test_acc:.2f}')

            acc_per.append(test_acc)
            fv_per.append(loss)
            gr_per.append(gr_per_norm)
            avg_weight = []


    return agents,x_bar, gr_per, fv_per, acc_per



def dgfm_algorithm(agents,dataset,oracle,com_net,delta=1e-3):
    N = len(agents)
    
    for k in range(N):
        x1,y1 = dataset.get_sample(k)
        w=agents[k].get_weight()
        grad = oracle.get_zo_grad(w,x1,y1,delta=delta)
        agents[k].update_y_grad(grad)
    
    com_net.propagate_dgfm_grad(agents)

    for k in range(N):
        agents[k].update_weight()   
    
    com_net.propagate_weights(agents) 
    
    return agents

def run_dgfm_algo(agents,dataset,oracle,com_net,num_iter=1e3,T_print=500):
    
    avg_weight = []
    gr_per = []

    fv_per = []
    acc_per = []

    print(f'DGFM initiated')

    grad_norm = np.zeros(int(num_iter))
    func_value = np.zeros(int(num_iter))
    X1,y1 = dataset.get_ds()
    X_test,y_test =dataset.get_test_set()


    for k in range(int(num_iter)):
        agents = dgfm_algorithm(agents,dataset,oracle,com_net)
        x_bar = com_net.get_average_weight(agents)
        avg_weight.append(x_bar)

        if k %T_print ==0:
            

            # epoch_grad = [oracle.get_gradients(w,X1,y1) for w in avg_weight]
            # gr = com_net.get_average(epoch_grad)
            x_bar = com_net.get_average_weight(agents)
            gr = oracle.get_gradients(x_bar,X1,y1)
            loss =  oracle.get_fn_val(x_bar,X1,y1)
            gr_per_norm=np.linalg.norm(gr)
            
            print(f'Iteration {k} Grad Norm : {gr_per_norm:.4f} Loss : {loss:.4f}')
            test_acc = oracle.get_accuracy(x_bar,X_test,y_test)
            print(f'Test Accuracy : {test_acc:.2f}')

            fv_per.append(loss)
            gr_per.append(gr_per_norm)
            acc_per.append(test_acc)
            avg_weight = []


    return agents,x_bar, grad_norm,func_value, gr_per,fv_per,acc_per

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

def gen_random_dsm(k):
    x = unitary_group.rvs(k)
    W = np.abs(x)**2
    return W

def gen_dsm_nbd(n,k):
    assert 2*k<n-1 , 'k must be smaller than n/2'
    W = np.zeros((n,n))
    for m in range(n):
        for j in range(m-k,m+k+1):
            if j<0:
                j=j+n
            if j>=n:
                j=j-n
            W[m,j]=1/(2*k+1)
    return W

def second_eig(W):
    Q,D = np.linalg.eig(W)
    Q = -np.sort(-np.abs(Q))
    return Q[1]

def gen_er_graph(num_agent,p):
    W1=np.eye(num_agent)
    while not check_conn(W1):
        print('We need connected network')
        G= nx.erdos_renyi_graph(num_agent,p)
        W = nx.adjacency_matrix(G).todense()
        w1 = W.sum(axis=0)
        W1=W/num_agent
        for k in range(num_agent):
            W1[k,k]=1-w1[k]/num_agent
    return W1

def check_conn(W1):
    D,Q = np.linalg.eig(W1)
    D1=np.sort(np.abs(D))[::-1]
    print(D1)
    return D1[1]<0.999
