import numpy as np
from Algorithms import GEPP, GEM,GE
from .get_gf import get_gf
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import math

def run_random_ensemble(methods,n_list=[5],samples=1000,end_gf=False):
    results={}

    # Name in rsullts dict is method name with n appended
    for n in n_list:
        A_list=[np.random.rand(n,n) for _ in range(samples)]

        for method in tqdm(methods):
            # n
            print("Running ",method," for n=",n)
            results[method+"_"+str(n)]=[]
            for A in A_list:
                if method=='GE':
                    gf=get_gf(A,method="GE",end_gf=end_gf)
                elif method=='GEPP':
                    gf=get_gf(A,method="GEPP",end_gf=end_gf)

                elif method=='GEM':
                    gf=get_gf(A,method="GEM",end_gf=end_gf)

                else:
                    raise NotImplementedError
                
                results[method+"_"+str(n)].append(gf)

    return results

def compute_stats(results):
    df=pd.DataFrame(results)
    return df.describe()

def plot_results(results:dict,log=False,plot_type='box',save_path="Plots/test.png"):
    results=results.copy()
    # resulrs is a dict of lists
    if log:
        for method in results:
            results[method]=np.log(results[method])
    df=pd.DataFrame(results)
    if plot_type=='box':
        plot=df.boxplot()
        if log:
            plt.ylabel("log(growth factor)")
        else:
            plt.ylabel("Growth factor")
        plt.title("Growth factor distribution")

        plt.savefig(save_path)
    elif plot_type=='hist':
        # draw pdf with sns
        for method in results:
            sns.distplot(results[method],hist=False,label=method)
        # label axes with log if necessary

        if log:
            plt.xlabel("log(growth factor)")
        else:
            plt.xlabel("Growth factor")
        plt.ylabel("PDF")
        plt.legend()

        plt.savefig(save_path)

        plt.title("Growth factor distribution")

        # return figure
        return plt.gcf()
       

    else:
        raise NotImplementedError        
    

def run_pivot_ensemble(methods,n_list=[5],samples=1000,end_gf=False):
    results={"Optimal":[]}
    for method in methods:
        results[method]=[]
    for n in n_list:
        for i in tqdm(range(samples)):
            A=np.random.randn(n,n)
            result=pivot_results(A,methods=methods,end_gf=end_gf)
            for method in methods:
                results[method].append(result[method])
            results["Optimal"].append(result["Optimal"])
    return results    
    
    # Is partial pivoting on an average as good as a random permutation? Ie. is P_pp=P_rand?

def pivot_results(A,methods=["GEPP","GE","GEM"],verbose=False,end_gf=False):
    n=A.shape[0]
    growth_factor_list=[]
    if verbose:
        for m in tqdm(itertools.permutations(A),total=math.factorial(n)):
            growth_factor_list.append(get_gf(np.array(m)))
    else:
        for m in itertools.permutations(A):
            growth_factor_list.append(get_gf(np.array(m)))
    
    optimal_gf=min(growth_factor_list)
    results={}

    for method in methods:
        results[method]=get_gf(A,method=method,end_gf=end_gf)
    
    results["Optimal"]=optimal_gf

    return results
