import math
import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt


lambda_ = np.arange(0.05, 0.55, 0.05) 
num_slots = 100000


def model_func(lambda_, num_slots):
    N = []
    G = []

    n0 = 0 

    for _ in range(num_slots):
   
        p_i = np.random.poisson(lambda_)


        if n0 > 0:
            p_transmit = 1 / n0
            r_i = np.random.binomial(n0, p_transmit)
        else:
            r_i = 0

        if r_i == 1:

            n0 -= 1
        elif r_i > 1:

            pass


        n0 += p_i
        N.append(n0)


        G.append(1 if r_i == 1 else 0)


    avg_n = np.mean(N)

    avg_t = avg_n / lambda_

    avg_g = np.mean(G)

    return avg_n, avg_t, avg_g

result_list_ns = []
result_list_ts = []
result_list_gs = []


for lambd in lambda_:
    avg_n, avg_t, avg_g = model_func(lambd, num_slots)
    result_list_ns.append(avg_n)
    result_list_ts.append(avg_t)
    result_list_gs.append(avg_g)

line_width = 1.5
marker_size = 6



def chance(i):
    if i == 1:
        return 1
    if i <= 0:
        return 0
    return (1 - 1/i)**(i - 1)
def puasson(lmbda, j):
    if j < 0:
        return 0
    if j == 0: 
        return math.exp(-lmbda)
    return (lmbda ** j) * math.exp(-lmbda) / factorial(j)
def from0toj(lmbda, j):
    return puasson(lmbda, j)
def fromitoi_(lmbda, i):
    if i < 1:
        return 0
    return chance(i) * puasson(lmbda, 0)
def stayi(lmbda, i):
    return (1 - chance(i)) * puasson(lmbda, 0) + chance(i) * puasson(lmbda, 1)
def fromitoj(lmbda, i, j):
    if j < i:
        return 0
    return chance(i) * puasson(lmbda, j - i + 1) + (1 - chance(i)) * puasson(lmbda, j - i)

def system(lmbda, N):
    transition_matrix = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            if i == 0:
                transition_matrix[i, j] = from0toj(lmbda, j)
            elif i == j:
                transition_matrix[i, j] = stayi(lmbda, i)
            elif i < j:
                transition_matrix[i, j] = fromitoj(lmbda, i, j)
            elif i == j + 1:
                transition_matrix[i, j] = fromitoi_(lmbda, i)
            else:
                transition_matrix[i, j] = 0 
    matrix = np.vstack([transition_matrix.T - np.eye(N + 1), np.ones(N + 1)])

    coeffs = np.zeros(N + 2)
    coeffs[-1] = 1

    pi = np.linalg.lstsq(matrix, coeffs, rcond=None)[0]
    return pi

def mean_abonents(lambdas, K):
    mean = []
    for i in lambdas:
        stat = system(i, K)
        average_users = np.dot(stat, np.arange(K + 1))
        mean.append(average_users)
    return mean

Numbers = [10, 50, 100, 1000] 
lambdas = np.arange(0.05, 0.55, 0.05)

final_result = []
for K in Numbers:
    mean = mean_abonents(lambdas, K)
    final_result.append(mean)

print(f"{'K (Макс. абонентов)':<25}{'λ (Интенсивность)':<20}{'Среднее количество абонентов':<30}")
print("-" * 75) 
for i, K in enumerate(Numbers):
    for lam, avg in zip(lambdas, final_result[i]):
        print(f"{K:<25}{lam:<20.2f}{avg:<30.4f}")
    print()

import matplotlib.pyplot as plt
import numpy as np

def average_counter(lambda_values, average_results, K_values, result_list_ns):
    plt.style.use('seaborn-darkgrid')

    plt.figure(figsize=(12, 7))


    colors = plt.cm.viridis(np.linspace(0, 1, len(K_values)))  
    markers = ['o', 's', 'D', '^', 'v']  

    for idx, K in enumerate(K_values):
        plt.plot(lambda_values, average_results[idx], 
                 label=f"Макс. абонентов = {K}", 
                 marker=markers[idx % len(markers)], 
                 linestyle='-', 
                 color=colors[idx],
                 markersize=8, 
                 linewidth=2)

    plt.plot(lambda_values, result_list_ns, 
             label="Имитационное моделирование для N = 100000", 
             marker='o', 
             linestyle='-', 
             color='purple', 
             markersize=8, 
             linewidth=3)

    plt.xlabel("λ", fontsize=14, fontweight='bold') 
    plt.ylabel("Среднее количество абонентов", fontsize=14, fontweight='bold')  
    plt.yscale("log")  
    plt.title("График зависимости среднего числа абонентов от интенсивности λ", fontsize=16, fontweight='bold') 
    plt.legend(fontsize=12)
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  
    plt.tight_layout() 
    plt.show()


average_counter(lambdas, final_result, Numbers,result_list_ns)