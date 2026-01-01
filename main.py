from qiskit import *
from qiskit_aer import *
from qiskit.quantum_info import *
from qiskit.visualization import *
from qiskit.quantum_info import partial_trace
from qiskit.circuit.library import UnitaryGate
from qiskit_aer.noise  import  NoiseModel , depolarizing_error
import matplotlib.pyplot as plt
import random
import numpy as np
import math
import os
import datetime

gap_represent = 'origin'  # [log, origin]
code_mode = 'annealing'  # [annealing, energy_gap, both, plot, test ...]
time_step_mode = 'standard'  # [flexible, standard]
# schedule_func_init = 'quadratic'  # [quadratic, tanh]
schedule_funcs = ['quadratic', 'tanh', 'arctan']

n=6
m=10*n  # Set m to be n^2
eta=0.1
kappa = 1  # The scaling of H_P (The objective hamiltonian) in the energy_gap calculation.

rounds = 1  # How many eta's do we want to run in eta_candidate
repeat = 1
eta_candidate = np.linspace(0.1, 0.3, 6)

T=50
M=10*T  # Annealing step
dt_standard=T/M
shots_sampling = 100
# c=5  # The scaling of H_P in the real code

delta_min = 0.5
amplitude_quad = 3  # The rate of gap increasing regarding the center
amplitude_tanh = 1  # The rate of gap increasing regarding the center
amplitude_arctan = 3  # The rate of gap increasing regarding the center
s_star = 0.5  # The step min_energy_gap occurs

s=[]
A=[]
y=[]
dt_array = []

now  = datetime.datetime.now()
folder_name = now.strftime("RST_%m-%d_%H%M")
plt_index = 1

#region === finetune time gap based on energy gap ===
def get_gap(s, schedule_func):
    if schedule_func == 'quadratic':
        return np.sqrt(delta_min**2 + amplitude_quad * (s - s_star)**2)
    elif schedule_func == 'tanh':
        return np.tanh((s - s_star) / amplitude_tanh)
    elif schedule_func == 'arctan':
        return np.arctan((s - s_star) / amplitude_arctan)
    else:
        raise ValueError("this schedule_func is not implemented.")

def scheduling(schedule_func):
    # if schedule_func == 'quadratic':
        # s_grid = np.linspace(0, 1, M)
    # elif schedule_func == 'tanh':
    #     s_grid = np.linspace(-1, 1, M)
    # else:
    #     raise ValueError("this schedule_func is not implemented.")
    global dt_array, plt_index
    s_grid = np.linspace(0, 1, M)
    weights = get_gap(s_grid, schedule_func)**2
    dt_array = (weights / np.sum(weights)) * T
    if code_mode == 'plot' or code_mode == 'annealing':
        x = np.linspace(0, M-1, M)
        plt.figure(plt_index)
        plt.plot(x, dt_array, label=f'time weight')
        plt.legend()
        plt.title(f"time weight_{schedule_func}")
        plt.xlabel("step")
        plt.ylabel("dt array")
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        plt.savefig(f"{folder_name}/dt_weight_{schedule_func}.png", dpi=300, bbox_inches='tight')
        plt.close()
        plt_index += 1
#endregion

#region === Initialization of matrix A and error vector e ===
def init():
    print("Qubit number (n):", n)
    global s, A, y
    s = []
    A = []
    y = []
    for i in range(n):
        s.append(random.randint(0,1))
    for i in range(m):
        A.append([])
        for j in range(n):
            A[i].append(random.randint(0,1))
        
        b=0
        for j in range(n):
            b^=A[i][j]*s[j]
        y.append(b)
    
    p=[]
    for i in range(m):
        p.append(i)
    random.shuffle(p)
    for i in range(int(eta*m)):
        y[p[i]]^=1
    
    print("Correct answer (s):", s)
#endregion

#region === Run the annealing algorithm to get the ground state ===
def run(repeat_idx, time_step_mode, schedule_func):
    t=0
    global A, s, y, plt_index

    circ = QuantumCircuit(n+1,n)
    for i in range(n):
        circ.h(i)

    current_time = 0.0
    for i in range(M):
        if time_step_mode == 'standard':
            dt = dt_standard
            t=(i+0.5)*dt
            alpha=(1-t/T)
            beta=t/T
        elif time_step_mode == 'flexible':
            dt = dt_array[i]
            t_mid = current_time + 0.5 * dt
            current_time += dt
            progress = t_mid / T
            alpha = 1.0 - progress
            beta = progress
            # print(t_mid)
        else:
            raise ValueError("Not correct value of \'time_step_mode\'")

        angle=2.0*beta*dt_standard

        for k in range(m):
            for j in range(n):
                if A[k][j]==1:
                    circ.cx(j,n)
            if y[k]==1:
                circ.x(n)

            circ.rz(angle/(m/n),n)

            if y[k]==1:
                circ.x(n)
            for j in range(n-1,-1,-1):
                if A[k][j]==1:
                    circ.cx(j,n)

        for j in range(n):
            circ.rx(2.0*alpha*dt_standard,j)

    circ.measure(range(n),range(n))

    simulator=AerSimulator()
    compiled_circuit=transpile(circ,simulator)
    job=simulator.run(compiled_circuit, shots=shots_sampling)
    result=job.result()
    counts = result.get_counts(circ)
    
    # === Save the annealing result ===
    categories = []
    values = []
    instances = []

    for (a,b) in counts.items():
        s=[1 if x=='1' else 0 for x in a[::-1]]
        E=0
        for i in range(m):
            sum=y[i]
            for j in range(n):
                sum^=A[i][j]*s[j]
            E+=sum
        # print(s,E)
        # if E==0:
            # print(s)
        categories.append(E)
        values.append(b)
        instances.append(s)
    
    # === Merging all the data, classified by energy ===
    merged_data = {}
    instances_data = {}
    for category, value, instance in zip(categories, values, instances):
        if category in merged_data:
            merged_data[category] += value
        else:
            merged_data[category] = value
            instances_data[category] = instance

    min_energy = min(instances_data.items(), key=lambda item: item[0])
    print(min_energy)
    print('Merged data: ', merged_data)

    plt.figure(plt_index)
    plt.xlim(0, 50)
    # plt.figure(figsize=(10, 6))
    plt.bar(merged_data.keys(), merged_data.values())
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_name = time_step_mode + '_' + schedule_func
    plt.savefig(f"{folder_name}/{file_name}_{eta}_{repeat_idx}.png", dpi=300, bbox_inches='tight')
    plt.close()
    plt_index += 1
    # plt.show()
#endregion

#region === Construct Hamiltonian H_D ===
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.array([[1, 0], [0, 1]], dtype=complex)
def kronecker_product_sum(N, operator_list):
    """
    计算 N 个量子比特系统中某个操作符（如 H_D 或 H_P）的总和。
    operator_list: 包含 (operator, index) 对的列表
    """
    H = np.zeros((2**N, 2**N), dtype=complex)
    
    for op, target_idx in operator_list:
        H_term = 1  # 初始为 1x1 矩阵，将被迭代地替换为正确的基态矩阵
        
        for i in range(N):
            if i == target_idx:
                H_term = np.kron(H_term, op) # 作用在目标比特上
            else:
                H_term = np.kron(H_term, I) # 作用在其他比特上（即单位矩阵）
        
        H += H_term
    return H / n

# driver_ops = []
# for i in range(n):
#     # -sigma_x 作用在每个比特上
#     driver_ops.append((-sigma_x, i))
# H_D = kronecker_product_sum(n, driver_ops)
#endregion

#region === Construct Hamiltonian H_P ===
def construct_problem_hamiltonian(n, m, A, y):
    """
    构造 H_P = sum_{i=1}^m (-1)^{y_i} * Z^{A_i}
    """
    H_P = np.zeros((2**n, 2**n), dtype=complex)
    for i in range(m):
        # 计算 (-1)^y_i
        coeff = (-1)**y[i]
        
        # 构造 Z^{A_i} 项：I \otimes Z \otimes I ...
        term = np.array([[1]], dtype=complex)
        for j in range(n):
            if A[i][j] == 1:
                term = np.kron(term, sigma_z)
            else:
                term = np.kron(term, I)
        
        H_P += coeff * term   
    H_P = H_P*kappa / m
    return -H_P

#endregion

#region === Calculate energy gap in each step ===
H_D = H_P = 0
gap = []
def construct_ham(i):
    t=(i+0.5)*dt_standard
    alpha=(1-t/T)
    beta=t/T
    H = alpha * H_D + beta * H_P
    return H

def calculate_energy_gap():
    """
    计算哈密顿量 H 的基态和第一激发态能量间隙。
    """
    # 使用 scipy.linalg.eigh 求解厄米矩阵的特征值。
    # only_eigenvalues=True 可以加快计算，但我们需要特征值本身。
    # 默认返回所有特征值，按升序排列。
    global gap
    gap = []
    for i in range(M):
        H = construct_ham(i)
        eigenvalues = np.linalg.eigvalsh(H)
        
        # 基态能量 E0 是最小的特征值 (第一个)
        E0 = eigenvalues[0]
        
        # 第一激发态能量 E1 是第二小的特征值 (第二个)
        E1 = eigenvalues[1]
        
        # 能量间隙
        if gap_represent == "log":
            gap_temp = math.log(E1-E0, 10)
        elif gap_represent == "origin":
            gap_temp = E1-E0
        else:
            raise NotImplementedError
        gap.append(gap_temp)
        # print(gap[i])
#endregion

#region === Energy Gap calculation for different number of qubits ===
if code_mode == 'energy_gap' or code_mode == 'both':
    n_max = 7
    gap_min = []
    for i in range(3, n_max+1, 1):
        n = i
        m = 10*n
        # === Construct basic hamiltonian ===
        driver_ops = []
        for i in range(n):
            # -sigma_x 作用在每个比特上
            driver_ops.append((-sigma_x, i))
        H_D = kronecker_product_sum(n, driver_ops)
        # === Construct objective hamiltonian ===
        init()
        H_P = construct_problem_hamiltonian(n, m, A, y)
        # == Calculate energy gap ===
        calculate_energy_gap()
        gap_min.append(np.min(gap))
        x = np.linspace(0, M-1, M)
        plt.plot(x, gap, label=f'n={n},m={m}')
        plt.legend()
        plt.title("Energy gap")
        plt.xlabel("step")
        if gap_represent == 'log':
            plt.ylabel("log(Energy)")
        else:
            plt.ylabel("Energy")
        # plt.show()
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    plt.savefig(f"{folder_name}/energy_gap.png", dpi=300, bbox_inches='tight')
    # print(gap_min)
#endregion

if code_mode == 'annealing' or code_mode == 'both':
    for i in range(rounds):
        # eta = 0.05*(i+1)
        eta = eta_candidate[i]
        for j in range(repeat):
            init()
            run(repeat_idx=j, time_step_mode='standard', schedule_func='')
            for function in schedule_funcs:
                scheduling(schedule_func=function)
                run(repeat_idx=j, time_step_mode='flexible', schedule_func=function)