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
code_mode = 'annealing'  # [annealing, energy_gap, both]

n=7
m=10*n  # Set m to be n^2
eta=0.1
kappa = 1  # The scaling of H_P (The objective hamiltonian)

T=10
M=100
dt=T/M

s=[]
A=[]
y=[]

now  = datetime.datetime.now()
folder_name = now.strftime("RST_%m-%d_%H%M")

#region === Initialization of matrix A and error vector e ===
def init():
    print(n)
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

        # A[i][random.randint(0,n-1)]=1
        # A[i][random.randint(0,n-1)]=1
        
        b=0
        for j in range(n):
            b^=A[i][j]*s[j]
        e=1 if random.random()<eta else 0
        y.append(b^e)
    print("s:", s)
#endregion

#region === Run the annealing algorithm to get the ground state ===
def run():
    t=0
    global A, s, y
    print(s)

    circ = QuantumCircuit(n+1,n)
    for i in range(n):
        circ.h(i)

    for i in range(M):
        t=(i+0.5)*dt
        alpha=(1-t/T)
        beta=t/T

        angle=2.0*beta*dt

        for k in range(m):
            for j in range(n):
                if A[k][j]==1:
                    circ.cx(j,n)
            if y[k]==1:
                circ.x(n)

            circ.rz(angle,n)

            if y[k]==1:
                circ.x(n)
            for j in range(n-1,-1,-1):
                if A[k][j]==1:
                    circ.cx(j,n)

        for j in range(n):
            circ.rx(2.0*alpha*dt,j)

    circ.measure(range(n),range(n))

    simulator=AerSimulator()
    compiled_circuit=transpile(circ,simulator)
    job=simulator.run(compiled_circuit,shots=100)
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
    print("合并后的数据：", merged_data)

    plt.figure(figsize=(10, 6))
    plt.bar(merged_data.keys(), merged_data.values())
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    plt.savefig(f"{folder_name}/eta={eta}.png", dpi=300, bbox_inches='tight')
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

#region === Generation of Hamiltonian ===
# init()
# H_P = construct_problem_hamiltonian(n, m, A, y)
# print(A)
# print(y)
# print(H_D)
# print(H_P)
# eigval_H_D = np.sort(np.linalg.eigvalsh(H_D))
# eigval_H_P = np.sort(np.linalg.eigvalsh(H_P))
# print(eigval_H_D)
# print(eigval_H_P)
# print("First energy of H_D:", eigval_H_D[0])
# print("First energy of H_P:", eigval_H_P[0])
# print("Second energy of H_D:", eigval_H_D[1])
# print("Second energy of H_P:", eigval_H_P[1])
# endregion

#region === Calculate energy gap in each step ===
H_D = H_P = 0
gap = []
def construct_ham(i):
    t=(i+0.5)*dt
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
    # if(n == n_max):
        # plt.savefig(f"{folder_name}/energy_gap.png", dpi=300, bbox_inches='tight')
#endregion
plt.savefig(f"{folder_name}/energy_gap.png", dpi=300, bbox_inches='tight')
print(gap_min)
# plt.savefig(f"{folder_name}/energy_gap.png", dpi=300, bbox_inches='tight')


rounds = 6
# for i in range(rounds):
#     eta = 0.05*(i+1)
#     init()
#     print(s)
#     run()

# calculate_energy_gap()