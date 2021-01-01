import numpy as np
import time
from matplotlib import pyplot as plt


#计时
start = time.perf_counter()

#参数声明
n = 10 + 1 #10等分，11个节点，n=11
dx =  1 / (n - 1) #dx长0.1
dy =  1 / (n - 1) #dy长0.1
qw = 1.0 #左边界单位时间、单位面积流入热流量
qe = 1.0 #右边界单位时间、单位面积流入热流量
dt = 0.001 #时间步长，稳定性要求其小于2.5e-3
t_max = 100 #最大时间，可修改
t_total = 0 #记录当前已计算完毕的已知时层
epsilon = 1e-4 #达到稳态的收敛判断条件
delta_x = dx #均匀分割网络，小delta x = dx
delta_y = dy #均匀分割网络，小delta y = dy
total_iteration = 1 #总迭代次数

#变量声明
T = np.zeros((n) * (n)).reshape((n, n)) #定义11 * 11的全0温度数组
T_now = np.copy(T) #定义一个“篮子”，临时存放当前时层刚刚计算好的结果

#根据初始条件，对0时刻的所有节点进行赋值
for i in range(0, n):
    T[i] = -0.1 * i + 1.0

#函数：根据显式通用离散格式直接求解未知时层的温度
def general_discrete_format(a_p, a_e, a_w, a_n, a_s, T_e, T_w, T_n, T_s, T_p0, Q):
    a_p0 = a_p - a_e - a_n - a_w - a_s
    T_p = (a_e * T_e + a_w * T_w + a_s * T_s + a_n * T_n + a_p0 * T_p0 + Q) / a_p
    return float(T_p)

#函数：求解n * n个点的温度演化情况
def calculate_T(T, dx, dy, qw, qe, dt, delta_x, delta_y, n):
    T_new = np.copy(T) #定义一个“篮子”，临时存放计算好的结果
    
    for j in range(n):
        T_new[0][j] = 1.0 #下边界，i=0, j=0到10，温度为1
        T_new[n - 1][j] = 0.0 #上边界，i=10, j=0到10，温度为0
    
    for i in range(1, n-1):
        T_new[i][0] = general_discrete_format(a_p=dx*dy/(2*dt), a_e=dy/dx, a_w=0.0, a_n=dx/(2*dy), a_s=dx/(2*dy), \
            T_e=T[i][1], T_w=0.0, T_n=T[i+1][0], T_s=T[i-1][0], T_p0=T[i][0], Q=qw*dy) #左边界，i=1到9, j=0

    for i in range(1, n-1):
        T_new[i][n-1] = general_discrete_format(a_p=dx*dy/(2*dt), a_e=0.0, a_w=dy/dx, a_n=dx/(2*dy), a_s=dx/(2*dy), \
            T_e=0.0, T_w=T[i][n-2], T_n=T[i+1][n-1], T_s=T[i-1][n-1], T_p0=T[i][n-1], Q=qe*dy) #右边界，i=1到9, j=10
    
    for i in range(1, n-1):
        for j in range(1, n-1):
            T_new[i][j] = general_discrete_format(a_p=dx*dy/dt, a_e=dy/dx, a_w=dy/dx, a_n=dx/dy, a_s=dx/dy, \
                T_e=T[i][j+1], T_w=T[i][j-1], T_n=T[i+1][j], T_s=T[i-1][j], T_p0=T[i][j], Q=0.0) #非边界点, i=2到9, j=2到9

    return T_new

#函数：判断结果是否收敛
def compare_restrain(T, T_now, epsilon, n):
    flag = 0 #收敛则flag保持0，不收敛则flag变为1
    for i in range(0, n-1):
        for j in range(0, n-1):
            if T[i][j] == 0:
                continue
            if (abs(T[i][j] - T_now[i][j]) / T[i][j]) >= epsilon:
                flag = 1
    return flag

#调用上述函数
while (t_total <= t_max):
    T_now = np.copy(calculate_T(T, dx, dy, qw, qe, dt, delta_x, delta_y, n))
    t_total += dt #时间前进
    total_iteration += 1
    if compare_restrain(T, T_now, epsilon, n) == 0: #判断是否达到稳态
        #计时
        dur = time.perf_counter() - start
        #输出
        print('{0:*^90}'.format('平衡时间如下'))
        print('无量纲时间为{0:}时，传热达到平衡，耗时为{1:} sec'.format(t_total, dur), end='\n'*2)
        print('总迭代次数为{0:}次'.format(total_iteration), end='\n'*2)
        print('{0:*^88}'.format('平衡时的温度分布如下'))
        for i in range(n-1, -1, -1):#由于ndarray格式直接输出是下标小的数在前，所以需要把数组i方向下标倒序输出以更直观
            for j in range(n):
                print('{0: ^8}'.format(round(T_now[i][j], 5)), end=' ')
            print('\n')
        break
    else:
        T = np.copy(T_now)


#作稳定状态温度图
plt.rcParams['font.sans-serif']=['FangSong']
fig=plt.figure()
plt.xlabel('X方向')
plt.ylabel('Y方向')
picture = plt.contourf(np.linspace(0, 1, n),np.linspace(0, 1, n),T_now,n,cmap=plt.cm.hot) 
plt.title("控制容积法显式格式稳态无量纲温度分布图")
plt.colorbar()


plt.show()