import time
import numpy as np
from matplotlib import pyplot as plt

#计时
start = time.perf_counter()

#参数声明
n = 10 + 1 #10等分，11个节点，n=11
dx =  1 / (n - 1) #dx长0.1
dy =  1 / (n - 1) #dy长0.1
qw = 1.0 #左边界单位时间、单位面积流入热流量
qe = 1.0 #右边界单位时间、单位面积流入热流量
dt = 0.01 #时间步长
t_max = 100 #最大时间，可修改
t_total = 0 #记录当前已计算完毕的已知时层
epsilon_iteration = 1e-4 #书p112要求迭代收敛时的相对变化量在1e-6到1e-3之间
epsilon_time = 1e-4 #达到稳态的收敛判断条件
delta_x = dx #均匀分割网络，小delta x = dx
delta_y = dy #均匀分割网络，小delta y = dy

#变量声明
T = np.zeros((n) * (n)).reshape((n, n)) #定义11 * 11的全0温度数组
T_iteration = np.copy(T) #定义一个“篮子”，临时存放当前迭代计算好的结果
T_now = np.copy(T) #定义一个“篮子”，临时存放当前时层计算好的结果
total_iteration = 1 #总迭代次数

#根据初始条件，对0时刻的所有节点进行赋值
for i in range(0, n):
    T[i] = -0.1 * i + 1.0

#作初始状态温度图
plt.rcParams['font.sans-serif']=['FangSong']
fig=plt.figure()
plt.xlabel('X方向')
plt.ylabel('Y方向')
picture = plt.contourf(np.linspace(0, 3, n),np.linspace(0, 3, n),3*T,n,cmap=plt.cm.hot) 
plt.title("初始时刻")
plt.colorbar()

#函数：对隐式通用离散格式进行G-S迭代的改造，求解未知迭代层的温度
def general_discrete_format(a_p0, a_e, a_w, a_n, a_s, T_e, T_w, T_n, T_s, T_p0, Q):
    a_p = a_p0 + a_e + a_n + a_w + a_s
    T_p = (a_e * T_e + a_w * T_w + a_s * T_s + a_n * T_n + a_p0 * T_p0 + Q) / a_p
    return float(T_p)

#函数：求解n * n个点的温度演化情况
def calculate_T(T, dx, dy, qw, qe, dt, delta_x, delta_y, n):
    T_new = np.copy(T) #定义一个“篮子”，临时存放计算好的结果

    for j in range(n):
        T_new[0][j] = 1.0 #下边界，i=0, j=0到10，温度为1
        T_new[n - 1][j] = 0.0 #上边界，i=10, j=0到10，温度为0

    for i in range(1, n-1):
        T_new[i][0] = general_discrete_format(a_p0=dx*dy/(2*dt), a_e=dy/dx, a_w=0.0, a_n=dx/(2*dy), a_s=dx/(2*dy), \
            T_e=T[i][1], T_w=0.0, T_n=T[i+1][0], T_s=T_new[i-1][0], T_p0=T[i][0], Q=qw*dy) #左边界，i=1到9, j=0
    
    for i in range(1, n-1):
        T_new[i][n-1] = general_discrete_format(a_p0=dx*dy/(2*dt), a_e=0.0, a_w=dy/dx, a_n=dx/(2*dy), a_s=dx/(2*dy), \
            T_e=0.0, T_w=T_new[i][n-2], T_n=T[i+1][n-1], T_s=T_new[i-1][n-1], T_p0=T[i][n-1], Q=qe*dy) #右边界，i=1到9, j=10

    for i in range(1, n-1):
        for j in range(1, n-1):
            T_new[i][j] = general_discrete_format(a_p0=dx*dy/dt, a_e=dy/dx, a_w=dy/dx, a_n=dx/dy, a_s=dx/dy, \
                T_e=T[i][j+1], T_w=T_new[i][j-1], T_n=T[i+1][j], T_s=T_new[i-1][j], T_p0=T[i][j], Q=0.0) #非边界点, i=2到9, j=2到9

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
    T_iteration = np.copy(calculate_T(T, dx, dy, qw, qe, dt, delta_x, delta_y, n))
    while( compare_restrain(T, T_iteration, epsilon_iteration, n) == 1 ): #判断是否达到迭代收敛
        T = np.copy(T_iteration) #把已知迭代结果存储
        T_iteration = np.copy(calculate_T(T, dx, dy, qw, qe, dt, delta_x, delta_y, n))
        total_iteration += 1 #迭代总次数增加1
        '''
        #作随迭代步前进时的状态温度图
        if (total_iteration % 32) == 0:
            plt.rcParams['font.sans-serif']=['FangSong']
            fig=plt.figure()
            plt.xlabel('X方向')
            plt.ylabel('Y方向')
            picture = plt.contourf(np.linspace(0, 3, n),np.linspace(0, 3, n),np.around(T_iteration * 3, decimals=3),n, cmap=plt.cm.hot)
            plt.title("迭代步第{0:}步".format(total_iteration))
            plt.colorbar()
            plt.show()
            # print(T_iteration)
        '''
    t_total += dt #时间前进
    T_now = np.copy(T_iteration) #把已知收敛的迭代结果作为已知时层的结果
    


    #判断是否达到稳态
    if compare_restrain(T, T_now, epsilon_time, n) == 0: #判断是否达到稳态
        #计时
        dur = time.perf_counter() - start
        #输出
        print('{0:*^90}'.format('Gauss-Seidel点迭代迭代平衡时间及迭代次数如下'))
        print('传热达到平衡，程序耗时为{0:} sec'.format(dur), end='\n'*2)
        print('总迭代次数为{0:}次'.format(total_iteration), end='\n'*2)
        print('{0:*^88}'.format('平衡时的温度分布如下'))
        for i in range(n-1, -1, -1):#由于ndarray格式直接输出是下标小的数在前，所以需要把数组i方向下标倒序输出以更直观
            for j in range(n):
                print('{0: ^8}'.format(round(T_now[i][j], 5)), end=' ')
            print('\n')
        break
    else:
        T = np.copy(T_now)

    #计时初始化
    start = time.perf_counter()

#作稳定状态温度图
plt.rcParams['font.sans-serif']=['FangSong']
fig=plt.figure()
plt.xlabel('X方向')
plt.ylabel('Y方向')
picture = plt.contourf(np.linspace(0, 1, n),np.linspace(0, 1, n),T_now,n,cmap=plt.cm.hot) 
plt.title("Gauss-Seidel点迭代稳态温度分布图")
plt.colorbar()


plt.show()



