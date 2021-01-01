import time
import numpy as np
from matplotlib import pyplot as plt

#计时
start = time.perf_counter()

#变量声明
n = 10 + 1 #10等分，11个节点，n=11
T = np.zeros((n) * (n)).reshape((n, n)) #定义11 * 11的全0温度数组

#根据初始条件，对0时刻的所有节点进行赋值
for i in range(0, n):
    T[i] = -0.1 * i + 1.0
T_pass = np.copy(T) #存储上个时刻温度分布
T_now = np.copy(T) #存储当前时刻温度分布
T_initial = np.copy(T) #存储初始时刻温度分布
T_iteration_pass = np.copy(T) #定义一个“篮子”，临时存放上个迭代步计算好的结果
T_iteration_end = np.copy(T) #定义一个“篮子”，临时存放当前完整1个迭代步计算好的结果

#参数声明
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
total_iteration = 0 #总迭代次数初始化
w = 0.0 #松弛因子初始化
judge = False #辅助循环的代码段

class WE(): #东西向
    def __init__(self):
        #a
        self.a = np.zeros(n-2)
        self.a[0] = 0
        self.a[1:] = -1
        #b
        self.b = np.full(n-2, fill_value=4)
        #c
        self.c = np.zeros(n-2)
        self.c[:-1] = -1
        self.c[-1] = 0
class NS(): #南北向
    def __init__(self):
        #a
        self.a = np.zeros(n)
        self.a[0] = 0
        self.a[1:-1] = -1
        self.a[-1] = -2
        #b
        self.b = np.full(n, fill_value=4)
        #c
        self.c = np.zeros(n)
        self.c[0] = -2
        self.c[1:-1] = -1
        self.c[-1] = 0
        
parameter_WE = WE()
parameter_NS = NS()

#函数：TDMA的P求解
def calculate_P(p, c, a, b, count):
    if count >= 1:
        return - c[count] / ( a[count] * calculate_P( p, c, a, b, count - 1 ) + b[count] )
    else:
        return - c[count] / b[count]

#函数：TDMA的Q求解
def calculate_Q(a, b, p, d, count):
    if count >= 1:
        return ( ( d[count] - a[count] * calculate_Q(a, b, p, d, count - 1) ) / ( a[count] * p[count - 1] + b[count] ) )
    else:
        return d[count] / b[count]

#函数：TDMA主函数
def TDMA_main(a, b, c, d, n):
    p = np.zeros(n) #定义1 * 11的全0系数矩阵
    q = np.zeros(n) #定义1 * 11的全0系数矩阵
    T = np.zeros(n) #定义1 * 11的全0温度矩阵
    for i in range(n-1, -1, -1):
        p[i] = calculate_P(p, c, a, b, i) #迭代求解p
    for i in range(n-1, -1, -1):
        q[i] = calculate_Q(a, b, p, d, i) #迭代求解q
    T[-1] = q[-1]
    for i in range(n-1, 0, -1):
        T[i-1] = T[i] * p[i-1] + q[i-1] #回代求解T
    return T

#函数：判断结果是否收敛
def compare_restrain(T, T_now, epsilon, n):
    flag = 0 #收敛则flag保持0，不收敛则flag变为1
    for i in range(0, n):
        for j in range(0, n):
            if T[i][j] == 0:
                continue
            if (abs(T[i][j] - T_now[i][j]) / T[i][j]) >= epsilon:
                flag = 1
    return flag

#函数：松弛因子加权
def calculate_final_T(T, T_pass, w):
    T_final = np.copy ( (1 - w) * np.copy(T_pass) + w * np.copy(T) )
    return T_final

#调用以上函数
while (w <= 2.0): #w查找范围为0.3到2.0，间隔0.1计算一次

    total_iteration = 1 #总迭代次数初始化
    w += 0.1
    T = np.copy(T_initial)
    T_iteration_pass = np.copy(T_initial) #上个迭代步结果用初始时刻温度进行初始化
    T_pass = np.copy(T_initial) #上个时间步结果用初始时刻温度进行初始化
    T_iteration_end = np.zeros((n) * (n)).reshape((n, n)) #定义11 * 11的全0温度数组

    while (t_total <= t_max):
        t_total += dt #时间前进

        #判断是否达到迭代收敛
        while( compare_restrain(T_iteration_pass, T_iteration_end, epsilon_iteration, n) == 1 ): 
            #如果不是从时刻步开始的迭代，那进行迭代步结果集体往后移动一步
            if judge == True:
                T_iteration_pass = np.copy(T_iteration_end)
            
            #前0.5个迭代步, 从南向北扫描, 求出的T温度是一行一行的
            #d
            parameter_NS.d_boundary = np.zeros(n)
            for i in range(1, 10):
                parameter_NS.d_boundary[0] = T[i+1][0] + T[i-1][0] + 0.2
                for j in range(1, 10):
                    parameter_NS.d_boundary[j] =  T[i+1][j] + T[i-1][j]
                parameter_NS.d_boundary[10] = T[i+1][10] + T[i-1][10] + 0.2
                T[i, ::] = np.copy(TDMA_main(parameter_NS.a, parameter_NS.b, parameter_NS.c, parameter_NS.d_boundary, n))

            #后0.5个迭代步，从西向东扫描, 求出的T温度是一列一列的
            #d
            #左边界d
            parameter_WE.d_W_boundary = np.zeros(n-2)
            parameter_WE.d_W_boundary[0] = np.copy( 2 * T[1][1] + 1.2 )
            for i in range(1, 9):
                parameter_WE.d_W_boundary[i] = 2 * T[i+1][1] + 0.2
            #左边界中段的点
            T[1:10, 0] = np.copy(TDMA_main(parameter_WE.a, parameter_WE.b, parameter_WE.c, parameter_WE.d_W_boundary, n-2))
            #中间d
            parameter_WE.d_inside = np.zeros(n-2)
            for j in range(1, 10):
                parameter_WE.d_inside[0] = T[1][j+1] + T[1][j-1] + 1.0
                for i in range(1, 9):
                    parameter_WE.d_inside[i] = T[i+1][j+1] + T[i+1][j-1]
                #中间的点
                T[1:10, j] = np.copy(TDMA_main(parameter_WE.a, parameter_WE.b, parameter_WE.c, parameter_WE.d_inside, n-2))
            #右边界d
            parameter_WE.d_E_boundary = np.zeros(n-2)
            parameter_WE.d_E_boundary[0] = np.copy( 2 * T[1][9] + 1.2 )
            for i in range(1, 9):
                parameter_WE.d_E_boundary[i] = 2 * T[i+1][9] + 0.2
            #右边界中段的点
            T[1:10, 10] = np.copy(TDMA_main(parameter_WE.a, parameter_WE.b, parameter_WE.c, parameter_WE.d_E_boundary, n-2))

            #松弛因子将当前迭代步原始结果和上个迭代步结果进行加权
            T_iteration_end = np.copy( calculate_final_T(T, T_iteration_pass, w) )

            #迭代步后移标志
            judge = True

            #迭代总次数增加1
            total_iteration += 1 


        T_now = np.copy(T_iteration_end) #把已知收敛的迭代结果作为已知时层的结果

        #判断是否达到稳态
        if compare_restrain(T_pass, T_now, epsilon_time, n) == 0: #判断是否达到稳态
            if np.isnan(T_now).any() == False: #判断数组是否出现nan，不出现才可以输出，出现nan的原因是因为判断相对变化时出现分母为0的情况
                #计时
                dur = time.perf_counter() - start
                #输出
                print('{0:*^90}'.format('基于带松弛的GS的交替方向隐式迭代平衡时间及迭代次数如下'))
                print('传热达到平衡，程序耗时为{0:} sec'.format(dur), end='\n'*2)
                print('总迭代次数为{0:}次'.format(total_iteration), end='\n'*2)
                print('此时w为{0:}'.format(round(w, 2)), end='\n'*2)
                print('{0:*^88}'.format('平衡时的温度分布如下'))
                for i in range(n-1, -1, -1):#由于ndarray格式直接输出是下标小的数在前，所以需要把数组i方向下标倒序输出以更直观
                    for j in range(n):
                        print('{0: ^8}'.format(round(T_now[i][j], 5)), end=' ')
                    print('\n')
                #作w=1.5时的稳定状态温度图
                if round(w, 2) == 1.5:
                    plt.rcParams['font.sans-serif']=['FangSong']
                    fig=plt.figure()
                    plt.xlabel('X方向')
                    plt.ylabel('Y方向')
                    picture = plt.contourf(np.linspace(0, 1, n),np.linspace(0, 1, n),T_now,n,cmap=plt.cm.hot) 
                    plt.title("基于Gauss-Seidel的交替方向线迭代稳态温度分布图")
                    plt.colorbar()
                    plt.show()
                break
        else:
            #时间步结果集体后移
            T_pass = np.copy(T_now)
            judge = False
            #迭代步结果集体后移
            T_iteration_pass = np.copy(T_now)
            #当前迭代步结果初始化
            T_iteration_end = np.zeros((n) * (n)).reshape((n, n))

    #计时初始化
    start = time.perf_counter()





