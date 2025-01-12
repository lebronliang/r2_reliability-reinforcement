from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
import numpy as np
from pymoo.factory import get_termination
import matplotlib.pyplot as plt



def system_reliability_c(system_reliability):
    final_list = []
    for i in range(0, len(system_reliability)):
        reliability_1 = system_reliability[i][0]
        reliability_2 = system_reliability[i][1]
        reliability_3 = system_reliability[i][2]
        reliability_4 = system_reliability[i][3]
        # final = reliability_1 * reliability_2 * reliability_3 * reliability_4
        final = (reliability_1 * 30000 + (1-reliability_1) * 12400 + reliability_2 * 30000 + (1-reliability_2) * 12480 + 30000 * reliability_3 + (1-reliability_3) * 21550 + reliability_4 * 30000 + (1-reliability_4) * 24960)/120000
        final_list.append(final)
    final_r = np.array(final_list)
    return final_r


def objective_function(r):
    target_list = []
    for i in range(0, len(r)):
        f = 5*np.log(1/(1+1e-8-r[i][0])) + 5*np.log(1/(1+1e-8-r[i][1])) + 5*np.log(1/(1+1e-8-r[i][2])) + 5*np.log(1/(1+1e-8-r[i][3]))
        target_list.append(f)
    final_target = np.array(target_list)
    obj = np.column_stack([final_target])
    return obj


def constraint_function(x):
    system_reliability = system_reliability_c(x)
    threshold_system = 0.96
    g1 = threshold_system - system_reliability
    constraints = np.column_stack([g1])
    return constraints


class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=4, n_obj=1, n_constr=1, xl=[0.88,0.88,0.88,0.88], xu=np.ones(4))

    def _evaluate(self,x,out, *args, **kwargs):
        f = objective_function(x)
        # print(f)
        g = constraint_function(x)
        # print(g)

        # print(g)
        out["F"] = f
        out["G"] = g



# for i in ....
#

problem = MyProblem()




algorithm = NSGA2(
        pop_size=60,  # 种群数量
        n_offsprings=100,  # 最大进化代数
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.7, eta=10),  # 交叉概率
        mutation=get_mutation("real_pm", prob=0.9, eta=10),  # 变异概率
        eliminate_duplicates=True
    )


results = minimize(problem, algorithm, get_termination("n_gen", 500),
                   verbose=1)

system_reliability = results.X


target = [i.F[0] for i in results.pop]
ind = [i for i in range(len(results.pop))]
# lowess = sm.nonparametric.lowess
# y_smooth = lowess(target,ind,frac=1./3.)[:,1]

plt.rcParams['font.sans-serif'] = ['Simhei'] #解决中文显示问题，目前只知道黑体可行
plt.rcParams['axes.unicode_minus'] = False #解决负数坐标显示问题
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.figure(dpi=200)
plt.xlabel(xlabel='evolutional generation', fontdict={'family':'Times New Roman','size':15})
plt.ylabel(ylabel="fitness", fontdict={'family':'Times New Roman','size':15})
plt.plot(ind, target)
plt.xticks(fontsize=12, fontproperties='Times New Roman')
plt.yticks(fontsize=12, fontproperties='Times New Roman')
plt.show()