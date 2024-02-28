#RuntimeWarning: invalid value encountered in scalar divide: o[i] = (maximum - fit[i]) / (maximum - minimum)
import numpy as np
import random
from losses.jFitnessFunction import jFitnessFunction
class jDingoOptimization:
    def __init__(self, N, max_Iter, loss_func, alpha=0.9, beta=0.1, thres=0.5, tau=1, rho=0.2, eta=1):
        self.N = N
        self.max_Iter = max_Iter
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.thres = thres
        self.tau = tau
        self.rho = rho
        self.eta = eta

    def optimize(self, x_train, x_test, y_train, y_test):

        dim = x_train.shape[1] # Dimension
        lb = 0 # Lower Bound
        ub = 1 # Upper Bound
        P = 0.5 # Hunting or Scavenger?
        FADs = 0.2 # Fish aggregating devices effect
        N = self.N
        # Get opts fields

        P = 0.5  # Hunting or Scavenger?  rate.  See section 3.0.4, P and Q parameters analysis
        Q = 0.7  # Group attack or persecution?
        beta1 = -2 + 4 * np.random.rand()  # -2 < beta < 2     Used in Eq. 2,
        beta2 = -1 + 2 * np.random.rand()  # -1 < beta2 < 1    Used in Eq. 2,3, and 4
        naIni = 2  # Minimum number of dingoes that will attack
        naEnd = N // naIni  # Maximum number of dingoes that will attack
        na = np.round(naIni + (
                    naEnd - naIni) * np.random.rand())  # Number of dingoes that will attack, used in Attack.m Section 2.2.1: Group attack
        X = np.random.rand(N, dim) * (ub - lb) + lb
        curve = np.zeros(self.max_Iter + 1)

        Fitness = np.zeros(N)
        for i in range(N):
            Fitness[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
        minIdx = np.argmin(Fitness)
        fitG = Fitness[minIdx]
        Xgb = X[minIdx, :]
        maxIdx = np.argmax(Fitness)
        vMax = Fitness[maxIdx]

        curve[0] = fitG
        survival = self._survival_rate(Fitness, fitG, vMax)  # Section 2.2.4 Dingoes'survival rates
        t = 0  # Loop counter
        # Main loop

        v = np.zeros((N,dim))

        for t in range(self.max_Iter):
            for r in range(N):
                sumatory = 0
                if random.random() < P:
                    sumatory = self._attack(N, na, X, r)
                    if random.random() < Q:
                        v[r, :] = beta1 * sumatory - Xgb
                    else:
                        r1 = int(np.round((N - 1) * random.random()))
                        v[r, :] = Xgb + beta1 * np.exp(beta2) * (X[r1, :] - X[r, :])
                else:
                    r1 = int(np.round((N - 1) * random.random()))
                    v[r, :] = np.exp(beta2) * X[r1, :] - ((-1) ** self._getBinary()) * X[r, :] / 2
                if survival[r] <= 0.3:
                    band = 1
                    while band:
                        r1 = int(np.round((N - 1) * random.random()))
                        r2 = int(np.round((N - 1) * random.random()))
                        if r1 != r2:
                            band = 0
                    v[r, :] = Xgb + (X[r1, :] - ((-1) ** self._getBinary()) * X[r2, :]) / 2
                Flag4ub = v[r, :] > ub
                Flag4lb = v[r, :] < lb
                v[r, :] = (v[r, :] * ~(Flag4ub + Flag4lb)) + ub * Flag4ub + lb * Flag4lb
                Fnew = self.loss_func(x_train[:, v[r, :] > self.thres], x_test[:, v[r, :] > self.thres],
                               y_train, y_test)
                if Fnew <= Fitness[r]:
                    X[r, :] = v[r, :]
                    Fitness[r] = Fnew
                if Fnew <= fitG:
                    Xgb = v[r, :]
                    fitG = Fnew
            curve[t + 1] = fitG
            print(f'\nIteration {t + 1} Best (DOA)= {fitG}')
            vMax, maxIdx = np.max(Fitness), np.argmax(Fitness)
            survival = self._survival_rate(Fitness, fitG, vMax)

        Pos = np.arange(1, dim + 1)
        Sf = Pos[(Xgb > self.thres) == 1] - 1

        # 主循环结束
        DOA = {}
        DOA['sf'] = Sf  # best_X
        DOA['c'] = curve
        return DOA

    import numpy as np
    @staticmethod
    def _survival_rate(fit, minimum, maximum):
        np.seterr(divide='ignore', invalid='ignore')
        o = np.zeros(fit.shape)
        for i in range(fit.shape[0]):
            o[i] = (maximum - fit[i]) / (maximum - minimum)
            #可能的报错原因：
            # 在代码的其他地方，maximum、minimum或fit[i] 的值被错误地设置为无效的值。
            # 在进行计算之前，这些变量没有被正确地初始化。
        return o

    @staticmethod
    def _attack(SearchAgents_no, na, Positions, r):
        sumatory = 0
        vAttack = VectorAttack(SearchAgents_no, na)
        for j in range(vAttack.shape[0]):
            sumatory += Positions[vAttack[j], :] - Positions[r, :]
        sumatory /= na
        return sumatory


    @staticmethod
    def _getBinary():
        return int(np.random.rand() < 0.5)


def VectorAttack(SearchAgents_no, na):
    c = 0
    vAttack = []
    while c < na:
        idx = np.round((SearchAgents_no - 1) * np.random.rand()).astype(int)
        if idx not in vAttack:
            vAttack.append(idx)
            c += 1
    return np.array(vAttack)


    @staticmethod
    def _findrep(val, vector):
        return val in vector
