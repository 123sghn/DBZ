import numpy as np
from scipy.special import gamma

from losses.jFitnessFunction import jFitnessFunction

class jMarinePredatorsAlgorithm:
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

        # 参数
        lb = 0
        ub = 1
        P = 0.5   # 常数
        FADs = 0.2   # 鱼类聚集设备效应
        N = self.N

        # 目标函数
        dim = x_train.shape[1]   # 维度个数
        X = np.zeros((N, dim))   # 初始个体位置
        for i in range(N):
            for d in range(dim):
                X[i, d] = lb + (ub - lb) * np.random.rand()

        # 预处理
        fit = np.zeros(N)
        fitG = np.inf
        fitM = np.zeros(N)
        curve = np.zeros(self.max_Iter)
        t = 0   # 迭代次数
        Xmb = np.copy(X)

        while t < self.max_Iter:
            # 计算适应度值
            for i in range(N):
                fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
                # 更新最佳值
                if fit[i] < fitG:
                    fitG = fit[i]
                    Xgb = X[i, :]

            # 存储最佳值和个体位置
            if t == 0:
                fitM = fit
                Xmb = X

            for i in range(N):
                if fitM[i] < fit[i]:
                    fit[i] = fitM[i]
                    X[i, :] = Xmb[i, :]

            Xmb = X
            fitM = fit

            # 构建精英个体
            Xe = np.tile(Xgb, (N, 1))

            # 自适应参数
            CF = (1 - (t / self.max_Iter)) ** (2 * (t / self.max_Iter))

            # [第一阶段]
            if t <= self.max_Iter / 3:
                for i in range(N):
                    RB = np.random.randn(dim)   # 布朗随机数

                    for d in range(dim):
                        R = np.random.rand()

                        stepsize = RB[d] * (Xe[i][d] - RB[d] * X[i][d])

                        X[i, d] = X[i][d] + P * R * stepsize

                    # 边界处理
                    X[i, :] = np.clip(X[i, :], lb, ub)

            # [第二阶段]
            elif t > self.max_Iter / 3 and t <= 2 * self.max_Iter / 3:
                for i in range(N):
                    if i <= N / 2:
                        RL = 0.05 * self._Levy(self.beta, dim)   # levy随机数

                        for d in range(dim):
                            R = np.random.rand()
                            stepsize = RL[d] * (Xe[i][d] - RL[d] * X[i][d])

                            #X[i, d] = X[i, d] + P * R * stepsize
                            X[i][d] = X[i][d] + P * R * stepsize

                    else:
                        RB = np.random.randn(dim)   # 布朗随机数

                        for d in range(dim):
                            stepsize = RB[d] * (RB[d] * Xe[i, d] - X[i, d])
                            X[i, d] = Xe[i, d] + P * CF * stepsize

                    X[i, :] = np.clip(X[i, :], lb, ub)

            # [第三阶段]
            elif t > 2 * self.max_Iter / 3:
                for i in range(N):
                    RL = 0.05 * self._Levy(self.beta, dim)   # levy随机数

                    for d in range(dim):
                        stepsize = RL[d] * (RL[d] * Xe[i, d] - X[i, d])
                        X[i, d] = Xe[i, d] + P * CF * stepsize

                    X[i, :] = np.clip(X[i, :], lb, ub)

            # 更新适应度值
            for i in range(N):
                fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
                # 更新最佳值
                if fit[i] < fitG:
                    fitG = fit[i]
                    Xgb = X[i, :]

            # 存储最佳值和个体位置
            for i in range(N):
                if fitM[i] < fit[i]:
                    fit[i] = fitM[i]
                    X[i, :] = Xmb[i, :]

            Xmb = X
            fitM = fit

            # 效应形成和鱼类聚集设备效应
            if np.random.rand() <= FADs:
                for i in range(N):
                    U = np.random.rand(dim) < FADs   # 计算U

                    for d in range(dim):
                        R = np.random.rand()
                        X[i, d] = X[i, d] + CF * (lb + R * (ub - lb)) * U[d]

                    X[i, :] = np.clip(X[i, :], lb, ub)

            else:
                r = np.random.rand()   # 在[0,1]上生成随机数
                Xr1 = X[np.random.permutation(N), :]   # 随机选择两个个体作为猎物
                Xr2 = X[np.random.permutation(N), :]

                for i in range(N):
                    for d in range(dim):
                        X[i, d] = X[i, d] + (FADs * (1 - r) + r) * (Xr1[i, d] - Xr2[i, d])

                    X[i, :] = np.clip(X[i, :], lb, ub)

            curve[t] = fitG
            print('\n迭代次数 {} 最佳值 (MPA)= {}'.format(t, curve[t]))
            t = t + 1

        # 根据索引选择特征
        Pos = np.arange(dim)
        Sf = Pos[(Xgb > self.thres) == 1]

        # 存储结果
        MPA = {}
        MPA['sf'] = Sf
        MPA['c'] = curve

        return MPA

    @staticmethod
    # Levy分布
    def _Levy(beta, dim):
        num = gamma(1 + beta) * np.sin(np.pi * beta / 2)
        deno = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
        sigma = (num / deno) ** (1 / beta)
        u = np.random.normal(0, sigma, (1, dim))
        v = np.random.normal(0, 1, (1, dim))
        LF = u / (np.abs(v) ** (1 / beta))

        return LF[0]
