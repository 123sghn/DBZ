import numpy as np

from losses.jFitnessFunction import jFitnessFunction

class jGeneticAlgorithm:
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

        # Parameters
        CR = 0.8    # crossover rate
        MR = 0.01   # mutation rate
        N = self.N


        # Number of dimensions
        dim = x_train.shape[1]
        # Initial
        X = self._initialization(N, dim)
        # Fitness
        fit = np.zeros(N)
        fitG = np.inf
        for i in range(N):
            fit[i] = self.loss_func(x_train[:, X[i, :] > self.thres], x_test[:, X[i, :] > self.thres],
                                        y_train, y_test)
            # Best update
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        # Pre
        curve = np.zeros(self.max_Iter)
        curve[0] = fitG
        t = 2
        # Generations
        while t <= self.max_Iter:
            # Get probability
            Ifit = 1 / (1 + fit)
            prob = Ifit / np.sum(Ifit)
            # Preparation
            Xc1 = np.zeros((N, dim))
            Xc2 = np.zeros((N, dim))
            fitC1 = np.ones(N)
            fitC2 = np.ones(N)
            z = 0
            for i in range(N):
                if np.random.rand() < CR:
                    # Select two parents
                    k1 = self._roulette_wheel_selection(prob)
                    k2 = self._roulette_wheel_selection(prob)
                    # Store parents
                    P1 = X[k1, :]
                    P2 = X[k2, :]
                    # Single point crossover
                    ind = np.random.randint(1, dim)
                    # Crossover between two parents

                    Xc1[z, :] = np.concatenate((P1[0:ind + 1], P2[ind + 1:dim]))
                    Xc2[z, :] = np.concatenate((P2[0:ind + 1], P1[ind + 1:dim]))
                    # Mutation
                    for d in range(dim):
                        # First child
                        if np.random.rand() < MR:
                            Xc1[z, d] = 1 - Xc1[z, d]
                        # Second child
                        if np.random.rand() < MR:
                            Xc2[z, d] = 1 - Xc2[z, d]
                    # Fitness
                    fitC1[z] = self.loss_func(x_train[:, Xc1[i, :] > self.thres], x_test[:, Xc1[i, :] > self.thres],
                                        y_train, y_test)
                    fitC2[z] = self.loss_func(x_train[:, Xc2[i, :] > self.thres], x_test[:, Xc2[i, :] > self.thres],
                                        y_train, y_test)
                    z = z + 1

            # Merge population
            XX = np.vstack((X, Xc1, Xc2))
            FF = np.concatenate((fit, fitC1, fitC2))
            # Select N best solution
            idx = np.argsort(FF)
            X = XX[idx[0:N], :]
            fit = FF[0:N]
            # Best agent
            if fit[0] < fitG:
                fitG = fit[0]
                Xgb = X[0, :]

            # Save
            curve[t - 1] = fitG
            print(f'\nGeneration {t} Best (GA)= {curve[t - 1]}')
            t = t + 1

        # Select features based on selected index
        Pos = np.arange(dim)
        Sf = Pos[Xgb == 1]
        # Store results
        GA = {
            'sf': Sf,
            'c': curve,
        }
        return GA

    @staticmethod
    def _roulette_wheel_selection(prob):
        # Cummulative summation
        C = np.cumsum(prob)
        # Random one value, most probability value [0~1]
        P = np.random.rand()
        # Route wheel
        for i in range(len(C)):
            if C[i] > P:
                Index = i
                break
        return Index

    @staticmethod
    def _initialization(N, dim):
        # Initialize X vectors
        X = np.zeros((N, dim))
        for i in range(N):
            for d in range(dim):
                if np.random.rand() > 0.5:
                    X[i, d] = 1
        return X
