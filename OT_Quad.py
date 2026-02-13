import numpy as np
import ot
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt


class KSC:
    def __init__(self, K, L, C, l, l_x, l_y, w=None, v=None):
        # K (n, n) symmetric
        # L (m, m) symmetric
        # C (n, m)
        # l - lambda for entropy
        # l_x, l_y - lambdas for Laplacian terms
        # w (n,)
        # v (m,)

        self.K = K
        self.L = L
        self.C = C
        self.l = l
        self.l_x = l_x
        self.l_y = l_y

        self.n, self.m = np.shape(C)

        if w is None:
            self.w = np.ones(self.n) / self.n
        else:
            self.w = w

        if v is None:
            self.v = np.ones(self.m) / self.m
        else:
            self.v = v

        self.Kinf = np.max(np.abs(K))    # L_infinity norm of K
        self.Linf = np.max(np.abs(L))    # L_infinity norm of L

    # quadratic function g = <pi, K pi> + <L pi, pi> + <C, pi>
    def g(self, pi):
        # return np.sum(np.multiply(pi, self.K@pi)) + np.sum(np.multiply(self.L@pi, pi)) + np.sum(np.multiply(self.C, pi))

        #My edit: implementig  g = <pi, K pi> + <pi, pi L> + <C, pi>
        return self.l_x*np.sum(np.multiply(pi, self.K@pi)) + self.l_y*np.sum(np.multiply(pi, pi@self.L)) + np.sum(np.multiply(self.C, pi))

    # gradient of g = 2 K pi + 2 L pi + C
    def g_grad(self, pi):
        # return (2*self.K@pi + 2*self.L@pi + self.C)

        #My edit: implementing g = 2 K pi + 2 pi L + C
        return (self.l_x*2*self.K@pi + self.l_y*2*pi@self.L + self.C)

    # entropy term
    def h(self, pi):
        return np.sum(np.multiply(pi, np.log(pi / np.e)))

    # fixed-point update
    def fp_update(self, pi):
        pi_next = ot.sinkhorn(self.w, self.v, self.g_grad(pi), reg=self.l)
        return pi_next

    # Gradient descent with KL divergence
    def gdkl_update(self, pi, tau):
        pi_next = ot.sinkhorn(self.w, self.v, self.g_grad(pi)+(self.l-(1.0/tau))*np.log(pi), reg=(1.0/tau))
        return pi_next

    # solve the optimization problem
    def solve(self, method, tau_scale=1.0, max_iter=10000, pi_tol=1e-5, obj_tol=1e-16, pi_init=None, verbose=False):

        # initialization
        if pi_init is None:
            pi = self.w[:,np.newaxis] @ self.v[np.newaxis,:]

        else:
            pi = pi_init

        # compute the objective
        obj = self.g(pi) + self.l*self.h(pi)
        

        if verbose==True:
            print(f'Initialization: objective = {obj}')

        # fixed point
        if method == 'fp':
            print('Implement fixed point algorithm')

            # iterations
            i = 0
            while i < max_iter:

                pi_prev = pi
                obj_prev = obj

                ###################################
                # implement the optimization update
                pi = self.fp_update(pi_prev)
                ###################################

                # compute the change in pi (L1)
                pi_change = np.sum(np.abs(pi - pi_prev))  # L1 distance

                # compute the relative change in the objective value
                obj = self.g(pi) + self.l*self.h(pi)
                print(f'max of entropy term :{np.max(self.l*self.h(pi))} \n max of main term: {np.max(self.g(pi))}')
                obj_change = obj - obj_prev
                obj_rel_change = obj_change / np.abs(obj_prev)

                ###################################
                # method-specific rules
                if obj_rel_change > 0.0:
                    print(f"Terminated after {i} iterations (objective increased)")
                    pi = pi_prev    # no update and return the previous pi
                    break
                ###################################

                # print the results
                if verbose==True:
                    print(f'Iteration {i + 1}: objective = {obj}, objective relative change = {obj_rel_change}, pi change (L1) = {pi_change}')

                ###################################
                # method-specific stopping criteria
                # stop if the objective does not change = obj(pi) is close to obj(pi_prev)
                if np.abs(obj_rel_change) < obj_tol:
                    print(f"Terminated after {i + 1} iterations (objective converged)")
                    break

                # stop if pi does not change = pi is close to pi_prev in L1
                if pi_change < pi_tol:
                    print(f"Terminated after {i + 1} iterations (coupling converged)")
                    break
                ###################################

                # to the next iteration
                i += 1

        # Gradient descent with KL divergence
        elif method == 'gdkl':
            print('Implement gradient descent with KL divergence')
            tau = tau_scale / (2*self.Kinf + 2*self.Linf + self.l)

            # iterations
            i = 0
            while i < max_iter:

                pi_prev = pi
                obj_prev = obj

                ###################################
                # implement the optimization update
                pi = self.gdkl_update(pi_prev, tau)
                ###################################

                # compute the change in pi (L1)
                pi_change = np.sum(np.abs(pi - pi_prev))  # L1 distance

                # compute the relative change in the objective value
                obj = self.g(pi) + self.l*self.h(pi)
                obj_change = obj - obj_prev
                obj_rel_change = obj_change / np.abs(obj_prev)

                ###################################
                # method-specific rules
                ###################################

                # print the results
                if verbose==True:
                    print(f'Iteration {i + 1}: objective = {obj}, objective relative change = {obj_rel_change}, pi change (L1) = {pi_change}')

                ###################################
                # method-specific stopping criteria
                # stop if the objective does not change = obj(pi) is close to obj(pi_prev)
                if np.abs(obj_rel_change) < obj_tol:
                    print(f"Terminated after {i + 1} iterations (objective converged)")
                    break

                # stop if pi does not change = pi is close to pi_prev in L1
                if pi_change < pi_tol:
                    print(f"Terminated after {i + 1} iterations (coupling converged)")
                    break
                ###################################

                # to the next iteration
                i += 1

        else:
            raise ValueError('Invalid method')

        if i == max_iter:
            print("Reached the max iteration")

        # save and return the solution
        self.pi = pi
        return pi


if __name__ == "__main__":
    from create_clouds import create_clouds
    from graph_laplacians_and_c import graph_laplacians, cost_matrix_c
    import matplotlib
    matplotlib.use("TkAgg")  # or "TkAgg" if you want an interactive window
    import matplotlib.pyplot as plt
    path_X = "Data/CAPOD/class3/m25.obj"
    path_Y = "Data/CAPOD/class3/m26.obj"
    X, Y = create_clouds(path_X, path_Y)
    X = X[:500, :]
    Y = Y[:500, :]
    L_X, L_Y, W_x, W_y, a, b = graph_laplacians(X, Y)
    W1_matrix = cost_matrix_c(X, Y, a, b)

    # Cost matrix
    C = W1_matrix

    # Laplacian matrices
    K = L_X
    L = L_Y

    # KSC
    # define the KSC problem
    # w, v set to uniform weights by default if not provided
    # solve the problem by the fixed-point (fp) or gradient descent with KL divergence (gdkl)
    # verbose=True to print the iteration information
    # method='fp' converges fast for lambda above certain threshold
    # method='gdkl' converges for any lambda but can be slow
    # for method='gdkl', the step size is (tau_scale) * (theoretical threshold)
    # providing tau_scale > 1.0 (default=1.0) to take a more aggressive step size may speed up method='gdkl'

    # 1. Fixed-point (lambda should not be too small)
    l = 0.7  # lambda
    l_x = 1
    l_y = 1
    prob = KSC(K, L, C, l, l_x, l_y, w=a, v=b)
    pi_fp = prob.solve(method='fp', verbose=True)

    prob_2 = KSC(K, L, C, l, l_x, l_y)
    pi_fp_1 = prob_2.solve(method='fp', verbose=True)


    matrices = [pi_fp, pi_fp_1, W1_matrix]
    titles = ['pi non_uniform', 'pi unifrom', 'C']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns
    for i, ax in enumerate(axes):
        im = ax.imshow(matrices[i], cmap='viridis', interpolation='nearest')
        ax.set_title(titles[i])
        ax.axis('off')  # optional: hide axis ticks

    plt.tight_layout()
    plt.show()
