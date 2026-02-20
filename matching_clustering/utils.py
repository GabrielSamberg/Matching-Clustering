import numpy as np
import ot

class KSC:
    def __init__(self, K, L, C, l, l_x, l_y, w=None, v=None):
        # K (n, n) symmetric
        # L (m, m) symmetric
        # C (n, m)
        # l = lambda
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