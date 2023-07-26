#Kernel Approximation Class

#importing libraries
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sympy 

import time
from tqdm import tqdm #import function for adding progress bar (algorithm takes a while)

from itertools import chain, combinations #import function for powerset

import concurrent.futures #import function for parallelization
import multiprocessing #for getting number of cores

#for interactive plots
#%matplotlib widget

######################################################################################################################################################################################################################################################################################################################################################################################################################################
class KernelApprox(object):
    ##############################################################################################################
    #initializing the class
    def __init__(self, Y, lamda = None, n = None, alpha = 4, num_workers = None, verbose = True):

        self.samps = Y #samples function to be approximated
        self.alpha = alpha #shape parameter for kernel (should be even)
        self.verbose = verbose #if true, print progress bar

        #useful constants for kernel
        self.w_const = ((2*np.pi)**alpha)/((-1)**((alpha/2)+1) * np.math.factorial(alpha)) #constant for kernel
        self.w_const2 = ((2*np.pi)**(alpha * 2))/((-1)**((alpha)+1) * np.math.factorial(alpha*2))  #constant for kernel if tau = 2
        #set dimension
        try:
            self.d = Y.shape[1] #set dimension
        except:
            self.d = 1
        #set number of samples
        self.m = len(Y)
        #regularization parameter
        if lamda is None:
            self.lamda = (self.m)**(-1/(1 + 1e-6 + (1/4) )) #if lamda is not given, set it to this value
        else:
            self.lamda = lamda
        #number of lattice points    
        if n is None:
            self.n = sympy.prevprime(int((self.m)**(1/( (alpha/2)*(1+1e-6) + 0.5 + (((alpha/4) - 1e-6)/2)*(1 + 1e-6 + 1/alpha)  )) ) + 1) #if n is not given, set it to this value
        else:
            self.n = n 
                    #set number of workers
        #number of workers
        if num_workers is None: #if number of workers is not given, set it to default value
            if self.m > int(1e7): # half the number of cores to avoid memory issues
                self.num_workers = (multiprocessing.cpu_count() - 4)/2
            else:
                self.num_workers = (multiprocessing.cpu_count() - 4) # default number of workers 
        else:
            self.num_workers = num_workers

        
    ##############################################################################################################

    ##############################################################################################################
    #function to calculate the weights (for given example)
    def weights(self, u):
        prod = 1
        for j in u:
            prod *= 1/(j**self.alpha)
        return prod
    ##############################################################################################################

    ##############################################################################################################
    #function to calculate omega
    def w(self, z, k):
        n, alpha = self.n, self.alpha #extracting n and alpha

        x = (np.outer(z,k) % n)/n
        
        temp = np.arange(alpha+1)
        binoms = scipy.special.binom(alpha,temp[::-1])
        return self.w_const * np.sum([binoms * scipy.special.bernoulli(alpha) * (a[:,None]**(temp[::-1])) for a in x], axis=2)

    def w_old(self, z, k):
        n, alpha = self.n, self.alpha #extracting n and alpha
        return self.w_const * sympy.bernoulli(alpha, ((k*z) % n)/n )
    ##############################################################################################################
    
    ##############################################################################################################
    # function to choose generating vector for lattice points via CBC construction
    def CBC(self):
        d, n, alpha = self.d, self.n, self.alpha #extracting short forms

        #initialize vector for storing z
        z = np.ones(d)
        
        #compute weights for all dimensions
        gamma = np.ones(d)
        for i in range(d):
            gamma[i] = self.weights([i+1])
        self.gamma = gamma #save weights

        #if dimension is 1, then z = 1
        if d == 1: 
            self.gen_vec = z
            return z

        #initialize matrix for storing Pd values
        Pd = np.ones((d, n))

        #construct matrices Psi and Omega
        z_s = np.arange(1,n)
        k = np.arange(n)
        Omega = self.w(z_s, k)
        Psi = Omega**2

        #loop over all dimensions (except first)
        for i in range(1, d):
            #compute vectors W_ds and V_ds
            W_ds = Pd[i-1] * (gamma[i])
            V_ds = Pd[i-1] * (gamma[i]**2)
            
            #compute criterion for all possible z_s
            crit = (np.matmul(Psi, V_ds) + 2*np.matmul(Omega, W_ds))/n

            #choose z_s that minimizes criterion
            z[i] = np.argmin(crit) + 1

            #update Pd (for next dimension)
            k = np.arange(n)
            Pd[i] = ( 1  + gamma[1] * self.w(z[i], k) ) * Pd[i-1]

        #save the generating vector
        self.gen_vec = z

        #return the generating vector
        return z
    ##############################################################################################################

    ##############################################################################################################
    #function to compute the lattice points
    def gen_lattice(self):
        d, n = self.d, self.n #extracting short forms

        #extract the generating vector
        if hasattr(self, 'gen_vec'):
            z = self.gen_vec
        else:
            z = self.CBC()
        
        #compute n lattice points
        k = np.arange(0,n)
        x = (np.matmul(k[:,None],z[None,:]) % n)/n

        #save the lattice points
        self.lattice = x

        #return the lattice points
        return x
    ##############################################################################################################

    ##############################################################################################################

    #function to get powerset of a set
    def powerset(self, iterable):
        "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        #doesnt include empty set
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))

    ##############################################################################################################

    ##############################################################################################################
    #function to compute the reproducing kernel
    def kernel(self, x, y, tau = 1):
        d, alpha = self.d, self.alpha #extracting short forms

        try: #check if gamma is already computed 
            self.gamma #should be if CBC is already run (should be if kernel is called)
        except:
            #compute weights for all dimensions
            gamma = np.ones(d)
            for i in range(d):
                gamma[i] = self.weights([i+1])
            self.gamma = gamma #save weights

        #tau = 1 except in special cases, so this is vectorised
        if (tau == 1): 
            try: 
                x.shape[1]
                z = (x - y)%1  
            except: #for single points
                z = np.array([(x - y)%1])

            del x, y #delete x and y to save memory

            #Compute bernoulli polynomial of order alpha at z for each dimension
            if self.alpha == 4:
                fy = z**4 - 2*z**3 + z**2 -1/30
            elif self.alpha == 2:
                fy = z**2 - z + 1/6
            else: #for general alpha (even)
                n = alpha*tau #order of bernoulli polynomial
                temp = np.arange(n+1) #powers of bernoulli polynomial
                binoms = scipy.special.binom(n,temp[::-1]) #binomial coefficients
                fy = np.sum([binoms * scipy.special.bernoulli(n) * (a[:,None]**(temp[::-1])) for a in z], axis=2) 

            #calculate 1 + constants*(bernoulli polynomial evaluated at z) for each dimension
            fy = 1 + (self.gamma**tau)*self.w_const*fy 
            fy = np.prod(fy, axis=1, keepdims=True) #product over all dimensions
            return fy
        
        elif tau == 2:
            w_const = self.w_const2
        else:
            w_const = ((2*np.pi)**(alpha*tau))/((-1)**((alpha*tau/2)+1) * np.math.factorial(alpha*tau)) 

        #compute the kernel
        sum = 1
        for u in list(self.powerset( np.arange(1,d+1) )):
            uu = np.array(u)
            gamma_u = np.prod(self.gamma[np.array(u)-1])
            prod = 1
            for j in (uu-1):
                prod *= sympy.bernoulli(alpha*tau, ((x[j]-y[j]) % 1) )
            sum += (gamma_u**tau)*(w_const**len(uu))*prod
        
        return sum
    ##############################################################################################################

    ##############################################################################################################
    #function for parallelizing computation of vector A
    def A_task(self, i):
        A_i = np.zeros(self.n)
        for j in range(self.n):
            A_i[j] = self.kernel(self.lattice[i,:], self.lattice[j,:], 2) + self.lamda * self.kernel(self.lattice[i,:], self.lattice[j,:], 1) 
        return A_i

    #function compute the Gram matrix A
    def Gram(self):
        try:
            self.lattice #x = self.lattice
        except:
            self.gen_lattice() #x = self.gen_lattice()

        n = self.n #extracting short forms

        #compute vector A (parallelised)
        with concurrent.futures.ProcessPoolExecutor(max_workers= self.num_workers) as executor:
            items = range(n)
            results = list(tqdm(executor.map(self.A_task, items), total=n, leave=True, position=0, desc="Computing Gram matrix", disable= not(self.verbose)))

        A = np.array(results).astype(np.float64)

        #save the Gram matrix
        self.A = A
        
        return A

    ##############################################################################################################

    ##############################################################################################################
    #function for parallelizing computation of vector b
    def b_task(self, i):
        return np.mean( self.kernel( (self.samps - self.lattice[i]) , 0, 1) )

    #function to compute vector b
    def gen_b(self):
        try:
            x = self.lattice
        except:
            x = self.gen_lattice()

        num_workers = self.num_workers


        n, m = self.n, self.m #extracting short forms

        
        #compute vector b (parallelised)
        with concurrent.futures.ProcessPoolExecutor(max_workers= num_workers) as executor:
            items = range(n)
            results = list(tqdm(executor.map(self.b_task, items), total=n, leave=True, position=0, desc="Computing vector b", disable= not(self.verbose)))

        b = np.array(results).astype(np.float64)

        #save the vector b
        self.b = b

        return b
    ##############################################################################################################

    ##############################################################################################################
    #function to compute coefficients c
    def gen_c(self):
        try:
            A = self.A
        except:
            A = self.Gram()

        try:
            b = self.b
        except:
            b = self.gen_b()

        c = np.linalg.solve(A, b)

        #save the coefficients c
        self.c = c

        return c
    ##############################################################################################################

    ##############################################################################################################
    #function for parallelising computation of kernel estimate
    def est_task(self, i):
        return self.c[i]*( self.kernel( (self.y - self.lattice[i]) , 0, 1) )
    


    #function from computing the kernel estimate for each point
    def kern_est(self, y, num_workers = None):
        n = self.n #extracting short forms

        self.y = y #save the points to estimate at (for use in est_task)
        del y #(for memory)

        try:
            x = self.lattice
        except:
            x = self.gen_lattice()

        try:
            c = self.c
        except:
            c = self.gen_c()

        #set the number of workers if not specified
        if num_workers is None:
            num_workers = self.num_workers

        if (len(self.y)*self.n) < 3e9: #check if the number of computations is too large for memory
            #compute the kernel estimate (parallelised)
            with concurrent.futures.ProcessPoolExecutor(max_workers= num_workers) as executor:
                items = range(n)
                results = list(tqdm(executor.map(self.est_task, items), total=n, leave=True, position=0, desc="Computing kernel estimate", disable= not(self.verbose)) )
            
            #est = np.sum(results, axis = 0)
            est = np.sum(np.array(results).astype(np.float64), axis = 0)
            
            #save the kernel estimate
            self.est = est

            return est
        else: #if the number of computations is too large for memory, split the computations into smaller chunks
            num_split = int((len(self.y)*self.n)/2e9) + 1 #number of splits
            y = np.array_split(self.y, num_split)
            est = np.zeros(len(self.y))
            count = 0

            for i in tqdm(range(num_split), desc="Computing kernel estimate", leave = True, disable= not(self.verbose), position=0):
                self.y = y[i]
                #compute the kernel estimate (parallelised)
                with concurrent.futures.ProcessPoolExecutor(max_workers= num_workers) as executor:
                    items = range(n)
                    results = list(tqdm(executor.map(self.est_task, items), total=n, leave=False, position=1, desc="Computing a partition", disable= not(self.verbose)) )
                
                est[count:count+len(y[i])] = np.sum(np.array(results).astype(np.float64), axis = 0).T
                count += len(y[i])
            
            #save the kernel estimate
            self.est = np.array([est]).T

            return self.est
    ##############################################################################################################

    ##############################################################################################################
    #function for computing the kernel estimate for all points
    def kern_est_grid(self, N = 100):
        d, n = self.d, self.n #extracting short forms

        #generate a grid
        XXX = np.meshgrid(*[np.linspace(i,j,N+1) for i,j in zip( np.zeros(d),np.ones(d) )])
        ys = np.vstack(list(map(np.ravel, XXX))).T #format usable by kern_est

        #save the points
        self.ys = ys

        est = self.kern_est(ys)

        #save the kernel estimate
        self.est_grid = est

        return est
    ##############################################################################################################

    ##############################################################################################################
    #function for plotting the kernel estimate for 2dims of the unit cube
    def plot_est(self, dim1 = 0, dim2 = 1, N = 100):
        #%matplotlib widget

        #plot in 3d
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm


        try:
            est = self.est_grid
        except:
            est = self.kern_est_grid(N)
            ys = self.ys

        try:
            ys = self.ys
        except:
            d = self.d
            XXX = np.meshgrid(*[np.linspace(i,j,N+1)[:-1] for i,j in zip( np.zeros(d),np.ones(d) )])
            ys = np.vstack( list( map(np.ravel, XXX) ) ).T #added list() to stop error notifcation
            self.ys = ys 
        
        
        if self.d == 1:
            plt.plot(ys, est)
            plt.xlabel('y')
            plt.ylabel('f(y)')
        else:
    
            X = np.reshape(ys[:,dim1], (N, N))
            Y = np.reshape(ys[:,dim2], (N, N))
            Z = np.reshape(est, (N, N))

            fig = plt.figure(figsize=(12,6))

            #first plot (zoomed in z)
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            surf = ax.plot_surface(X,Y,Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

            ax.set_ylim(0, 1)
            ax.set_xlabel('y1') #x-axis label
            ax.set_ylabel('y2') #y-axis label
            ax.set_zlabel('f(y)') #y-axis label

            #second plot
            ax = fig.add_subplot(1, 2, 2, projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            
            ax.set_zlim(0, 1)
            ax.set_xlabel('y1') #x-axis label
            ax.set_ylabel('y2') #y-axis label
            ax.set_zlabel('f(y)') #y-axis label

        plt.show()
    ##############################################################################################################

    ##############################################################################################################
    #function for error analysis
    def calc_error_L2(self, f, N = int(1e4), sob_pts = 7, x = None, num_workers = None):

        #get a lattice for computing the error
        if x is None:
            save_n = self.n #save the current n
            self.n = N #set n to N for CBC algorithm

            z = self.CBC()
            k = np.arange(1,N+1)
            x = (np.matmul(k[:,None],z[None,:]) % N)/N #lattice points

            self.n = save_n #reset n

        #get some sobol points
        sobs = scipy.stats.qmc.Sobol(self.d)
        temp = sobs.random_base2(sob_pts) # 2**sob_pts sobol points

        #add the sobol points to the lattice points
        points = np.zeros((len(temp)*len(x), self.d))
        points.shape
        for i in range(len(temp)):
            points[i*len(x):(i+1)*len(x)] = (x + temp[i]) %1

        #compute the error (MSE)
        err_raw = (self.kern_est(points, num_workers) - f(points))**2
        err = np.mean(err_raw)
        err_var = np.var(err_raw)/len(err_raw)

        if self.verbose:
            print("Error = ", err, " +/- ", np.sqrt(err_var))

        #save the error
        self.L2_error = err
        self.L2_error_var = err_var
        self.lattice_N = x # save the lattice points

        return err
    ##############################################################################################################

    ##############################################################################################################
    def gen_samples(self, f, K, const = 1.1):
        d = self.d

        #proposal is const*uniform[0,1]

        #extra points to sample to ensure K points are found
        extra = const**2 + 0.05 + 1/np.sqrt(K*0.1) + 1/np.sqrt(self.m*0.1) #tends to const**2 + 0.05 as K,M -> infinty
        K_adj = int(K*extra) #adjusted K

        u = scipy.stats.uniform.rvs(size = K_adj)
        y = scipy.stats.uniform.rvs(size = (K_adj, d))

        fy = f(y)

        Y_est = y[np.where(np.less(u,fy[:,0]/(const**2)))]

        if len(Y_est) < K:
            while len(Y_est) < K:
                u = scipy.stats.uniform.rvs(size = K_adj)
                y = scipy.stats.uniform.rvs(size = (K_adj, d))

                fy = f(y)

                Y_est = np.concatenate((Y_est, y[np.where(np.less(u,fy[:,0]/(const**2)))]), axis = 0)

        return Y_est[:K]

    ##############################################################################################################

    ##############################################################################################################

    #function for calculating the weak error
    def calc_error(self, f, g, K = int(1e8), num_workers = None, const = 1.1):
        #set the number of workers if not specified
        if num_workers is None:
            num_workers = self.num_workers
        
        #correlated samples
        ############################################################
        #extra points to sample to ensure K points are found
        extra = const**2 + 0.05 + 1/np.sqrt(K*0.1) + 1/np.sqrt(self.m*0.1) #tends to const**2 + 0.05 as K,M -> infinty

        K_adj = int(K*extra)

        u = scipy.stats.uniform.rvs(size = K_adj)
        y = scipy.stats.uniform.rvs(size = (K_adj, self.d))

        fy_est = self.kern_est(y, num_workers = num_workers)
        fy = f(y)

        accept_f = np.less(u,fy[:,0]/(const**2)) # proposal samples accepted for f 
        accept_f_est = np.less(u,fy_est[:,0]/(const**2)) #proposal samples accepted for f_est

        #samples accepted for either f or f_est
        accept_either =  np.logical_or( np.logical_or( np.logical_and(accept_f, accept_f_est),  np.logical_and(accept_f, np.logical_not(accept_f_est)) ), np.logical_and(np.logical_not(accept_f), accept_f_est) )

        #subset of accept_either where only one of f or f_est is accepted
        accept_just_f = np.logical_and(accept_f, np.logical_not(accept_f_est))[(accept_either)]
        accept_just_f_est = np.logical_and(np.logical_not(accept_f), accept_f_est)[(accept_either)]

        Y = y[(accept_either)]
        Y_est = y[(accept_either)]

        #subset of Y where just f is accepted
        if np.sum(accept_just_f_est) > 0: 
            Y[(accept_just_f_est)] = self.gen_samples(f, np.sum(accept_just_f_est), const = const)
        if np.sum(accept_just_f) > 0:
            Y_est[(accept_just_f)] = self.gen_samples(self.kern_est, np.sum(accept_just_f), const = const)

        Y = Y[:K]
        Y_est = Y_est[:K]

        self.Y = Y # save the samples
        self.Y_est = Y_est # save the samples
        ############################################################
        g_peram = 1e-3
        ############################################################

        err = abs( np.mean( (g(Y_est, g_peram) - g(Y, g_peram)) ) ) #numpy.subtract()
        err_var = np.var( (g(Y_est, g_peram) - g(Y, g_peram)) )/K

        if self.verbose:
            print("Error = ", err, " +/- ", np.sqrt(err_var))

        #save the error
        self.error = err
        self.error_var = err_var

        return err

    
######################################################################################################################################################################################################################################################################################################################################################################################################################################
