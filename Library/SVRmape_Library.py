class SVR_mape:
    
    """
    SVR based on MAPE loss.
    Fits models where the values of the target variable are positive.
    
        -- Parameter --
            C: determines the number of points that contribute to creation of the boundary. 
               (Default = 0.1)
               The bigger the value of C, the lesser the points that the model will consider.
               
            epsilon: defines the maximum margin in the feature space (Default = 0.1).
                        The bigger its value, the more general~underfitted the model is.
                        
            lamda: controls the implication of the weighted Elastic Net regularization.
                        
            kernel: name of the kernel that the model will use. Written in a string format.
                    (Default = "linear"). 
        
                    acceptable parameters: 
                        "linear", "poly", "polynomial", "rbf", 
                        "laplacian", "cosine".
        
                    for more information about individual kernels, visit the 
                    sklearn pairwise metrics affinities and kernels user guide.
                    
                    https://scikit-learn.org/stable/modules/metrics.html
            
            Specific kernel parameters: 

        --Methods--
            fit(X, y): Learn from the data. Returns self.

            predict(X_test): Predicts new points. Returns X_test labels.

            coef_(): Returns alpha support vectors (sv) coefficient, X sv, and b.

            For more information about each method, visit specific documentations.
            
        --Example-- 
            ## Call the class in a jupyter notebook
            >>> from SVRmape_Library import SVR_mape
            ...
            ## Initialize the SVR object with custom parameters
            >>> model = SVR_mape(C = 10, kernel = "rbf", gamma = 0.1)
            ...
            ## Use the model to fit the data
            >>> fitted_model = model.fit(X, y)
            ...
            ## Predict with the given model
            >>> y_prediction = fitted_model.predict(X_test)
            ...
            ## e.g
            >>> print(y_prediction)
            np.array([12.8, 31.6, 16.2, 90.5, 28, 1, 49.7])
    
    """
    
    def __init__(self, C = 0.1, epsilon = 0.01, kernel = "linear", **kernel_param):
        import numpy as np
        from cvxopt import matrix, solvers, sparse
        from sklearn.metrics.pairwise import pairwise_kernels
        self.sparse = sparse
        self.matrix = matrix
        self.solvers = solvers
        self.np = np
        
        self.C = C
        self.epsilon = epsilon
        
        self.kernel = kernel
        self.pairwise_kernels = pairwise_kernels
        self.kernel_param = kernel_param
        
        
    def fit(self, X, y):
        """ 
        Computes coefficients for new data prediction.
        
            --Parameters--
                X: nxm matrix that contains all data points
                   components. n is the number of points and
                   m is the number of features of each point.
                   
                y: nx1 matrix that contains labels for all
                   the points.
            
            --Returns--
                self, containing all the parameters needed to 
                compute new data points.
        """
        # hyperparameters
        C = self.C 
        epsilon =  self.epsilon
        
        kernel = self.kernel
        pairwise_kernels = self.pairwise_kernels
        
        sparse = self.sparse 
        matrix = self.matrix 
        solvers = self.solvers 
        np = self.np
        
        # Useful parameters
        ydim = y.shape[0]
        onev = np.ones((ydim,1))
        x0 = np.random.rand(ydim)
        
        # Prematrices for the optimizer
        K = pairwise_kernels(X, X, metric = kernel, **self.kernel_param)
        
        A = onev.T
        b = 0.0
        G = np.concatenate((np.identity(ydim), -np.identity(ydim)))
        h_ = np.concatenate((100*C*np.ones(ydim)/y, 100*C*np.ones(ydim)/y)); 
        h = h_.reshape(-1, 1)

        # Matrices for the optimizer
        A = matrix(A)
        b = matrix(b)
        G = sparse(matrix(G))
        h = matrix(h)
        Ev = (epsilon*y.T)/100
        
        # functions for the optimizer
        def obj_func(x):
            return 0.5* x.T @ K @ x + Ev @ (np.abs(x)) - y.T @ x

        def obj_grad(x):
            return x.T @ K + Ev @ (x/np.abs(x)) - y
        
        def F(x = None, z = None):
            if x is None: return 0, matrix(x0)
            # objective dunction
            val = matrix(obj_func(x))
            # obj. func. gradient
            Df = matrix(obj_grad(x))
            if z is None: return val, Df
            # hessian
            H = matrix(z[0] * K)
            return val, Df, H
        
        # Solver
        solvers.options['show_progress'] = False
        sol = solvers.cp(F=F, G=G, h=h, A=A, b=b)

        # Support vectors
        beta_1 = np.array(sol['x']).reshape(-1)
        beta_n = np.abs(beta_1)/beta_1.max()
        indx = beta_n > 1e-4
        beta_sv = beta_1[indx]
        x_sv = X[indx,:]
        y_sv = y[indx]
        
        # get w_phi and b
        k_sv = pairwise_kernels(x_sv, x_sv, metric = kernel, **self.kernel_param)
        
        cons = np.where(beta_sv >= 0, 1 - epsilon/100, 1 + epsilon/100)
        
        w_phi = beta_sv @ k_sv
        b = np.mean((y_sv*cons - w_phi)); self.b = b
        self.beta_sv = beta_sv; self.x_sv = x_sv
        return self
        
    def predict(self, X_):
        """
            Predicts new labels for a given set of new 
           independent variables (X_test).
           
           --Parameters--
               X_test: nxm matrix containing all the points that 
                       will be predicted by the model.
                       n is the number of points. m represents the
                       number of features/dimensions of each point.
            
           --Returns--
               a nx1 vector containing the predicted labels for the 
               input variables.
                
        """
        k_test = self.pairwise_kernels(self.x_sv, X_, metric = self.kernel, **self.kernel_param)

        w_phi_test = self.beta_sv @ k_test
        predict = w_phi_test + self.b
        return predict
    
    def coef_(self):
        """
        --Returns--
            - dual support vectors
            - primal support vectors
            - intercept
        """
        return self.beta_sv, self.x_sv, self.b
