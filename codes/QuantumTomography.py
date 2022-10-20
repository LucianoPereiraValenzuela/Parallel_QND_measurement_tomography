#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize, least_squares

# In[2]:

class Results:
    # class for results of tomography
    def __init__(self) -> None:
        pass

def Outer2Kron( A, Dims ):
    # From vec(A) outer vec(B) to A kron B
    N   = len(Dims)
    Dim = A.shape
    A   = np.transpose( A.reshape(2*Dims), np.array([range(N),range(N,2*N) ]).T.flatten() ).flatten()
    return A.reshape(Dim)

def Kron2Outer( A, Dims ):
    # From A kron B to vec(A) outer vec(B)
    N   = len(Dims)
    Dim = A.shape
    A   = np.transpose( A.reshape( np.kron(np.array([1,1]),Dims) ), np.array([range(0,2*N,2),range(1,2*N,2)]).flatten() ).flatten()
    return A.reshape(Dim)
    
def LocalProduct( Psi, Operators , Dims=[] ):
    """
    Calculate the product (A1xA2x...xAn)|psi>
    """
    sz = Psi
    if not Dims: 
        Dims = [ Operators[k].shape[-1] for k in range( len(Operators) ) ]
    N = len(Dims)
    for k in range(N):
        Psi  = (( Operators[k]@Psi.reshape(Dims[k],-1) ).T ).flatten()
    return Psi

def InnerProductMatrices( X, B, Vectorized = False ):
    """
    Calculate the inner product tr( X [B1xB2x...xBn])
    """
    X = np.array(X)
    
    if isinstance(B, list): 
        B = B.copy()
        nsys = len(B)
        nops = []
        Dims = []
        if Vectorized == False :
            for j in range(nsys):
                B[j] = np.array(B[j])
                if B[j].ndim == 2 :
                    B[j] = np.array([B[j]])
                nops.append( B[j].shape[0] )
                Dims.append( B[j].shape[1] )
                B[j] = B[j].reshape(nops[j],Dims[j]**2)
        elif Vectorized == True :
            for j in range(nsys):
                nops.append( B[j].shape[0] )
                Dims.append( int(np.sqrt(B[j].shape[1])) )                
        
        if X.ndim == 2 :       
            TrXB = LocalProduct( Outer2Kron( X.flatten(), Dims ), B ) 
        elif X.ndim == 3 :
            TrXB = []
            for j in range( X.shape[0] ):
                TrXB.append( LocalProduct( Outer2Kron( X[j].flatten(), Dims ), B ) )
        elif X.ndim == 1:
            TrXB = LocalProduct( Outer2Kron( X, Dims ), B ) 
        
        return np.array( TrXB ).reshape(nops)
        
    elif isinstance(B, np.ndarray):     
        
        if B.ndim == 2 and Vectorized == False :
            return np.trace( X @ B )
        
        elif B.ndim == 4 :
            nsys = B.shape[0]
            nops = nsys*[ B[0].shape[0] ]
            Dims = nsys*[ B[0].shape[1] ]
            B = B.reshape(nsys,nops[0],Dims[0]**2)
            
        elif B.ndim == 3 :
            if Vectorized == False :
                nsys = 1
                nops = B.shape[0]       
                Dims = [ B.shape[1] ]
                B = B.reshape(nsys,nops,Dims[0]**2)
            if Vectorized == True :
                nsys = B.shape[0]
                nops = nsys*[ B[0].shape[0] ]
                Dims = nsys*[ int(np.sqrt(B[0].shape[1])) ]
        if X.ndim == 2 :       
            TrXB = LocalProduct( Outer2Kron( X.flatten(), Dims ), B ) 
        elif X.ndim == 3 :
            TrXB = []
            for j in range( X.shape[0] ):
                TrXB.append( LocalProduct( Outer2Kron( X[j].flatten(), Dims ), B ) )
        elif X.ndim == 1:
            TrXB = LocalProduct( Outer2Kron( X, Dims ), B ) 

        return np.array( TrXB ).reshape(nops)
    

def LinearCombinationMatrices( c, B, Vectorized = False ):  
    
    nsys = len(B)
    nops = [ np.array(B[k]).shape[0] for k in range( nsys ) ] 

    if Vectorized == False:
        Dims = [ np.array(B[k]).shape[1] for k in range( nsys ) ]
        Bv = []
        for k in range( len(B) ):
            Bv.append( B[k].reshape(-1,Dims[k]**2).T )
    else:
        Dims = [ np.sqrt(np.array(B[k]).shape[1]) for k in range( nsys ) ]
        Bv = B
        
    Lambda = Kron2Outer( LocalProduct( c.flatten(), Bv ), Dims ).reshape( np.prod(Dims), -1  )
    
    return Lambda
    
    
def QuantitiesRho( rho, Pi, p_ex , Vectorized = False ):    
    """
    Calculate  the log-likelihood multinomial function 
    """
    p_th = InnerProductMatrices( rho, Pi, Vectorized=Vectorized )
    f    = np.sum( - p_ex * np.log( p_th + 1e-12 ) ) 
    return f
def VectorizeVectors( Vectors ):
    if Vectors.ndim == 2 :
        Vectors = np.array([ np.outer(Vectors[:,j],Vectors[:,j].conj() ) for j in range(Vectors.shape[1]) ])
        return Vectors.reshape([ Vectors.shape[0], -1 ]).T
    elif Vectors.ndim == 3 :
        return Vectors.reshape([ -1, Vectors.shape[2] ])

def SimulateMeasurement( rho , Measures , sample_size , vectorized = False, output=0 ):
    
    if rho.ndim == 3 and vectorized == False:
        Measures = VectorizeVectors(Measures)
        rho      = VectorizeVectors(rho)
    elif rho.ndim == 2 and vectorized == False:
        Measures = VectorizeVectors(Measures)
        rho      = rho.flatten()
        
    def Temp( rho, Measures, sample_size, output ):
        probabilities = np.abs(np.conj(Measures).T@rho)
        if sample_size > 0:
            probabilities = np.random.multinomial( sample_size, probabilities  )
        if output == 0:
            return probabilities
        else:
            return probabilities/np.sum(probabilities)            
    
    if rho.ndim == 1:
        return Temp( rho, Measures, sample_size,output )
    elif rho.ndim == 2:
        return np.array( [ Temp(rho[:,k],Measures,sample_size,output) for k in range(rho.shape[1]) ] ).T
    

def HermitianPart(A):
    return 0.5*(A+A.conj().T)
    
def Complex2Real(A):
    return np.array([ np.real(A), np.imag(A) ])

def Real2Complex(A):
    return A[0,:]+1j*A[1,:]


# In[3]:


def RandomDensity( Dimension, Number = 1, rank = 1 ):  
    if Number == 1:
        Z = np.random.randn(Dimension,rank)+1j*np.random.randn(Dimension,rank) + np.finfo(float).eps
        Rho = Z@np.conj(Z).T + np.finfo(float).eps*np.eye(Dimension)
        Rho = Rho/np.trace(Rho)
        return .5*(Rho+np.conj(Rho).T)
    else:
        Z   = np.random.randn(Dimension,rank,Number)+1j*np.random.randn(Dimension,rank,Number)
        Rho = np.zeros([ Dimension, Dimension, Number], dtype = complex)
        for k in range(Number):
            Rho[:,:,k] = Z[:,:,k]@Z[:,:,k].conj().T
            Rho[:,:,k] = Rho[:,:,k]/np.trace(Rho[:,:,k])
        return 0.5*( Rho + np.transpose(Rho,[1,0,2]).conj() )
        
def RandomState(  Dimension, Number = 1):
    if Number == 1:
        Z = np.random.randn(Dimension)+1j*np.random.randn(Dimension)
        return Z/la.norm(Z)
    else:
        Z = np.random.randn(Dimension,Number)+1j*np.random.randn(Dimension,Number)
        for k in range(Number):
            Z[:,k] = Z[:,k]/la.norm(Z[:,k])
        return Z

def RandomUnitary( Dimension, Number = 1 ):
    if Number == 1:
        Z   = np.random.randn(Dimension,Dimension)+1j*np.random.randn(Dimension,Dimension)
        Q,R = la.qr(Z)
        R   = np.diag(R)
        R   = np.diag(R/np.abs(R))
        return Q@R
    else:
        Z=np.random.randn(Dimension,Dimension,Number)+1j*np.random.randn(Dimension,Dimension,Number)
        for k in range(Number):
            Q,R = la.qr(Z[:,:,k])
            R   = np.diag(R)
            R   = np.diag(R/np.abs(R))
            Z[:,:,k] = Q@R
        return Z

def Fidelity(state1,state2):
    n1 = state1.ndim
    n2 = state2.ndim
    if n1 == n2:
        if n1 == 1:
            fidelity = np.abs(np.vdot(state1,state2))**2
        else:
            temp     = la.sqrtm(state2)
            temp     = 0.5*(temp+np.conj(temp).T)
            temp     = la.sqrtm(temp@state1@temp)
            fidelity = np.trace(0.5*(temp+np.conj(temp).T))
    else:
        if n1==1:
            fidelity = np.vdot(state1,state2@state1)
        else:
            fidelity = np.vdot(state2,state1@state2)
    return np.real(fidelity)

def Infidelity(rho,sigma):
    return 1-Fidelity(rho,sigma)

def Cholesky(rho):
    d = np.shape(rho)[0]
    if d == 1:
        tm = np.real(np.sqrt(rho))
        return tm
    tm = np.zeros(np.shape(rho))+0j
    last_element = rho[d-1][d-1]
    tm[d-1][d-1] = np.real(np.sqrt(last_element))
    if last_element > 0:
        temp = rho[d-1][range(d-1)]
        tm[d-1][range(d-1)] = temp/np.sqrt(last_element)
        recurse = np.hsplit(rho[range(d-1)],[d-1,d])[0] - np.outer(temp.conj().transpose(),temp)/last_element
    else:
        tm[d-1][range(d-1)] = np.zeros(d)
        recurse = np.hsplit(rho[range(d-1)],[d-1,d])[0]
    for i in range(d-1):
        tm[i][range(d-1)] = Cholesky(recurse)[i][range(d-1)]    
    return tm   

def PositiveMatrix2CholeskyVector(rho):
    rho = rho + 1e-8*np.eye(rho.shape[0])
#     tm  = Cholesky(rho)
    tm  = la.cholesky(rho)
    return Triangular2Vector(tm)  

def Triangular2Vector(tm):
    d          = len(tm)
    idx        = 0
    cur_length = d
    t          = np.zeros(d**2)   
    for j in range(d):
        t[np.arange(idx,idx+cur_length)] = np.real(np.diag(tm,j))
        idx = idx + cur_length
        if j>0:
            t[np.arange(idx,idx+cur_length)] = np.imag(np.diag(tm,j))
            idx = idx + cur_length
        cur_length = cur_length -1
    return t     

def CholeskyVector2PositiveMatrix( t, mark = 0 ):
    # mark = 0, no trace constraint
    # mark > 0, trace = mark
    tm  = Vector2Triangular(t)
    rho = np.dot(tm.conj().transpose(),tm)
    rho = ( (mark==0) + mark*(mark>0) )*rho/( (mark==0) + np.trace(rho)*(mark>0) ) 
    return 0.5*( rho + rho.conj().T ) 

def Vector2Triangular(t):
    d          = np.int(np.sqrt(len(t)))
    idx        = 0
    cur_length = d
    tm         = np.zeros([d,d])
    for j in range(np.int(d)):
        tm  = tm + 1*np.diag(t[np.arange(idx,idx+cur_length)],j)
        idx = idx + cur_length
        if j>0:
            tm     = tm + 1j*np.diag(t[np.arange(idx,idx+cur_length)],j)
            idx    = idx + cur_length
        cur_length = cur_length - 1
    return tm

def MaximumLikelihoodStateTomography( Measurements, counts_exp, Guess = [] , Func = 0, vectorized=False ):
    if vectorized == False:
        Measurements = VectorizeVectors(Measurements)
    if isinstance(Guess, list) and not Guess:
        Guess = PositiveOperatorProjection( LinearStateTomography( Measurements, counts_exp, True ) , 1-(Func==0) )
    t_guess   = PositiveMatrix2CholeskyVector(Guess)
    counts_th = lambda t : np.real(Measurements.conj().T@CholeskyVector2PositiveMatrix( t, Func==0  ).flatten())
    fun       = lambda t : LikelihoodFunction( counts_th(t), counts_exp, Func) 
    results   = minimize( fun, t_guess, method = 'SLSQP')  
    t         = results.x
    return CholeskyVector2PositiveMatrix(t, 1)

def MaximumLikelihoodStateTomography_v1( Measurements, counts_exp, Guess = [] , Func = 0, vectorized=False ):
    if vectorized == False:
        Measurements = VectorizeVectors(Measurements)
    if isinstance(Guess, list) and not Guess:
        Guess = PositiveOperatorProjection( LinearStateTomography( Measurements, counts_exp, True ) )
    print(Guess)
    t_guess   = PositiveMatrix2CholeskyVector(Guess)
    counts_th = lambda t : np.real(Measurements.conj().T@CholeskyVector2PositiveMatrix(t).flatten())
    fun       = lambda t : LikelihoodFunction( counts_th(t), counts_exp, Func) 
    gun       = lambda t : np.real( np.trace( CholeskyVector2PositiveMatrix(t) ) - 1 )
    Grad_fun  = lambda t : 2*Triangular2Vector( Vector2Triangular(t) @GradLikelihoodFunction(counts_th(t), counts_exp, Measurements, Func)) 
    Grad_gun  = lambda t : 2*t
    results   = minimize( fun, t_guess, method = 'trust-constr', 
                         jac = Grad_fun , constraints = ({'type': 'eq', 'fun': gun, 'jac': Grad_gun}) ,
                         options = { 'disp' : True } )  
    t         = results.x
    return CholeskyVector2PositiveMatrix(t,1)


def LikelihoodFunction( counts_th, counts_ex, Func=0):
    counts_th = counts_th.flatten()
    counts_ex = counts_ex.flatten()
    if Func == 0: #Multinomial
        #counts_th[counts_ex == 0] = 1
        LogLikelihood = - np.sum( counts_ex*np.log10(counts_th + 1e-16) )  
    elif Func == 1: #Chi-Square
        LogLikelihood = np.sum( (counts_th - counts_ex)**2/(counts_ex + 1e-16 ) )
    elif Func == 2: #Normal
        LogLikelihood = np.sum( (counts_th - counts_ex)**2/(counts_th + 1e-16 ) )
    elif Func == 3: #Square Error 
        LogLikelihood = np.sum( (counts_th - counts_ex)**2 )
    return LogLikelihood

def GradLikelihoodFunction( counts_th, counts, Measurement, Func=0):
    if Func == 0: #Multinomial
        GradLogLikelihood = -np.sum( Measurement*(counts/(counts_th+np.finfo(float).eps)) , 1).reshape( 2*[int(np.sqrt(Measurement.shape[0])) ]  ) 
    return GradLogLikelihood 

def PositiveMatrix2CholeskyVectorV2(rho):
#     T    = Cholesky(rho) rho = Tdag*T
    T    = la.cholesky(rho)
    d    = len(T)
    t_re = np.real( T[ np.tril_indices(d, k= 0, m=d) ] )
    t_im = np.imag( T[ np.tril_indices(d, k= -1, m=d) ] )
    return np.append(t_re,t_im) 

def CholeskyVector2PositiveMatrixV2( t, mark = 0 ):
    # mark = 0, no trace constraint
    # mark > 0, trace = mark
    d = np.int(np.sqrt(len(t)))
    T = np.zeros( [d,d], dtype=complex )
    T[ np.tril_indices(d, k= 0, m=d) ] = t[:int(d*(d+1)/2)]
    T[ np.tril_indices(d, k=-1, m=d) ] += 1j*t[int(d*(d+1)/2):]
    rho = np.dot(T.conj().transpose(),T)
    return ( (mark==0) + mark*(mark>0) )*rho/( (mark==0) + np.trace(rho)*(mark>0) ) 
    
def LinearStateTomography( Measurements, counts, vectorized=False ):
    if vectorized == False:
        Measurements = VectorizeVectors(Measurements)
    Dim   = int(np.sqrt(Measurements[:,0].size))
    State = (la.pinv(Measurements.conj().T)@counts).reshape([Dim,Dim])
    return State

def PositiveOperatorProjection( R, mark = 0 ):
    # Near positive matrix to R
    # mark=0, near with trace=1
    # mark=1, near with trace = tr(R)
    # mark=2, near without constraint
    # mark=3, making positive neagtive eigenvalues
    [ Val, Vec ] = la.eigh(R)
    if mark == 0 or mark == 1:   
        sz   = np.shape( R )[0]
        Indx = np.argsort( np.real(Val) )[::-1]
        M    = ( mark==0 )*1 + ( mark==1 )*np.sum( Val )
        Val  = Val[Indx]
        Vec  = Vec[:,Indx]
        u    = Val - ( np.cumsum( Val ) - M )/np.arange( 1, sz+1 )
        u    = np.max( np.arange( 1, sz+1 )[u>0] )
        if u == 0:
            Val = np.sqrt(Val*(Val>0))
        w    = (np.sum(Val[:u]) - M)/u
        Val  = np.maximum(Val-w, np.finfo(float).eps)
    elif mark == 2:
        Val  = Val*(Val>0)
    elif mark == 3 :
        Val = np.abs(Val)
    R = Vec@np.diag( Val )@Vec.conj().T
    return 0.5*( R + R.conj().T )


# In[4]:


def LinearDetectorTomography(ProveStates, counts, vectorized = False ):
    if vectorized == False:
        ProveStates = VectorizeVectors(ProveStates)
    Measurements = la.pinv(ProveStates.conj().T)@(counts/sum(counts,0)).T
    return Measurements

def ProbabilityOperatorsProjection(Pi):

    dim = int(np.sqrt(Pi.shape[0]))
    Num = Pi.shape[1]
    Pi  = np.array( [ PositiveOperatorProjection( Pi[:,k].reshape(dim,dim), 2).flatten() for k in range(Num) ] ).T
    
    return Pi
    
def POVM_from_t( t, Dim, Num ):
    POVM = np.array([ CholeskyVector2PositiveMatrix( t.reshape([-1,Dim**2] )[k,:] ).flatten() for k in range(Num) ])
    POVM = POVM.reshape(Num, Dim**2).T
    norm = np.trace(np.sum(POVM,1).reshape(Dim,Dim))
    return Dim*POVM/norm 

def MaximumLikelihoodDetectorTomography( ProveStates, counts, Guess = None , Func = 1, vectorized=False ):
    
    if vectorized == False:
        ProveStates = VectorizeVectors(ProveStates)
    
    Dim = int(np.sqrt(ProveStates[:,0].size))
    Num = counts[:,0].size
    
    if Guess is None:
        Guess = ProbabilityOperatorsProjection( LinearDetectorTomography( ProveStates, counts, True ) )
    t_guess   = np.array([ PositiveMatrix2CholeskyVector(Guess[:,k].reshape([Dim,Dim])) for k in range(Num) ]).flatten() 
    
    counts_th   = lambda t : np.real( POVM_from_t( t, Dim, Num ).T.conj()@ProveStates )    
    fun         = lambda t : LikelihoodFunction( counts_th(t), counts, Func )
    constraints = ({'type': 'eq', 'fun': lambda t: la.norm(
                    np.sum(POVM_from_t( t, Dim, Num ), axis=1).flatten() - np.eye(Dim).flatten() ) })
    
    results  = minimize( fun, t_guess, constraints = constraints, method = 'SLSQP',
                        options={'maxiter': 250, 'ftol': 1e-06} )  
    t        = results.x
    Estimate = POVM_from_t( t, Dim, Num )

    result             = Results()
    result.measurement = Estimate
    result.fun         = fun(t) 
    result.entropy     = LikelihoodFunction( counts, counts, Func )

    return result

def LinearProcessTomography(States, Measurements, counts, vectorized = False ):
    if vectorized == False:
        States = VectorizeVectors(States)
        Measurements = VectorizeVectors(Measurements)
    Process = la.pinv(Measurements.conj().T)@(counts)@la.pinv(States.T.conj()).T.conj()
    return Process

def Process2Choi(Process):
    dim = int(np.sqrt(Process[0,:].size))
    Process = np.transpose(Process.reshape([dim,dim,dim,dim]),[0,2,1,3]).reshape([dim**2,dim**2])
    return Process

def ProcessOperatorProjection(Process):
    Dim = int(np.sqrt(Process.shape[0]))
    Choi = PositiveOperatorProjection( Process2Choi(Process),1 )
    t0 = PositiveMatrix2CholeskyVector( Choi )
    fun = lambda t : la.norm( CholeskyVector2PositiveMatrix(t,Dim).flatten() - Choi.flatten(),  )
    const = lambda t : la.norm( PartialTrace( CholeskyVector2PositiveMatrix(t,Dim) ,[Dim,Dim], 0) - np.eye(Dim) ).flatten() 
    constraints = ({'type': 'eq', 'fun': const })
    t = minimize( fun, t0 , constraints = constraints )
    Choi = CholeskyVector2PositiveMatrix(t.x,Dim)
    return Process2Choi(Choi)

def PartialTrace(rho,Systems,Subsystem):
    #Partial Trace, only works for bipartite systems
    rho = rho.reshape(Systems+Systems).transpose(0,2,1,3).reshape(np.array(Systems)**2)
    if Subsystem == 0:
        rho = ( np.eye(Systems[Subsystem]).reshape(1,-1)@rho  ).flatten()
    elif Subsystem == 1:
        rho = ( rho @ (np.eye(Systems[Subsystem]).reshape(-1,1) ) ).flatten()
    rho = rho.reshape( 2*[ int(np.sqrt(rho.size)) ] )
    
#     if Subsystem == 0:
#         rho = np.trace(rho.reshape(Systems+Systems), axis1=0, axis2=2)
#     elif Subsystem == 1:
#         rho = np.trace(rho.reshape(Systems+Systems), axis1=1, axis2=3)
    return rho

def MaximumLikelihoodProcessTomography( States, Measurements, counts, Guess = [] , Func = 1, vectorized=False ):
    if vectorized == False:
        States = VectorizeVectors(States)
        Measurements = VectorizeVectors(Measurements)  
    Dim = int(np.sqrt(States[:,0].size))
    Num = counts[:,0].size
    if not Guess:
        Guess = ProcessOperatorProjection( LinearProcessTomography( States, Measurements, counts, True) )
    t_guess = PositiveMatrix2CholeskyVector( Process2Choi(Guess) )
    counts_th = lambda t : np.real( Measurements.conj().T@Process2Choi( CholeskyVector2PositiveMatrix( t ) )@States )  
    fun = lambda t : LikelihoodFunction(counts_th(t),counts,Func)
    constraints = ({'type': 'eq', 'fun': lambda t:  la.norm( PartialTrace( CholeskyVector2PositiveMatrix( t , Dim ) ,[Dim,Dim], 0) - np.eye(Dim) ) })
    results = minimize( fun, t_guess, bounds = tuple(len(t_guess)*[(-1,1)]), method = 'SLSQP', constraints = constraints)  
    t = results.x
    result = Results()
    result.measurement = Process2Choi( CholeskyVector2PositiveMatrix ( t, Dim) )
    result.fun = fun(t) 
    result.entropy = LikelihoodFunction( counts, counts, Func )

    return result


# In[6]:


def LinearCompleteDetectorTomography(States, Measurements, counts, vectorized = False ):
    if vectorized == False:
        States = VectorizeVectors(States)
        Measurements = VectorizeVectors(Measurements)
    Process = np.zeros( [States.shape[0],States.shape[0],counts.shape[2]], dtype=complex )
    for k in range(counts.shape[2]):
        Process[:,:,k] = la.pinv(Measurements.conj().T)@( counts[:,:,k] )@la.pinv(States.T).T
    return Process

def CompleteDetectorOperatorProjection(Choiv_in):
    Num = Choiv_in.shape[2]
    Choiv_out = np.array( [ Process2Choi( PositiveOperatorProjection(Process2Choi(Choiv_in[:,:,k]), 2 ) ) for k in range(Num) ] ).transpose([1,2,0])
    return Choiv_out

def MaximumLikelihoodCompleteDetectorTomography( States, Measurements, Probs_ex , Func = 0, vectorized=False , Pi = None, out = 0 ):
    
    if vectorized == False:
        States = VectorizeVectors(States)
        Measurements = VectorizeVectors(Measurements)
        
    Dim = int(np.sqrt(States.shape[0]))
    Num = Probs_ex.shape[2]

    povm_guess = np.zeros( (Dim,Dim,Num) )
    povm_guess[ list(range(Dim)), list(range(Dim)), list(range(Num))  ] = 1
    povm_guess = povm_guess.reshape(Dim**2, Num )

    if Pi is None:
        Probs_0   = np.sum( Probs_ex , 0).T
        results_0 = MaximumLikelihoodDetectorTomography( States, Probs_0, vectorized = True , Func = Func )
        Pi        = results_0.measurement
        
    Choiv = []
    funs  = []
    funs0 = []
    entropies = [  ]

    for k in range(Num):
        Choiv_Lin  = LinearProcessTomography( States, Measurements, Probs_ex[:,:,k], vectorized = True )
        Y_guess    = PositiveOperatorProjection( Process2Choi( Choiv_Lin ), 2 )
        # Y_guess   = np.zeros((Dim**2,Dim**2))
        # Y_guess[ (Dim+1)*k, (Dim+1)*k] = 1
        t_guess   = PositiveMatrix2CholeskyVector( Y_guess )

        Probs_th  = lambda t : np.real( Measurements.conj().T @ Process2Choi( CholeskyVector2PositiveMatrix( t ) )@States )  
        fun       = lambda t : LikelihoodFunction( Probs_th(t), Probs_ex[:,:,k], Func )
        con       = lambda t:  la.norm( PartialTrace( CholeskyVector2PositiveMatrix( t ),
                                            [Dim,Dim], 0).T.flatten() - Pi[:,k].flatten() )**2    
        
        constraints = ({'type': 'eq', 'fun': con })
        results     = minimize( fun, t_guess, constraints = constraints, method = 'SLSQP', 
                                options={'maxiter': 250, 'ftol': 1e-07} )  
        t           = results.x   
        Choiv.append( Process2Choi( CholeskyVector2PositiveMatrix( t ) ) ) 
        # funs.append( fun(t) )
        entropies.append( LikelihoodFunction( Probs_ex[:,:,k], Probs_ex[:,:,k], Func ) )

    norm = np.trace( Process2Choi( np.sum( Choiv, 0 ) ) )
    for k in range(Num):
        Choiv[k] = Dim * Choiv[k] / norm
        t = PositiveMatrix2CholeskyVector( Process2Choi(Choiv[k]) )
        funs.append( fun(t) )

    results = Results()
    results.measurement         = Pi
    results.measurement_process = Choiv
    results.funs                = funs
    results.funs0               = funs0
    results.entropies           = entropies

    return results

    # if out == 0 :
    #     return Choiv
    # elif out == 1:
    #     return Pi, Choiv


# In[7]:


def PauliMatrices(k):
    if k==0:
        M=np.eye(2)
    if k==1:
        M = np.array([0,1,1,0]).reshape([2,2])
    elif k==2:
        M = np.array([0,-1j,1j,0]).reshape([2,2])
    elif k==3:
        M = np.array([1,0,0,-1]).reshape([2,2])
    return M

def Onesij(Dim,i,j):
    A = np.zeros([Dim,Dim],dtype=complex)
    A[i,j] = 1
    return A


# In[8]:


def LinearGateSetTomography(counts, rho_tarjet, Pi_tarjet, Gates_tarjet ):
    sz = counts.shape
    N_gates = sz[0]
    N_states = sz[1]
    N_Measures = sz[2]
    N_outcomes = sz[3]
    P0 = counts[0,:,:,:].reshape([N_states,-1]).T
    Gates = N_gates*[None]
    for k in range(N_gates):
        Gates[k] = la.pinv(P0)@counts[k,:,:,:].reshape([N_states,-1]).T
    Gates = np.array(Gates)
    Q_States = counts[0,:,0,:]
    Q_Measures = counts[0,0,:,:].T/np.sum(counts[0,0,:,:].T,0)
    
    State = la.pinv(P0)@Q_States.flatten()
    Detector = Q_Measures.T
    
    t0 = Complex2Real(np.array([Gates_tarjet[i,:,:]@rho_tarjet for i in range(4)]).T.flatten()).flatten()
    B_t = lambda t : Real2Complex(t.reshape([2,-1])).reshape(2*[State.shape[0]])
    fun = lambda t : GaugeFix_Fun(t,State,Detector,Gates, rho_tarjet,Pi_tarjet,Gates_tarjet,B_t)
    results = least_squares( fun, t0 )  
    B = B_t( results.x )
    
    return B@State, la.inv(B).T.conj()@Detector, B@Gates@la.inv(B)

def MaximumLikelihoodGateSetTomography(counts, rho_tarjet, Pi_tarjet, Gamma_tarjet, gauge='gate_set' ):
    
    State, Detector, Gates = LinearGateSetTomography( counts, rho_tarjet, Pi_tarjet, Gamma_tarjet )
    
    Dim = int(np.sqrt(len(State)))
    N_outcomes = Detector.shape[1]
    N_Gates = Gates.shape[0]
    
    t0_state    = PositiveMatrix2CholeskyVector( PositiveOperatorProjection( State.reshape(Dim,Dim), 0) ) #size Dim**2 
    t0_Detector = np.array([PositiveMatrix2CholeskyVector( PositiveOperatorProjection( Detector[:,k].reshape(Dim,Dim), 2 ) ) for k in range(N_outcomes)  ]) #Size Dim**2xN_outcomes
    t0_Gates    = np.array([PositiveMatrix2CholeskyVector( PositiveOperatorProjection( Kron2Outer(Gates[k,:,:], [Dim,Dim] ), 2 )) for k in range(N_Gates)    ]) #size Dim**2xN_Gates
    t0 = np.concatenate((t0_state,t0_Detector.flatten(),t0_Gates.flatten()), axis=0)
    
    fun = lambda t : LikelihoodFunction( Counts_GQT(t,Dim,N_outcomes,N_Gates), counts, 0 )
    con = lambda t : Constraints_GSQT( t, Dim, N_outcomes, N_Gates )
        
    results = minimize( fun, t0, constraints = ({'type': 'eq', 'fun': con }), 
                        method = 'SLSQP', options={'maxiter': 250, 'ftol': 1e-06}  ) # , bounds = tuple(len(t0)*[(-1,1)])
    t = results.x

    results         = Results()   
    results.fun     = fun( t )
    results.entropy = LikelihoodFunction( counts, counts, 0 )

    #if fun(t0) < fun(t):
    #    t = t0
    
    t          = t.reshape(-1,Dim**2)
    t_state    = t[0,:]
    t_Detector = t[1:1+N_outcomes,:]
    t_Gates    = t[1+N_outcomes:,:].reshape(-1,Dim**4)
    State      = CholeskyVector2PositiveMatrix( t_state, 1 ).flatten()
    Detector   = np.array([ CholeskyVector2PositiveMatrix(t_Detector[k,:]).flatten() for k in range(N_outcomes)]).T.reshape(Dim**2,N_outcomes)
    Gates      = np.array([ Process2Choi( CholeskyVector2PositiveMatrix(t_Gates[k,:]) ) for k in range(N_Gates) ] )   
    
    t0 = Complex2Real(np.eye(Dim,dtype=complex).flatten()).flatten()
    B_t = lambda t : np.kron( UnitaryProjection( Real2Complex(t.reshape([2,-1])).reshape(2*[Dim]) ), UnitaryProjection( Real2Complex(t.reshape([2,-1])).reshape(2*[Dim]) ).conj() )
    fun = lambda t : GaugeFix_Fun( t, State, Detector, Gates, rho_tarjet, Pi_tarjet, Gamma_tarjet, B_t, gauge )
    results_ls = least_squares( fun, t0, bounds= (-1,1) )  
    B = B_t(results_ls.x)
  
    State    = B@State
    Detector = la.inv(B).conj().T@Detector
    Gates    = B@Gates@la.inv(B)
    
    for j in range(N_Gates):
        Gates[j] = Dim * Gates[j] / np.trace( Process2Choi( Gates[j] ) )
    
    
    results.state       = State
    results.measurement = Detector
    results.process     = Gates

    return results
    
def UnitaryProjection(Z):
    U = Z@la.fractional_matrix_power( Z.conj().T@Z , -0.5 )
    return U
    
def Counts_GQT(t,Dim,N_outcomes,N_Gates):
    t = t.reshape(-1,Dim**2)
    t_state    = t[0,:]
    t_Detector = t[1:1+N_outcomes,:]
    t_Gates    = t[1+N_outcomes:,:].reshape(-1,Dim**4)
    
    rho      = CholeskyVector2PositiveMatrix( t_state, 1 ).flatten()
    # Detector = np.array([ CholeskyVector2PositiveMatrix(t_Detector[k,:]).flatten() for k in range(N_outcomes)]).T
    Detector = POVM_from_t(t_Detector, Dim, Dim )
    Gates    = np.array( [ Process2Choi(CholeskyVector2PositiveMatrix(t_Gates[k,:])) for k in range(N_Gates) ] )
    
    Counts   = np.array( [[[ Detector.conj().T@Gates[j,:,:]@Gates[k,:,:]@Gates[i,:,:]@rho for j in range(N_Gates)]for i in range(N_Gates)]for k in range(N_Gates)] )
    
    return np.real( Counts )

def Constraints_GSQT(t,Dim,N_outcomes,N_Gates):
    t = t.reshape(-1,Dim**2)
    t_state = t[0,:]
    t_Detector = t[1:1+N_outcomes,:]
    t_Gates = t[1+N_outcomes:,:].reshape(-1,Dim**4)
    f1 = la.norm( np.sum(POVM_from_t(t_Detector, Dim, Dim ),1).flatten() - np.eye(Dim).flatten() )**2
    f2 = la.norm( np.array([ PartialTrace( CholeskyVector2PositiveMatrix(t_Gates[k,:]) ,[Dim,Dim], 0).flatten() - np.eye(Dim).flatten() for k in range(N_Gates) ]) )**2
    return np.append(f2,f1)

def GaugeFix_Fun( t, rho, Pi, Gamma, rho_tarjet, Pi_tarjet, Gamma_tarjet, B_t, gauge='gate_set'):
    B = B_t(t)
    invB = la.inv(B)
    
    if gauge=='gate_set':
        f1 = ( B@rho - rho_tarjet ).flatten()
        f2 = ( invB.conj().T@Pi - Pi_tarjet ).flatten()
        f3 = np.array( [ B@Gamma[k,:,:]@invB - Gamma_tarjet[k,:,:,] for k in range(1,4)] ).flatten()
        f  = Complex2Real( np.concatenate([f1,f2,f3],axis=0) ).flatten()
    elif gauge=='detector':
        f = Complex2Real( invB.conj().T@Pi - Pi_tarjet ).flatten()
        
    return f

