import numpy as np
import matplotlib.pyplot as plt
import QuantumTomography as qt
from qiskit import QuantumCircuit
from qiskit.providers.aer.noise import NoiseModel, thermal_relaxation_error
from qiskit.visualization import plot_gate_map
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib import ticker
import seaborn as sns
import networkx as nx
color_map = sns.cubehelix_palette(reverse=True, as_cmap=True)
from joblib import Parallel, delayed
import scipy.sparse as sp

def get_noise( job ):
    readout_error = [ job.properties().readout_error(j) for j in range(7)  ]
    T1 = [ job.properties().t1(j) for j in range(7)  ]
    return readout_error, T1

def plot_error_map( backend, single_gate_errors, double_gate_errors ):

    single_gate_errors = 100*single_gate_errors
    single_norm = matplotlib.colors.Normalize( vmin=min(single_gate_errors), vmax=max(single_gate_errors))
    q_colors = [color_map(single_norm(err)) for err in single_gate_errors]
    
    double_gate_errors = 100*double_gate_errors
    double_norm = matplotlib.colors.Normalize( vmin=min(double_gate_errors), vmax=max(double_gate_errors))
    l_colors = [color_map(double_norm(err)) for err in double_gate_errors]
    
    figsize=(12, 9)
    fig = plt.figure(figsize=figsize)
    gridspec.GridSpec(nrows=2, ncols=3)

    grid_spec = gridspec.GridSpec(
        12, 12, height_ratios=[1] * 11 + [0.5], width_ratios=[2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
    )

    left_ax = plt.subplot(grid_spec[2:10, :1])
    main_ax = plt.subplot(grid_spec[:11, 1:11])
    right_ax = plt.subplot(grid_spec[2:10, 11:])
    bleft_ax = plt.subplot(grid_spec[-1, :5])
    bright_ax = plt.subplot(grid_spec[-1, 7:])

    plot_gate_map(backend, qubit_color=q_colors, line_color=l_colors, line_width=5,
                plot_directed=False,
                ax=main_ax )

    main_ax.axis("off")
    main_ax.set_aspect(1)

    single_cb = matplotlib.colorbar.ColorbarBase(
                bleft_ax, cmap=color_map, norm=single_norm, orientation="horizontal"
            )
    tick_locator = ticker.MaxNLocator(nbins=5)
    single_cb.locator = tick_locator
    single_cb.update_ticks()
    single_cb.update_ticks()
    bleft_ax.set_title(f"H error rate")

    cx_cb = matplotlib.colorbar.ColorbarBase(
                bright_ax, cmap=color_map, norm=double_norm, orientation="horizontal"
            )
    tick_locator = ticker.MaxNLocator(nbins=5)
    cx_cb.locator = tick_locator
    cx_cb.update_ticks()
    bright_ax.set_title(f"CNOT error rate")
    
    return fig

def get_backend_conectivity(backend):
	"""
	Get the connected qubit of q backend. Has to be a quantum computer.

	Parameters
	----------
	backend: qiskit.backend

	Return
	------
	connexions: (list)
		List with the connected qubits
	"""
	defaults = backend.defaults()
	connexions = [indx for indx in defaults.instruction_schedule_map.qubits_with_instruction('cx')]
	return connexions


def tomographic_gate_set(n=1):
    """
    Create circuits for perform Pauli tomography of a single qubit.
    """
    
    circ_0 = QuantumCircuit(n)

    circ_x = QuantumCircuit(n)
    circ_x.x(range(n))

    circ_h = QuantumCircuit(n)
    circ_h.h(range(n))

    circ_k = QuantumCircuit(n)
    circ_k.u( np.pi/2, np.pi/2, -np.pi/2, range(n))

    circ_gates = [ circ_0, circ_x, circ_h, circ_k]

    return circ_gates


def tomographic_gate_set_tomography_circuits(n=1):
    
    
    circ_gates = tomographic_gate_set(n)
    circ_gst = [] 

    for circ_j in circ_gates :
        for circ_i in circ_gates :
            for circ_k in circ_gates :
                qc = QuantumCircuit(n)
                qc.compose( circ_i, range(n), inplace=True )
                qc.compose( circ_j, range(n), inplace=True )
                qc.compose( circ_k, range(n), inplace=True )
                qc.measure_all()
                circ_gst.append( qc )
                
    return circ_gst


class tomographic_gate_set_tomography_fitter:
    
    def __init__( self, results, circ_gst,resampling=0 ):
        self._n = circ_gst[0].num_qubits
        self._probs = []
        for qc in circ_gst:
            temp  = results.get_counts(qc)
            p = dict2array( temp, self._n, resampling=resampling )
            p = p / np.sum( p )
            self._probs.append( p )
        
        del results, circ_gst, p
        self._probs = np.array(self._probs).reshape([64]+self._n*[2])
        
    def fit( self ):
        
        rho = np.array([1,0,0,0])
        Detector = np.array([ [1,0], [0,0], [0,0], [0,1] ])
        I = np.kron( np.eye(2), np.eye(2) )
        X = np.kron( qt.PauliMatrices(1), qt.PauliMatrices(1) )
        H = np.kron( qt.PauliMatrices(1) + qt.PauliMatrices(3), qt.PauliMatrices(1) + qt.PauliMatrices(3) )/2
        K = np.kron( qt.PauliMatrices(0) + 1j*qt.PauliMatrices(1), qt.PauliMatrices(0) - 1j*qt.PauliMatrices(1) )/2
        Gates = np.array([I,X,H,K])

        rho_hat_all    = []
        Detetor_hat_all = []
        Gates_hat_all   = []
        for m in range(self._n) :
            ran = tuple( [ idx for idx in range(1,self._n+1) if idx != m+1] )
            probs = np.sum( self._probs, axis = ran ).reshape(4,4,4,2)
            rho_hat, Detetor_hat, Gates_hat = qt.MaximumLikelihoodGateSetTomography( probs, rho, Detector, Gates,'detector')
            rho_hat_all.append( rho_hat )
            Detetor_hat_all.append( Detetor_hat )
            Gates_hat_all.append( Gates_hat )
        
        return [rho_hat_all, Detetor_hat_all, Gates_hat_all]
            

def measurement_process_tomography_circuits(n=1, circ_detector=None):
    
    circ_0, circ_x, circ_h, circ_k = tomographic_gate_set(n)
    circs_mpt = []
    
    if circ_detector is None:
        for circ_hk in [circ_0, circ_h, circ_k ]:
            for circ_0x in  [ circ_0, circ_x ]:
                for circ_xyz in [circ_0, circ_h, circ_k ]:
                    qc = QuantumCircuit( n, 2*n )
                    qc.compose( circ_0x, range(n), inplace=True )
                    qc.compose( circ_hk, range(n), inplace=True )
                    qc.barrier()
                    qc.measure( range(n), range(n) )
                    qc.barrier()
                    qc.compose( circ_xyz, range(n), inplace=True )
                    qc.barrier()
                    qc.measure( range(n), range(n,2*n)  )
                    circs_mpt.append( qc )
                    
    else:
        qb = circ_detector.num_qubits
        cb = circ_detector.num_clbits
        qc = QuantumCircuit( n*qb, n*cb )
        for j in range(n):
            qc.compose(circ_detector, 
                       qubits=range(j*qb,(j+1)*qb),
                       clbits=range(j*cb,(j+1)*cb),
                       inplace=True 
                      )
        circ_detector = qc
        del qc
        for circ_hk in [circ_0, circ_h, circ_k ]:
            for circ_0x in  [ circ_0, circ_x ]:
                for circ_xyz in [circ_0, circ_h, circ_k ]:
                    qc = QuantumCircuit( n*qb, 2*n*cb )
                    qc.compose( circ_0x, range(0,qb*n,qb), inplace=True )
                    qc.compose( circ_hk, range(0,qb*n,qb), inplace=True )
                    qc.barrier()
                    qc.compose( circ_detector, qubits=range(qb*n), clbits=range(cb*n), inplace=True )
                    qc.barrier()
                    qc.compose( circ_xyz, range(0,qb*n,qb), inplace=True )
                    qc.barrier()
                    qc.compose( circ_detector, qubits=range(qb*n), clbits=range(cb*n,cb*2*n), inplace=True )
                    circs_mpt.append( qc )
                    
    return circs_mpt


def measurement_process_tomography_circuits_new(n=1, circ_detector=None):
    
    if circ_detector is None:
        circ_detector = QuantumCircuit( n, n )
        circ_detector.measure( range(n), range(n) )
        qb = 1
        cb = 1
    else:
        qb = circ_detector.num_qubits
        cb = circ_detector.num_clbits
        print(qb,cb)
        qc = QuantumCircuit( n*qb, n*cb )
        for j in range(n):
            qc.compose(circ_detector, 
                       qubits=range(j*qb,(j+1)*qb),
                       clbits=range(j*cb,(j+1)*cb),
                       inplace=True 
                      )
        circ_detector = qc
        del qc
    
    circ_0, circ_x, circ_h, circ_k = tomographic_gate_set(n)
    circs_mpt = []
    for circ_hk in [circ_0, circ_h, circ_k ]:
        for circ_0x in  [ circ_0, circ_x ]:
            for circ_xyz in [circ_0, circ_h, circ_k ]:
                qc = QuantumCircuit( n*qb, 2*n*cb )
                qc.compose( circ_0x, range(0,qb*n,qb), inplace=True )
                qc.compose( circ_hk, range(0,qb*n,qb), inplace=True )
                qc.barrier()
                qc.compose( circ_detector, qubits=range(qb*n), clbits=range(cb*n), inplace=True )
                qc.barrier()
                qc.compose( circ_xyz, range(0,qb*n,qb), inplace=True )
                qc.barrier()
                qc.compose( circ_detector, qubits=range(qb*n), clbits=range(cb*n,cb*2*n), inplace=True )
                circs_mpt.append( qc )
    return circs_mpt

            
class measurement_process_tomography_fitter:           
            
    def __init__( self, results, circs_mpt, gate_set = None, resampling=0 ):
        self._n = int( 0.5*circs_mpt[0].num_clbits  ) 
        self._probs = []
        for qc in circs_mpt:
            counts = results.get_counts(qc)
            probs_loop = dict2array(counts, 2*self._n, resampling=resampling ) 
            probs_loop = probs_loop / np.sum(probs_loop)
            self._probs.append(probs_loop)   
        del results, circs_mpt, probs_loop
        self._probs = np.array(self._probs).reshape([6,3]+(2*self._n)*[2]  )/3
        
        if gate_set is None :
            self._gateset = False 
            self._states = np.array( [ [[1,0],[0,0]],
                          [[0,0],[0,1]],
                          [[1/2,1/2],[1/2,1/2]],
                          [[1/2,-1/2],[-1/2,1/2]],
                          [[1/2,-1j/2],[1j/2,1/2]],
                          [[1/2,1j/2],[-1j/2,1/2]],
                         ]).reshape( 6,4 ).T

            self._measurements = self._states / 3   
        else :
            self._gateset = True
            self._states, self._measurements = gate_set
            
    def fit( self, out=0 ):
        
        if self._gateset is False :

            Υ_hat_all = []
            for m in range(self._n) :
                ran = tuple( [ idx for idx in range(2,2*self._n+2) if idx != m+2 and idx != self._n+m+2] )
                probs = np.sum( self._probs, axis = ran ).transpose(0,1,3,2).reshape(6,6,2).transpose(1,0,2)
                Υ_hat = qt.MaximumLikelihoodCompleteDetectorTomography( self._states, 
                                                                       self._measurements, 
                                                                       probs , Func = 0, 
                                                                       vectorized=True, out=out )
                Υ_hat_all.append( Υ_hat)    
                
        elif self._gateset is True :
            
            Υ_hat_all = []
            for m in range(self._n) :
                ran = tuple( [ idx for idx in range(2,2*self._n+2) if idx != m+2 and idx != self._n+m+2] )
                probs = np.sum( self._probs, axis = ran ).transpose(0,1,3,2).reshape(6,6,2).transpose(1,0,2)
                Υ_hat = qt.MaximumLikelihoodCompleteDetectorTomography( self._states[m], 
                                                                       self._measurements[m], 
                                                                       probs, 
                                                                       Func = 0, 
                                                                       vectorized=True, out=out  )
                Υ_hat_all.append( Υ_hat)    
        
        if len(Υ_hat_all) == 1 :
            return Υ_hat_all[0]
        else:
            return Υ_hat_all

    
def dict2array(counts, n_qubits, sparse=False, resampling=0 ):
    if sparse is False:
        p = np.zeros( 2**n_qubits )
    else:
        p = sp.lil_matrix( (2**n_qubits,1) )
    for idx in counts :
        p[ int(idx[::-1],2) ] = counts[idx]
     
    if resampling > 0 :
        p = np.random.multinomial( resampling, p/np.sum(p) )
        
    if sparse is True:
        p = sp2.COO.from_scipy_sparse(p)    
        
    return p.reshape( n_qubits*[2] )

    
def decoherence_noise( T1=5e3, T2=200e3 ):

    # T1 and T2 values for qubits 0-3
    T1s = np.random.normal( T1, np.sqrt(T1), 7) # Sampled from normal distribution mean 50 microsec
    T2s = np.random.normal( T2, np.sqrt(T2), 7)  # Sampled from normal distribution mean 50 microsec
    
    # Truncate random T1s <= 0
    T1s[T1s<0]=0
    
    # Truncate random T2s <= T1s
    T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(7)])

    # Instruction times (in nanoseconds)
    time_u1 = 0   # virtual gate
    time_u2 = 50  # (single X90 pulse)
    time_u3 = 100 # (two X90 pulses)
    time_cx = 300
    time_reset = 1000  # 1 microsecond
    time_measure = 1000 # 1 microsecond

    # QuantumError objects
    errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                    for t1, t2 in zip(T1s, T2s)]
    errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                      for t1, t2 in zip(T1s, T2s)]
    errors_u1  = [thermal_relaxation_error(t1, t2, time_u1)
                  for t1, t2 in zip(T1s, T2s)]
    errors_u2  = [thermal_relaxation_error(t1, t2, time_u2)
                  for t1, t2 in zip(T1s, T2s)]
    errors_u3  = [thermal_relaxation_error(t1, t2, time_u3)
                  for t1, t2 in zip(T1s, T2s)]
    errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
                 thermal_relaxation_error(t1b, t2b, time_cx))
                  for t1a, t2a in zip(T1s, T2s)]
                   for t1b, t2b in zip(T1s, T2s)]

    # Add errors to noise model
    noise_thermal = NoiseModel()
    for j in range(7):
        noise_thermal.add_quantum_error(errors_reset[j], "reset", [j])
        noise_thermal.add_quantum_error(errors_measure[j], "measure", [j])
        noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
        noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
        noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
        for k in range(4):
            noise_thermal.add_quantum_error(errors_cx[j][k], "cx", [j, k])
            
    return noise_thermal
            
            
def measurement_process_tomography_circuits_v2( n=1, p=1 ):
    
    circ_0, circ_x, circ_h, circ_k = tomographic_gate_set(p)
    
    circs_state_s = []
    for circ_hk in [circ_0, circ_h, circ_k ]:
        for circ_0x in  [ circ_0, circ_x ]:
            qc = QuantumCircuit(p)
            qc.compose( circ_0x, range(p), inplace=True )
            qc.compose( circ_hk, range(p), inplace=True )
            circs_state_s.append( qc )
    
    circs_measure_s = [circ_0, circ_h, circ_k ]    
    
    circ_state = []
    for j in range(n):
        list_qubits = range(j,n*p,n)
        if j == 0 :
            for circ in circs_state_s:
                qc0 = QuantumCircuit( n*p )
                qc0.compose(circ.copy(), qubits=list_qubits, inplace=True)
                circ_state.append(qc0)
        else:
            circ_loop = circ_state.copy()
            circ_state = []
            for qc1 in circ_loop:
                for circ in circs_state_s:
                    qc2 = qc1.compose(circ.copy(), qubits=list_qubits)
                    circ_state.append(qc2)

    circ_measure = []
    for j in range(n):
        list_qubits = range(j,n*p,n)
        if j == 0 :
            for circ in circs_measure_s:
                qc0 = QuantumCircuit( n*p )
                qc0.compose(circ.copy(), qubits=list_qubits, inplace=True)
                circ_measure.append(qc0)
        else:
            circ_loop = circ_measure.copy()
            circ_measure = []
            for qc1 in circ_loop:
                for circ in circs_measure_s:
                    qc2 = qc1.compose(circ.copy(), qubits=list_qubits)
                    circ_measure.append(qc2)     
    
    circs_mpt = []
    for i in range(6**n):
        for j in range(3**2):        
            qc = QuantumCircuit( n*p, 2*n*p )
            qc.compose( circ_state[i], qubits=range(n*p), inplace=True )
            qc.measure( range(n*p), range(n*p) )
            qc.compose( circ_measure[j], qubits=range(n*p), inplace=True )
            qc.measure( range(n*p), range(n*p,2*n*p) )
            circs_mpt.append( qc )
               
    return circs_mpt
            
class measurement_process_tomography_fitter_v2:           
            
    def __init__( self, results, circs_mpt, gate_set = None, resampling=0 ):         
        
        self._n = int( np.log(len(circs_mpt))/np.log(18) )
        self._p = int( .5*circs_mpt[0].num_clbits / self._n )
        
        self._probs = []
        for qc in circs_mpt :
            counts  = results.get_counts(qc)
            probs_loop = dict2array(counts, 2*self._n*self._p, 
                                    resampling=resampling)
            probs_loop = probs_loop / np.sum(probs_loop)
            self._probs.append(probs_loop)
        del probs_loop, circs_mpt, results
        
        self._probs = np.array(self._probs).reshape([6**self._n,
                                                     3**self._n]
                                                    +(2*self._p)*[2**self._n]
                                                    )/3**self._n        
            
        if gate_set is None :
            self._gateset = False
            states_s = np.array( [ [[1,0],[0,0]],
                          [[0,0],[0,1]],
                          [[1/2,1/2],[1/2,1/2]],
                          [[1/2,-1/2],[-1/2,1/2]],
                          [[1/2,-1j/2],[1j/2,1/2]],
                          [[1/2,1j/2],[-1j/2,1/2]],
                         ])

            measures_s = states_s.reshape(3,2,2,2)

            states = []
            for s1 in states_s:
                for s2 in states_s:
                    state_temp = np.kron( s1, s2 )
                    states.append( state_temp.flatten() )
            self._states = np.array(states).T

            measures = []
            for r1 in range(3):
                for r2 in range(3):
                    for s1 in range(2):
                        for s2 in range(2):
                            measures_temp = np.kron( measures_s[r1,s1], measures_s[r2,s2] )
                            measures.append( measures_temp.flatten() )   
            self._measures = np.array(measures).T/9    
            
            
        else :
            self._gateset = True
            states_s, measures_s = gate_set 
            measures_s = np.array(measures_s).reshape(self._n*self._p,4,3,2).transpose(0,2,3,1)
            
            self._states = []
            self._measures = []
            
            for m in range(self._p):
                
                states = []
                for s1 in range(6):
                    for s2 in range(6):
                        state_temp = qt.Outer2Kron( np.kron( states_s[m*self._n][:,s1], states_s[m*self._n+1][:,s2] ), [2,2] )
                        states.append( state_temp.flatten() )
                self._states.append( np.array(states).T )


                measures = []
                for r1 in range(3):
                    for r2 in range(3):
                        for s1 in range(2):
                            for s2 in range(2):
                                measures_temp = qt.Outer2Kron( np.kron( measures_s[m*self._n,r1,s1], measures_s[m*self._n+1,r2,s2] ), [2,2] )
                                measures.append( measures_temp.flatten() )   
                self._measures.append( np.array(measures).T ) 
            
            
    def fit( self, out = 0 ):
        
        if self._gateset is False :
            Υ_hat_all = []
            for m in range(self._p) :
                ran = tuple( [ idx for idx in range(2,2*self._p+2) if idx != m+2 and idx != self._p+m+2] )
                probs_loop = np.sum( self._probs, axis = ran ).transpose(0,1,3,2).reshape(6**self._n,6**self._n,2**self._n).transpose(1,0,2)
                self._probs_loop = probs_loop
                Υ_hat = qt.MaximumLikelihoodCompleteDetectorTomography( self._states, 
                                                                       self._measures, 
                                                                       probs_loop, 
                                                                       Func = 0, 
                                                                       vectorized=True, 
                                                                       out=out )
                Υ_hat_all.append( Υ_hat) 
        
        elif self._gateset is True :
            Υ_hat_all = []
            for m in range(self._p) :
                ran = tuple( [ idx for idx in range(2,2*self._p+2) if idx != m+2 and idx != self._p+m+2] )
                probs_loop = np.sum( self._probs, axis = ran ).transpose(0,1,3,2).reshape(6**self._n,6**self._n,2**self._n).transpose(1,0,2)
                Υ_hat = qt.MaximumLikelihoodCompleteDetectorTomography( self._states[m], self._measures[m], 
                                                                       probs_loop, Func = 0, vectorized=True, out=out)
                Υ_hat_all.append( Υ_hat ) 
        
        return Υ_hat_all

    
############ Quantities ###################

def readout_fidelity( Pi ):
    d, N = Pi.shape
    d = int(np.sqrt(d))
    f = 0.
    for n in range(N):
        f += Pi[:,n].reshape(d,d)[n,n]/N
    return np.real( f )  

def qnd_fidelity( choi ):
    N = len(choi)
    d = int(np.sqrt(choi[0].shape[0]))
    f = 0
    for n in range(N):
        f += choi[n][(1+d)*n,(1+d)*n]/N
    return np.real( f )     
    
def destructiveness( chois ):
    choi = np.sum( chois, axis=0 )
    d = int(np.sqrt(choi.shape[0]))
    if d == 2:
        O = np.array([1,0,0,-1])/np.sqrt(2)
        D = np.linalg.norm( O - choi.T.conj()@O )/np.sqrt(8)
    else:
        P = np.eye(d)
        Bs = np.zeros((d**2,d),dtype=complex)
        for k in range(d):
            pp = np.kron( P[:,k], P[:,k] )
            Bs[:,k] = pp - choi.T.conj()@pp 
        B = Bs.T.conj()@Bs
        vals, vecs = np.linalg.eigh(B)
        D = 0.5 * np.sqrt( np.max(vals)/2 )
    return D    
    
def Quantities( Pi, choi ):
    
    f = readout_fidelity( Pi )
    q = qnd_fidelity( choi )
    d = 1 - destructiveness( choi )
    
    return f, q, d
            
def Kron_Choi( Choi_1, Choi_2 ):
    Y0 = [] 
    for i in range( len(Choi_1) ):
        for j in range(len(Choi_2)):
            Y_loop = np.kron( Choi_1[i],  Choi_2[j]) 
            Y_loop =  Y_loop.reshape(8*[2]).transpose(0,2,1,3,4,6,5,7).reshape(16,16) 
            Y0.append( Y_loop )
    return Y0

def Cross_QNDness( Choi_single_1, Choi_single_2, Choi_double  ):
    Y0 = [ qt.Process2Choi( A )/4 for A in Kron_Choi( Choi_single_1, Choi_single_2 )]
    Y1 = [ qt.Process2Choi( A )/4 for A in Choi_double]
    # f = 0
    # for i in range(4):
    #     f += qt.Fidelity( Y0[i], Y1[i] ) 
    f = np.linalg.norm( np.array(Y0) - np.array(Y1) ) / 4
    return f

def Cross_Fidelity( Pi_single_1, Pi_single_2, Pi_double  ):
    Pi0 = [ np.kron(A,B)/4 for A in Pi_single_1.reshape(2,2,2).transpose(1,2,0) for B in Pi_single_2.reshape(2,2,2).transpose(1,2,0) ]
    Pi1 = Pi_double.reshape(4,4,4).transpose(1,2,0)/4
    # f = 0
    # for i in range(4):
    #     f += qt.Fidelity( Pi0[i], Pi1[i] )
    f = np.linalg.norm( np.array(Pi0) - np.array(Pi1) )
    return f

def Cross_Quantities( Pi1, Choi1, Pi2, Choi2, Pi12, Choi12 ):
    
    f = Cross_Fidelity( Pi1, Pi2, Pi12 )
    q = Cross_QNDness( Choi1, Choi2, Choi12 )
    
    return f, q

### Plots ###

def sph2cart(r, theta, phi):
    '''spherical to Cartesian transformation.'''
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def sphview(ax):
    '''returns the camera position for 3D axes in spherical coordinates'''
    r = np.square(np.max([ax.get_xlim(), ax.get_ylim()], 1)).sum()
    theta, phi = np.radians((90-ax.elev, ax.azim))
    return r, theta, phi

def getDistances(view, xpos, ypos, dz):
    distances  = []
    for i in range(len(xpos)):
        distance = (xpos[i] - view[0])**2 + (ypos[i] - view[1])**2 + (dz[i] - view[2])**2
        distances.append(np.sqrt(distance))
    return distances

def Bar3D( A , ax = None, xpos=None, ypos=None, zpos=None, dx=None, dy=None, M = 0, **args ):
    
    d = A.shape[0]
    camera = np.array([13.856, -24. ,0])
    
    if xpos is None :
        xpos = np.arange(d) 
    if ypos is None :
        ypos = np.arange(d)
    xpos, ypos = np.meshgrid( xpos, ypos )
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    
    if zpos is None :
        zpos = np.zeros_like(xpos)
    else :
        zpos = zpos.flatten()
    
    if dx is None :
        dx = 0.5 * np.ones_like(xpos)
    else :
        dx = dx * np.ones_like(ypos)
        
    if dy is None :
        dy = 0.5 * np.ones_like(ypos)
    else :
        dy = dy * np.ones_like(ypos)
    
    dz = A.flatten()
    z_order = getDistances(camera, xpos, ypos, zpos)
    
    if ax == None :
        fig = plt.figure()   
        ax  = fig.add_subplot( 1,1,1, projection='3d')  
    maxx    = np.max(z_order) + M
    
#     plt.rc('font', size=15) 
    for i in range(xpos.shape[0]):
        pl = ax.bar3d(xpos[i], ypos[i], zpos[i], 
                      dx[i], dy[i], dz[i], 
                      zsort='max', **args )
        pl._sort_zpos = maxx - z_order[i]
#        ax.set_xticks( [0.25,1.25,2.25,3.25] )
#        ax.set_xticklabels((r'$|gg\rangle$',r'$|ge\rangle$',
#                                r'$|eg\rangle$',r'$|ee\rangle$'))
#        ax.set_yticks( [0.25,1.25,2.25,3.25] )
#        ax.set_yticklabels((r'$\langle gg|$',r'$\langle ge|$',
#                                r'$\langle eg|$',r'$\langle ee|$'))
#         ax.set_title( label, loc='left', fontsize=20, x = 0.1, y=.85)
    ax.set_zlim([0,1])
    return ax            


def Abs_Bars3D(Y, lim=None, horizontal=True ):
    
    if horizontal is True:
        size_x = 1
        size_y = len(Y)
        
    elif horizontal is False:
        size_x = len(Y)
        size_y = 1
        
    fig = plt.figure(figsize=(4*size_y,4*size_x)) 
    for y in range(len(Y)):
        ax  = fig.add_subplot( size_x, size_y, y+1,  projection='3d')
        Bar3D( np.abs( Y[y] ).T, ax=ax ) 
        
        if lim is not None:
            ax.set_zlim(lim)  
    
    return fig


############# Device Tomography ################

            
def device_process_measurement_tomography_circuits( backend, max_qobj=900 ):
    """
    Circuits to perform the process measurement tomography of each pair of connected qubits on a device
    
    In:
        backend
    out:
        circs_all : all the circuits for the tomography.
        circs_pkg : efficient storage of the circuits for execution.
        pkg_idx   : pkg_idx[j] is the position of circs_all[j] on circs_pkg.
    
    """
    
    num_qubits = len( backend.properties().qubits )
    coupling_map = get_backend_conectivity(backend)

    G = nx.Graph()
    G.add_node( range(num_qubits) )
    G.add_edges_from(coupling_map)
    G = nx.generators.line.line_graph(G)
    G_coloring = nx.coloring.greedy_color(G)
    degree = max( G_coloring.values() ) + 1
    parall_qubits = degree*[None]
    for x in G_coloring:
        if parall_qubits[G_coloring[x]] is None:
            parall_qubits[G_coloring[x]] = []
        parall_qubits[G_coloring[x]].append(x)

        
    circs_all = [ tomographic_gate_set_tomography_circuits(num_qubits), measurement_process_tomography_circuits( num_qubits ) ]

    for pairs in parall_qubits :

        p = len(pairs)
        qubits = pairs
        qubits = [item for t in qubits for item in t]
        circ_double = measurement_process_tomography_circuits_v2( 2, p )
        circs = []
        for circ_loop in circ_double:
            circ = QuantumCircuit( num_qubits, 4*p )
            circ.compose(circ_loop, qubits=qubits, inplace=True)
            circs.append( circ )
        circs_all.append( circs )
        
    circs_pkg = []
    circ_temp = []
    pkg_idx = []

    idx = 0
    for circs in circs_all:
        if len(circ_temp)+len(circs)<=max_qobj:
            circ_temp += circs.copy()
            pkg_idx.append( idx ) 
        else :
            circs_pkg.append( circ_temp )
            circ_temp = circs.copy()
            idx += 1
    circs_pkg.append( circ_temp )
    pkg_idx.append( idx ) 
    
    return circs_all, circs_pkg, pkg_idx, parall_qubits
             
            
def device_process_measurement_tomography_fitter( results, circs_all, circs_pkg, pkg_idx, parall_qubits, out=1, resampling=0, paralell=True ):
        
    """"
    Process measurement tomography of all pairs of connected qubits of a device.
    
    In: 
        results: list of qiskit results. This can be the execution of circs_all or circs_pkg. 
                 The position of each list of circuits of circs_all on the list results is specified by pkg_idx.
        circs_all : all the circuits for the tomography.
        circs_pkg : efficient storage of the circuits for execution.
        pkg_idx   : pkg_idx[j] is the position of circs_all[j] on circs_pkg.
    out :
        choi_single : Choi matrices of each single qubit.
        choi_double : Choi matrices of each pair of connected qubits.
        gateset     : gate set of each single qubit.
    """
    
    del circs_pkg
    
    gateset = tomographic_gate_set_tomography_fitter( results[pkg_idx[0]] , circs_all[0], resampling=resampling ).fit()
    
    num_qubits = circs_all[0][0].num_qubits
    
    states_gst= []
    measures_gst = []
    for m in range(num_qubits):
        rho = gateset[0][m]
        Pi  = gateset[1][m]
        Y   = gateset[2][m]
        states_gst_temp   = []
        measures_gst_temp = []
        for v in [ np.eye(4), Y[2], Y[3] ]:
            for u in [ np.eye(4), Y[1] ]:
                states_gst_temp.append( v@u@rho )
            measures_gst_temp.append( v.T.conj()@Pi )    

        states_gst.append( np.array(states_gst_temp).T )
        measures_gst.append( np.array(measures_gst_temp).transpose(1,0,2).reshape(4,-1)/3 )

    states_gst   = np.array( states_gst )
    measures_gst = np.array( measures_gst )

    choi_single = measurement_process_tomography_fitter( results[pkg_idx[0]], circs_all[1], resampling=resampling, gate_set = [states_gst,measures_gst] ).fit( out = out )

    
    if paralell is False:
        choi_double = []
        for k in range(2,len(circs_all)) :
            qubits = np.array(parall_qubits[k-2]).flatten()
            choi_double.append( measurement_process_tomography_fitter_v2( results[pkg_idx[k]],
                                                                         circs_all[k], 
                                                                         gate_set = [ states_gst[qubits], measures_gst[qubits] ],
                                                                         resampling = resampling ).fit( out = out ) )
    elif paralell is True:
        fun_par = lambda k : measurement_process_tomography_fitter_v2( results[pkg_idx[k]], 
                                                                      circs_all[k], 
                                                                      gate_set   = [ states_gst[np.array(parall_qubits[k-2]).flatten()], 
                                                                                    measures_gst[np.array(parall_qubits[k-2]).flatten()] ],
                                                                      resampling=resampling ).fit(out = out)
        choi_double = Parallel(n_jobs=-1)( delayed( fun_par )(k) for k in range(2,len(circs_all)) ) 
    
    return choi_single, choi_double, gateset 
                
            
            

            
            
            
            
            
            
            
            
            
            
