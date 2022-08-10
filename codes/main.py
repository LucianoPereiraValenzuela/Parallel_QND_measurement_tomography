import numpy as np
import QuantumTomography as qt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.aer.noise import NoiseModel, thermal_relaxation_error
from qiskit.visualization import plot_gate_map
from qiskit.result import Result
import networkx as nx
from joblib import Parallel, delayed

def get_noise( job ):
    readout_error = [ job.properties().readout_error(j) for j in range(7)  ]
    T1 = [ job.properties().t1(j) for j in range(7)  ]
    return readout_error, T1
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
def get_backend_conectivity(backend):
	"""
	Get the connected qubit of a backend. Has to be a quantum computer.

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


def marginal_counts_dictionary( counts , idx ):
    
    if len(idx) == 0 :
        marginal_counts = counts
    else:
        marginal_counts = {}
        for key in counts:
            key_short = key.replace(' ','')
            sub_key = ''
            for k in idx:
                sub_key += key_short[::-1][k]
            sub_key = sub_key[::-1]
            if sub_key in marginal_counts:
                marginal_counts[sub_key] += counts[key]
            else:
                marginal_counts[sub_key] = counts[key]
                
    return marginal_counts


def dict2array(counts, n_qubits ):

    p = np.zeros( 2**n_qubits )

    for idx in counts :
        idx = idx.replace(' ','')
        p[ int(idx[::-1],2) ] = counts[idx]
               
    return p.reshape( n_qubits*[2] )


def resampling_counts( counts, resampling=0 ):
    
    if resampling > 0 :
        keys  = counts.keys()
        probs = np.array(list(counts.values()))
        probs = np.random.multinomial( resampling, probs/np.sum(probs) )
        counts = dict(zip(keys, probs) )
    
    return counts


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


class tomographic_gate_set_tomography:
    
    def __init__( self, n ):
        self._n = n
        
    def circuits( self ):
        n = self._n
        circ_gates = tomographic_gate_set(n)
        circ_gst = [] 
        i = 0
        for circ_j in circ_gates :
            for circ_i in circ_gates :
                for circ_k in circ_gates :
                    qc = QuantumCircuit( n, name='circuit_gate_set_{}_{}'.format(n,i) )
                    i += 1
                    qc.compose( circ_i, range(n), inplace=True )
                    qc.compose( circ_j, range(n), inplace=True )
                    qc.compose( circ_k, range(n), inplace=True )
                    qc.measure_all()
                    circ_gst.append( qc )
        
        self._circ_gst = circ_gst

        return circ_gst        
        
    def fit( self, results, circ_gst=None, resampling=0 ):
        
        if circ_gst is None :
            circ_gst = self._circ_gst
        
        self._counts = []
        for qc in circ_gst:
            counts  = resampling_counts( results.get_counts(qc), resampling=resampling )
            self._counts.append( counts )
        
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
            probs = []
            ran   = [m]
            for counts in self._counts:
                probs_temp = dict2array( marginal_counts_dictionary( counts, ran ), 1 ) 
                probs.append( probs_temp/np.sum(probs_temp)  )
            del probs_temp
            probs = np.array( probs ).reshape(4,4,4,2)
            rho_hat, Detetor_hat, Gates_hat = qt.MaximumLikelihoodGateSetTomography( probs, rho, 
                                                                                    Detector, Gates, 'gate_set')
            rho_hat_all.append( rho_hat )
            Detetor_hat_all.append( Detetor_hat )
            Gates_hat_all.append( Gates_hat )
        
        return [rho_hat_all, Detetor_hat_all, Gates_hat_all]
    
    def gateset2spam( self, gateset ):
        states_gst= []
        measures_gst = []
        for m in range(self._n):
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

        return [ states_gst, measures_gst ]

            
class measurement_process_tomography:           
    """
    clase para realizar la tomografía de procesos de un detector.
    """
    def __init__( self, n=1, p=1, m=None ):
        """
        n : numero de qubits
        p : tomografias paralelas
        """
        self._n = n
        self._p = p
        self._postselection = False
        
    def circuits(self, circ_detector = None, name = 'circuit_mpt' ):
        
        """
        circ_detector : detector custom
        """
        
        n = self._n
        p = self._p
        
        # circuitos necesarios para una tomografía de Pauli
        circ_0, circ_x, circ_h, circ_k = tomographic_gate_set(p)
        
        # circuitos de los estados locales iniciales
        circs_state_s = []
        for circ_hk in [circ_0, circ_h, circ_k ]:
            for circ_0x in  [ circ_0, circ_x ]:
                qc = QuantumCircuit(p)
                qc.compose( circ_0x, range(p), inplace=True )
                qc.compose( circ_hk, range(p), inplace=True )
                circs_state_s.append( qc )
        
        # circuitos para medir los observables de Pauli
        circs_measure_s = [circ_0, circ_h, circ_k ]    
        
        # circuitos de los estados n qubits p veces en paralelo
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
                        
        # circuitos para las medidas de Pauli de n qubits p veces en paralelo
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
        
        # Circuitos para la process measurement tomography
        circs_mpt = []
        # para un detector estandar
        if circ_detector is None:
            for i in range(6**n):
                for j in range(3**n):        
                    qc = QuantumCircuit( n*p, 2*n*p, name=name+'_{}_{}_{}_{}'.format(n,p,i,j) )
                    qc.compose( circ_state[i], qubits=range(n*p), inplace=True )
                    qc.measure( range(n*p), range(n*p) )
                    qc.compose( circ_measure[j], qubits=range(n*p), inplace=True )
                    qc.measure( range(n*p), range(n*p,2*n*p) )
                    circs_mpt.append( qc )
        # para un detector custom
        elif isinstance( circ_detector, QuantumCircuit ):
            qc = QuantumCircuit(QuantumRegister(1,'qrd0'))
            qc.add_register(ClassicalRegister(1,'crd0'))
            for j in range(1,p*n):
                qc.add_register(QuantumRegister(1,'qrd{}'.format(j)))
                qc.add_register(ClassicalRegister(1,'crd{}'.format(j)))
            for j in range(p):
                qc.compose(circ_detector, 
                           qubits=range( j*n, (j+1)*n ),
                           clbits=range( j*n, (j+1)*n ),
                           inplace=True 
                          )
            circ_detector = qc
            del qc
            
            qc0 = QuantumCircuit(QuantumRegister(1,'qr0'))  
            qc0.add_register(ClassicalRegister(1,'cr0'))    
            qc0.add_register(ClassicalRegister(1,'cr1'))    
            for j in range(1,p*n):
                qc0.add_register(QuantumRegister(1,'qr{}'.format(j)))
                qc0.add_register(ClassicalRegister(1,'cr{}'.format(2*j)))
                qc0.add_register(ClassicalRegister(1,'cr{}'.format(2*j+1)))
            
            for i in range(6**n):
                for j in range(3**n):
                    # qc = QuantumCircuit( p*n, 2*p*n )
                    qc = qc0.copy(name+'_{}_{}_{}_{}'.format(n,p,i,j))                    
                    qc.compose( circ_state[i], qubits=range(n*p), inplace=True )
                    qc.barrier()
                    qc.compose( circ_detector, qubits=range(n*p), clbits=range(n*p), inplace=True )
                    qc.barrier()
                    qc.compose( circ_measure[j], qubits=range(n*p), inplace=True )
                    qc.barrier()
                    qc.compose( circ_detector, qubits=range(n*p), clbits=range(n*p,2*n*p), inplace=True )
                    circs_mpt.append( qc )         
                
        elif isinstance( circ_detector, list ):
            self._postselection = True
            
            circ_detector_temp = []
            for circ_detector_loop in circ_detector:
            
                qc = QuantumCircuit(QuantumRegister(1,'qrd0'))
                qc.add_register(ClassicalRegister(1,'crd0'))
                for j in range(1,p*n):
                    qc.add_register(QuantumRegister(1,'qrd{}'.format(j)))
                    qc.add_register(ClassicalRegister(1,'crd{}'.format(j)))
                for j in range(p):
                    qc.compose(circ_detector_loop, 
                               qubits=range( j*n, (j+1)*n ),
                               clbits=range( j*n, (j+1)*n ),
                               inplace=True 
                              )
                circ_detector_temp.append( qc )
            del qc, circ_detector
                
                
            for k in range(len(circ_detector_temp)):
                
                for l in range(len(circ_detector_temp)):
                
                    qc0 = QuantumCircuit(QuantumRegister(1,'qr0'))  
                    qc0.add_register(ClassicalRegister(1,'cr0'))    
                    qc0.add_register(ClassicalRegister(1,'cr1'))    
                    for j in range(1,p*n):
                        qc0.add_register(QuantumRegister(1,'qr{}'.format(j)))
                        qc0.add_register(ClassicalRegister(1,'cr{}'.format(2*j)))
                        qc0.add_register(ClassicalRegister(1,'cr{}'.format(2*j+1)))
                
                    for i in range(6**n):
                        for j in range(3**n):
                            # qc = QuantumCircuit( p*n, 2*p*n )
                            qc = qc0.copy(name+'_{}_{}_{}_{}_{}_{}'.format(n,p,k,l,i,j))                    
                            qc.compose( circ_state[i], qubits=range(n*p), inplace=True )
                            qc.barrier()
                            qc.compose( circ_detector_temp[k], qubits=range(n*p), clbits=range(n*p), inplace=True )
                            qc.barrier()
                            qc.compose( circ_measure[j], qubits=range(n*p), inplace=True )
                            qc.barrier()
                            qc.compose( circ_detector_temp[l], qubits=range(n*p), clbits=range(n*p,2*n*p), inplace=True )
                            circs_mpt.append( qc )  
                
                
        self._circuits = circs_mpt
                   
        return circs_mpt
                  
        
    def fit( self, results, circuits=None, gate_set = None, resampling = 0, out = 0 ):         
                 
        if circuits is None :
            circuits = self._circuits
            
        if self._p is None : 
            self._p  = int( circuits[0].num_qubits / self._n )
        
        if self._postselection is True:
            counts = []
            for circuit_loop in circuits:
                counts.append( results.get_counts(circuit_loop) )
            
            counts00 = counts[0:18]
            counts01 = counts[18:2*18]
            counts10 = counts[2*18:3*18]
            counts11 = counts[3*18:4*18]
        
            counts_ps = [ {} for _ in range(18) ]
            
            for j in range(18):
                if '0 0' in counts00[j]:
                    counts_ps[j]['0 0'] = counts00[j]['0 0']
                if '1 0' in counts01[j]:
                    counts_ps[j]['1 0'] = counts01[j]['1 0']
                if '0 1' in counts10[j]:
                    counts_ps[j]['0 1'] = counts10[j]['0 1']
                if '1 1' in counts11[j]:
                    counts_ps[j]['1 1'] = counts11[j]['1 1']
            
            circuits = circuits[0:18]
            results  = counts_ps 
            
        self._counts = []
        if isinstance( results, Result ) :
            for qc in circuits:
                self._counts.append( resampling_counts( results.get_counts(qc), resampling=resampling ) )
        elif isinstance( results, list ) :
            for qc in range( len(circuits) ) :
                self._counts.append( resampling_counts( results[qc], resampling=resampling ) )
        del results
        
        if self._n == 1:
            # self._counts = []
            # for qc in circuits:
            #     self._counts.append( resampling_counts( results.get_counts(qc), resampling=resampling ) )
    
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
                
            Y_hat_all = []
            for m in range(self._p) :
                ran = [ m, self._p+m ] 
                probs = []
                for counts in self._counts:
                    probs_temp = dict2array( marginal_counts_dictionary( counts, ran ), 2 ) 
                    probs.append( probs_temp/np.sum(probs_temp)  )
                del probs_temp
                probs = np.array(probs).reshape([6,3,2,2]).transpose(0,1,3,2).reshape(6,6,2).transpose(1,0,2)/3
                if self._gateset is False :
                    Y_hat = qt.MaximumLikelihoodCompleteDetectorTomography( self._states, 
                                                                           self._measurements, 
                                                                           probs , Func = 0, 
                                                                           vectorized=True, out=out )
                elif self._gateset is True : 
                    Y_hat = qt.MaximumLikelihoodCompleteDetectorTomography( self._states[m], 
                                                                           self._measurements[m], 
                                                                           probs, Func = 0, 
                                                                           vectorized=True, out=out )
                Y_hat_all.append( Y_hat)
        elif self._n == 2:  
            # self._counts = []
            # for qc in circuits:
            #     self._counts.append( resampling_counts( results.get_counts(qc), resampling=resampling ) )       
    
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
                            state_temp = qt.Outer2Kron( np.kron( states_s[m*self._n][:,s1], 
                                                                states_s[m*self._n+1][:,s2] ), [2,2] )
                            states.append( state_temp.flatten() )
                    self._states.append( np.array(states).T )
    
    
                    measures = []
                    for r1 in range(3):
                        for r2 in range(3):
                            for s1 in range(2):
                                for s2 in range(2):
                                    measures_temp = qt.Outer2Kron( np.kron( measures_s[m*self._n,r1,s1], 
                                                                           measures_s[m*self._n+1,r2,s2] ), [2,2] )
                                    measures.append( measures_temp.flatten() )   
                    self._measures.append( np.array(measures).T ) 
            
            Y_hat_all = []
            for m in range(self._p) :
                ran = [ 2*m, 2*m+1, 2*self._p+2*m, 2*self._p+2*m+1 ]    
                probs = []
                for counts in self._counts:
                    probs_temp = dict2array( marginal_counts_dictionary( counts, ran ), 4 ) 
                    probs.append( probs_temp/np.sum(probs_temp) )
                del probs_temp
                probs_loop = np.array(probs).reshape(36,9,4,4
                                                     ).transpose(0,1,3,2
                                                                 ).reshape(6**self._n,
                                                                           6**self._n,
                                                                           2**self._n
                                                                           ).transpose(1,0,2)/3**2
                self._probs_loop =  probs_loop 
                if self._gateset is False :
                    Y_hat = qt.MaximumLikelihoodCompleteDetectorTomography( self._states, 
                                                                           self._measures, 
                                                                           probs_loop, Func = 0, 
                                                                           vectorized=True, out=out )
                elif self._gateset is True :    
                    Y_hat = qt.MaximumLikelihoodCompleteDetectorTomography( self._states[m], 
                                                                           self._measures[m], 
                                                                           probs_loop, Func = 0, 
                                                                           vectorized=True, out=out )
                Y_hat_all.append( Y_hat ) 
        
        elif self._n == 3:
            probs = []
            for counts in self._counts:
                probs_temp = dict2array( counts, 2*self._n ) 
                probs.append( probs_temp/np.sum(probs_temp) )

            probs = np.array(probs).reshape(6**self._n,3**self._n,2**self._n,2**self._n
                                                ).transpose(0,1,3,2
                                                            ).reshape(6**self._n,
                                                                        6**self._n,
                                                                        2**self._n
                                                                        ).transpose(1,0,2)/3**self._n

            Y_hat = qt.MaximumLikelihoodCompleteDetectorTomography( self._states, 
                                                                    self._measures, 
                                                                    probs, Func = 0, 
                                                                    vectorized=True, out=out )

        # if len(Υ_hat_all) == 1 :
        #     return Υ_hat_all[0]
        # else:
        return Y_hat_all
               


class device_process_measurement_tomography :
    
    def __init__( self, backend, parall_qubits=None ) :
        
        self._backend    = backend
        self._num_qubits = len( backend.properties().qubits )
        
        if parall_qubits is None:
            coupling_map = get_backend_conectivity( self._backend )
        
            G = nx.Graph()
            G.add_node( range(self._num_qubits) )
            G.add_edges_from(coupling_map)
            G = nx.generators.line.line_graph(G)
            G_coloring = nx.coloring.greedy_color(G)
            degree = max( G_coloring.values() ) + 1
            parall_qubits = degree*[None]
            for x in G_coloring:
                if parall_qubits[G_coloring[x]] is None:
                    parall_qubits[G_coloring[x]] = []
                parall_qubits[G_coloring[x]].append(x)
    
            
        circs_all = [ tomographic_gate_set_tomography( self._num_qubits ).circuits(), 
                     measurement_process_tomography( 1, self._num_qubits ).circuits() ]
    
        for pairs in parall_qubits :
    
            p = len(pairs)
            qubits = pairs
            qubits = [item for t in qubits for item in t]
            circ_double = measurement_process_tomography( 2, p ).circuits()
            circs = []
            for circ_loop in circ_double:
                circ = QuantumCircuit( self._num_qubits, 4*p )
                circ.compose(circ_loop, qubits=qubits, inplace=True)
                circs.append( circ )
            circs_all.append( circs )
            
        self._circuits = circs_all
        self._parall_qubits = parall_qubits
        

    def circuits( self ):
        """
        Circuits to perform the process measurement tomography of each pair of connected qubits on a device.
        
        """
        
        circuits = []
        for circuits_idx in self._circuits:
            circuits += circuits_idx
        
        return circuits

    
    def fit( self, results, out=1, resampling=0, paralell=True, gate_set=False ):
        
        gateset = tomographic_gate_set_tomography( self._num_qubits ).fit( results, 
                                                         self._circuits[0], 
                                                         resampling = resampling )
            
        states_gst= []
        measures_gst = []
        for m in range(self._num_qubits):
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
        
        if gate_set is False:
            choi_single = measurement_process_tomography(1,self._num_qubits).fit( results, 
                                                           self._circuits[1], 
                                                           resampling=resampling,
                                                           out = out)
            if paralell is False:
                choi_double = []
                for k in range(2,len(self._circuits)):
                    qubits = np.array(self._parall_qubits[k-2]).flatten()
                    choi_double.append( measurement_process_tomography(2,len(self._parall_qubits[k-2])).fit( 
                                        results, 
                                        self._circuits[k], 
                                        resampling = resampling, 
                                        out = out  ) )
            elif paralell is True:
                fun_par = lambda k : measurement_process_tomography(2,len(self._parall_qubits[k-2])).fit( 
                                        results, 
                                        self._circuits[k], 
                                        resampling = resampling,
                                        out = out )
                choi_double = Parallel(n_jobs=-1)( delayed( fun_par )(k) 
                                                  for k in range(2,len(self._circuits)) )      
            
            
            
            
        elif gate_set is True:
            choi_single = measurement_process_tomography(1,self._num_qubits).fit( results, 
                                                               self._circuits[1], 
                                                               gate_set=[states_gst,measures_gst] ,
                                                               resampling=resampling,
                                                               out = out)

            if paralell is False:
                choi_double = []
                for k in range(2,len(self._circuits)):
                    qubits = np.array(self._parall_qubits[k-2]).flatten()
                    choi_double.append( measurement_process_tomography(2,len(self._parall_qubits[k-2])).fit( 
                                        results, 
                                        self._circuits[k], 
                                        gate_set   = [ states_gst[qubits], measures_gst[qubits] ],
                                        resampling = resampling, 
                                        out = out  ) )
            elif paralell is True:
                fun_par = lambda k : measurement_process_tomography(2,len(self._parall_qubits[k-2])).fit( 
                                        results, 
                                        self._circuits[k], 
                                        gate_set   = [ states_gst[np.array(self._parall_qubits[k-2]).flatten()], 
                                                      measures_gst[np.array(self._parall_qubits[k-2]).flatten()] ] ,
                                        resampling = resampling,
                                        out = out )
                choi_double = Parallel(n_jobs=-1)( delayed( fun_par )(k) 
                                                  for k in range(2,len(self._circuits)) ) 
        
        return choi_single, choi_double, gateset 
                 




############# Noise model ################

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
    """"
    Quantities: readout fidelity, qndness and destructiveness 
    """
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

def Cross_Fidelity_Choi( Choi_single_1, Choi_single_2, Choi_double  ):
    Y0 = [ qt.Process2Choi( A )/2 for A in Kron_Choi( Choi_single_1, Choi_single_2 )]
    Y1 = [ qt.Process2Choi( A )/2 for A in Choi_double]
    f = 0
    for i in range(4):
        f += qt.Fidelity( Y0[i], Y1[i] )/2
    return f

def Cross_Fidelity_POVM( Pi_single_1, Pi_single_2, Pi_double  ):
    Pi0 = [ np.kron(A,B)/2 for A in Pi_single_1.reshape(2,2,2).transpose(1,2,0) for B in Pi_single_2.reshape(2,2,2).transpose(1,2,0) ]
    Pi1 = Pi_double.reshape(4,4,4).transpose(1,2,0)/2
    f = 0
    for i in range(4):
        f += qt.Fidelity( Pi0[i], Pi1[i] )/2
    return f

def Cross_Error_Choi( Choi_single_1, Choi_single_2, Choi_double  ):
    Y0 = [ qt.Process2Choi( A ) for A in Kron_Choi( Choi_single_1, Choi_single_2 )]
    Y1 = [ qt.Process2Choi( A ) for A in Choi_double]
    f = np.linalg.norm( np.array(Y0) - np.array(Y1) ) / np.sqrt( np.array(Y0).size )
    return f

def Cross_Error_POVM( Pi_single_1, Pi_single_2, Pi_double  ):
    Pi0 = [ np.kron(A,B) for A in Pi_single_1.reshape(2,2,2).transpose(1,2,0) 
           for B in Pi_single_2.reshape(2,2,2).transpose(1,2,0) ]
    Pi1 = Pi_double.reshape(4,4,4).transpose(1,2,0)
    f = np.linalg.norm( np.array(Pi0) - np.array(Pi1) ) / np.sqrt( np.array(Pi0).size )
    return f

def Cross_Quantities( Pi1, Choi1, Pi2, Choi2, Pi12, Choi12 ):
    
    # f = Cross_Fidelity_POVM( Pi1, Pi2, Pi12 )
    # q = Cross_Fidelity_Choi( Choi1, Choi2, Choi12 )
    f1 = Cross_Error_POVM( Pi1, Pi2, Pi12 )
    q1 = Cross_Error_Choi( Choi1, Choi2, Choi12 )
    return f1, q1

def Cross_Probability( f1, f2, pairs ):
    
    f = []
    for j in range(len(f2)):
        p = pairs[j]
        f.append( abs( f1[p[0]] * f1[p[1]] - f2[j]  ) )
    return f
    
def cross_qndness( choi1, choi2, choi12 ):
    N = len(choi1)
    d = int(np.sqrt(choi1[0].shape[0]))
    f = 0
    for n in range(N):
        for m in range(N):
            l = n*N + m
            f += abs( choi12[l][(1+d**2)*l,(1+d**2)*l] 
                     - choi1[n][(1+d)*n,(1+d)*n]*choi2[m][(1+d)*m,(1+d)*m]  )**2      
    return np.sqrt( f )    # / np.sqrt(N**2   )

def cross_fidelity( Pi1, Pi2, Pi12 ):
    d, N = Pi1.shape
    d = int(np.sqrt(d))
    f = 0.
    for n in range(N):
        for m in range(N):
            l = n*N + m
            f += abs( Pi12[:,l].reshape(d**2,d**2)[l,l]
                - Pi1[:,n].reshape(d,d)[n,n]*Pi2[:,m].reshape(d,d)[m,m] )**2
    return np.sqrt( f ) # / np.sqrt(N**2) 



###################### Plots ############################

# def sph2cart(r, theta, phi):
#     '''spherical to Cartesian transformation.'''
#     x = r * np.sin(theta) * np.cos(phi)
#     y = r * np.sin(theta) * np.sin(phi)
#     z = r * np.cos(theta)
#     return x, y, z

# def sphview(ax):
#     '''returns the camera position for 3D axes in spherical coordinates'''
#     r = np.square(np.max([ax.get_xlim(), ax.get_ylim()], 1)).sum()
#     theta, phi = np.radians((90-ax.elev, ax.azim))
#     return r, theta, phi

# def getDistances(view, xpos, ypos, dz):
#     distances  = []
#     for i in range(len(xpos)):
#         distance = (xpos[i] - view[0])**2 + (ypos[i] - view[1])**2 + (dz[i] - view[2])**2
#         distances.append(np.sqrt(distance))
#     return distances

# def Bar3D( A , ax = None, xpos=None, ypos=None, zpos=None, dx=None, dy=None, M = 0, **args ):
    
#     d = A.shape[0]
#     camera = np.array([13.856, -24. ,0])
    
#     if xpos is None :
#         xpos = np.arange(d) 
#     if ypos is None :
#         ypos = np.arange(d)
#     xpos, ypos = np.meshgrid( xpos, ypos )
#     xpos = xpos.flatten()
#     ypos = ypos.flatten()
    
#     if zpos is None :
#         zpos = np.zeros_like(xpos)
#     else :
#         zpos = zpos.flatten()
    
#     if dx is None :
#         dx = 0.5 * np.ones_like(xpos)
#     else :
#         dx = dx * np.ones_like(ypos)
        
#     if dy is None :
#         dy = 0.5 * np.ones_like(ypos)
#     else :
#         dy = dy * np.ones_like(ypos)
    
#     dz = A.flatten()
#     z_order = getDistances(camera, xpos, ypos, zpos)
    
#     if ax == None :
#         fig = plt.figure()   
#         ax  = fig.add_subplot( 1,1,1, projection='3d')  
#     maxx    = np.max(z_order) + M
    
#     plt.rc('font', size=15) 
#     for i in range(xpos.shape[0]):
#         pl = ax.bar3d(xpos[i], ypos[i], zpos[i], 
#                       dx[i], dy[i], dz[i], 
#                       zsort='max', **args )
#         pl._sort_zpos = maxx - z_order[i]
#         ax.set_xticks( [0.25,1.25,2.25,3.25] )
#         ax.set_xticklabels((r'$|gg\rangle$',r'$|ge\rangle$',
#                                 r'$|eg\rangle$',r'$|ee\rangle$'))
#         ax.set_yticks( [0.25,1.25,2.25,3.25] )
#         ax.set_yticklabels((r'$\langle gg|$',r'$\langle ge|$',
#                                 r'$\langle eg|$',r'$\langle ee|$'))
#         ax.set_title( label, loc='left', fontsize=20, x = 0.1, y=.85)
#         ax.set_zlim([0,1])
#     return ax            


# def Abs_Bars3D(Y):
#     fig = plt.figure(figsize=(len(Y)*4,5)) 
#     for y in range(len(Y)):
#         ax  = fig.add_subplot( 1, len(Y), y+1,  projection='3d')
#         Bar3D( np.abs( Y[y] ).T, ax=ax )   
#     return fig


            
            
            
            
            
            
            
            
            
            
