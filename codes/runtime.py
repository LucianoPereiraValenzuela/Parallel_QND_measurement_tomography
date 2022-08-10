import numpy as np
from qiskit import transpile
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.providers.aer.noise import NoiseModel

#####main runtime function
def main( backend, user_messenger ):

    parall_qubits = [  [(0,1)]
                    ]

    qndmt = device_process_measurement_tomography( backend, parall_qubits=parall_qubits )
    circuits_qndmt = qndmt.circuits()

    # job_manager = IBMQJobManager()
    # job = job_manager.run( transpile( circuits_qndmt, backend ) , backend=backend, shots=2**13 )

    # job_id = job.job_set_id()
    
    # results = job.results().combine_results()
    
    n_circuits = len( circuits_qndmt )
    n_steps = 1 + n_circuits // 100
    circuit_split = n_circuits // n_steps
    jobs_id = []
    results = []
    
    for j in range(n_steps):
        job = backend.run( transpile( circuits_qndmt, backend ), shots=2**13  )
        jobs_id.append( job.job_id )
        
    results = job.result()

    choi_single, choi_double, gateset  = qndmt.fit( results, paralell=True, gate_set=True ) 

    dict_results = {  
                    'choi_single' : choi_single,
                    'choi_double' : choi_double,
                    'gateset' : gateset
                    }

    return dict_results

#####main.py
import numpy as np
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
        X = np.kron( PauliMatrices(1), PauliMatrices(1) )
        H = np.kron( PauliMatrices(1) + PauliMatrices(3), PauliMatrices(1) + PauliMatrices(3) )/2
        K = np.kron( PauliMatrices(0) + 1j*PauliMatrices(1), PauliMatrices(0) - 1j*PauliMatrices(1) )/2
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
            rho_hat, Detetor_hat, Gates_hat = MaximumLikelihoodGateSetTomography( probs, rho, 
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
                    Y_hat = MaximumLikelihoodCompleteDetectorTomography( self._states, 
                                                                           self._measurements, 
                                                                           probs , Func = 0, 
                                                                           vectorized=True, out=out )
                elif self._gateset is True : 
                    Y_hat = MaximumLikelihoodCompleteDetectorTomography( self._states[m], 
                                                                           self._measurements[m], 
                                                                           probs, Func = 0, 
                                                                           vectorized=True, out=out )
                Y_hat_all.append( Y_hat)
        else:  
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
                            state_temp = Outer2Kron( np.kron( states_s[m*self._n][:,s1], 
                                                                states_s[m*self._n+1][:,s2] ), [2,2] )
                            states.append( state_temp.flatten() )
                    self._states.append( np.array(states).T )
    
    
                    measures = []
                    for r1 in range(3):
                        for r2 in range(3):
                            for s1 in range(2):
                                for s2 in range(2):
                                    measures_temp = Outer2Kron( np.kron( measures_s[m*self._n,r1,s1], 
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
                    Y_hat = MaximumLikelihoodCompleteDetectorTomography( self._states, 
                                                                           self._measures, 
                                                                           probs_loop, Func = 0, 
                                                                           vectorized=True, out=out )
                elif self._gateset is True :    
                    Y_hat = MaximumLikelihoodCompleteDetectorTomography( self._states[m], 
                                                                           self._measures[m], 
                                                                           probs_loop, Func = 0, 
                                                                           vectorized=True, out=out )
                Y_hat_all.append( Y_hat ) 
        
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
    Y0 = [ Process2Choi( A )/2 for A in Kron_Choi( Choi_single_1, Choi_single_2 )]
    Y1 = [ Process2Choi( A )/2 for A in Choi_double]
    f = 0
    for i in range(4):
        f += Fidelity( Y0[i], Y1[i] )/2
    return f

def Cross_Fidelity_POVM( Pi_single_1, Pi_single_2, Pi_double  ):
    Pi0 = [ np.kron(A,B)/2 for A in Pi_single_1.reshape(2,2,2).transpose(1,2,0) for B in Pi_single_2.reshape(2,2,2).transpose(1,2,0) ]
    Pi1 = Pi_double.reshape(4,4,4).transpose(1,2,0)/2
    f = 0
    for i in range(4):
        f += Fidelity( Pi0[i], Pi1[i] )/2
    return f

def Cross_Error_Choi( Choi_single_1, Choi_single_2, Choi_double  ):
    Y0 = [ Process2Choi( A ) for A in Kron_Choi( Choi_single_1, Choi_single_2 )]
    Y1 = [ Process2Choi( A ) for A in Choi_double]
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


            
            
            
            
            
            
            
            
            
            

#####QuantumTomography.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize, least_squares

# In[2]:

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
        LogLikelihood = -np.sum( counts_ex*np.log10(counts_th + 1e-10) )
    elif Func == 1: #Chi-Square
        LogLikelihood = np.sum( (counts_th - counts_ex)**2/(counts_ex + 1e-10 ) )
    elif Func == 2: #Normal
        LogLikelihood = np.sum( (counts_th - counts_ex)**2/(counts_th + 1e-10 ) )
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
#     dim = int(np.sqrt(Pi.shape[0]))
#     t0  = np.array( [ PositiveMatrix2CholeskyVector( PositiveOperatorProjection( Pi[:,k].reshape(dim,dim), 2) )  for k in range(Pi.shape[1])] ).flatten()
#     fun = lambda t : la.norm( np.array([ CholeskyVector2PositiveMatrix(t.reshape(-1,dim**2)[k,:]).flatten() - Pi[:,k] for k in range(Pi.shape[1]) ])  )
#     con = lambda t : la.norm( np.sum(np.array([ CholeskyVector2PositiveMatrix(t.reshape(-1,dim**2)[k,:]).flatten() for k in range(Pi.shape[1]) ]),0) - np.eye(dim).flatten() )
#     t   = minimize( fun, t0 , constraints = ({'type': 'eq', 'fun': con }) )
#     Pi  = np.array([ CholeskyVector2PositiveMatrix(t.x.reshape(-1,dim**2)[k,:]).flatten() for k in range(Pi.shape[1]) ]).T
#     return Pi
    dim = int(np.sqrt(Pi.shape[0]))
    Num = Pi.shape[1]
    Pi  = np.array( [ PositiveOperatorProjection( Pi[:,k].reshape(dim,dim), 2).flatten() for k in range(Num) ] ).T
    return Pi
    

def MaximumLikelihoodDetectorTomography( ProveStates, counts, Guess = [] , Func = 1, vectorized=False ):
    if vectorized == False:
        ProveStates = VectorizeVectors(ProveStates)
    Dim = int(np.sqrt(ProveStates[:,0].size))
    Num = counts[:,0].size
    if not Guess:
        Guess = ProbabilityOperatorsProjection( LinearDetectorTomography( ProveStates, counts, True ) )
    t_guess   = np.array([ PositiveMatrix2CholeskyVector(Guess[:,k].reshape([Dim,Dim])) for k in range(Num) ]).flatten() 
    counts_th = lambda t : np.real( 
                    np.array([ CholeskyVector2PositiveMatrix( t.reshape([-1,Dim**2] )[k,:] ).flatten() for k in range(Num) ]).conj()@ProveStates )    
    fun         = lambda t : LikelihoodFunction( counts_th(t), counts, Func )
    constraints = ({'type': 'eq', 'fun': lambda t: la.norm(
                    np.sum(np.array([ CholeskyVector2PositiveMatrix( t.reshape([-1,Dim**2] )[k,:] ).flatten() for k in range(Num) ]),0)-np.eye(Dim).flatten() ) })
    results  = minimize( fun, t_guess, constraints = constraints, method = 'SLSQP' )  
    t        = results.x
    Estimate = np.array([ CholeskyVector2PositiveMatrix( t.reshape([-1,Dim**2] )[k,:] ).flatten() for k in range(Num) ]).T
    norm     = np.trace(np.sum(Estimate,1).reshape(Dim,Dim))
    Estimate = Dim * Estimate / norm
    return Estimate


# In[5]:


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
    Estimate = Process2Choi( CholeskyVector2PositiveMatrix ( t, Dim) )
    return Estimate


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

# def MaximumLikelihoodCompleteDetectorTomography( States, Measurements, counts, Guess = [] , Func = 0, vectorized=False ):
#     if vectorized == False:
#         States = VectorizeVectors(States)
#         Measurements = VectorizeVectors(Measurements)  
#     Dim = int(np.sqrt(States.shape[0]))
#     Num = counts.shape[2]
#     if isinstance(Guess, list) and not Guess:
#         Guess = CompleteDetectorOperatorProjection( LinearCompleteDetectorTomography( States, Measurements, counts, True) )
#     t_guess =np.array([ PositiveMatrix2CholeskyVector( Process2Choi(Guess[:,:,k]) ) for k in range(Num) ]).flatten()
#     counts_th = lambda t : np.real( np.array([ 
#                                      Measurements.conj().T@Kron2Outer( CholeskyVector2PositiveMatrix( t.reshape(-1,Dim**4)[k,:] ),[Dim,Dim] )@States 
#                                     for k in range(Num) ]).transpose( [1,2,0] ) )
#     fun = lambda t : LikelihoodFunction(counts_th(t),counts,Func)
#     Cons = lambda t: la.norm(PartialTrace( np.sum(np.array([ CholeskyVector2PositiveMatrix( t.reshape(-1,Dim**4)[k,:] ) for k in range(Num)] ),0), 
#                                                                 [Dim,Dim], 0) - np.eye(Dim, dtype=complex) )
#     results = minimize( fun, t_guess, constraints = ({'type': 'eq', 'fun': Cons }) )  
#     t = results.x
#     Estimate = np.array([Kron2Outer( CholeskyVector2PositiveMatrix( t.reshape(-1,Dim**4)[k,:] ),[Dim,Dim] ) for k in range(Num)  ]).transpose([1,2,0])
#     return  Estimate


def MaximumLikelihoodCompleteDetectorTomography( States, Measurements, Probs_ex , Func = 0, vectorized=False , Pi = None, out = 0 ):
    
    if vectorized == False:
        States = VectorizeVectors(States)
        Measurements = VectorizeVectors(Measurements)
        
    Dim = int(np.sqrt(States.shape[0]))
    Num = Probs_ex.shape[2]
    
    if Pi is None:
        Probs_0 = np.sum( Probs_ex , 0).T
        Pi = MaximumLikelihoodDetectorTomography( States, Probs_0, vectorized = True , Func = Func )
        
    Choiv = []
    
    for k in range(Num):
        Choiv_Lin  = LinearProcessTomography( States, Measurements, Probs_ex[:,:,k], vectorized = True )
#         print(Choiv_Lin)
#         Choiv.append( Choiv_Lin )
        Y_guess    = PositiveOperatorProjection( Process2Choi( Choiv_Lin ), 2 )
        
#         print(k, Y_guess)
        
        t_guess   = PositiveMatrix2CholeskyVector( Y_guess )
        Probs_th  = lambda t : np.real( Measurements.conj().T @ Process2Choi( CholeskyVector2PositiveMatrix( t ) )@States )  
        fun       = lambda t : LikelihoodFunction( Probs_th(t), Probs_ex[:,:,k], Func )
        con       = lambda t:  la.norm( PartialTrace( CholeskyVector2PositiveMatrix( t ) ,[Dim,Dim], 0).T.flatten() - Pi[:,k].flatten() )**2 
        
#         print( 'prob',Probs_th(t_guess), Probs_ex[:,:,k])
#         print( 'fun_in', fun(t_guess), con(t_guess) )
        
        constraints = ({'type': 'eq', 'fun': con })
        results   = minimize( fun, t_guess, constraints = constraints, method = 'SLSQP' )  
        t         = results.x
        
#         print( 'fun_out', fun(t), con(t) )
        
        Ye = Process2Choi( CholeskyVector2PositiveMatrix( t ) )
        Choiv.append( Ye ) 
        
    norm = np.trace( Process2Choi( np.sum( Choiv, 0 ) ) )
    
    for k in range(Num):
        Choiv[k] = Dim * Choiv[k] / norm
      
    if out == 0 :
        return Choiv
    elif out == 1:
        return Pi, Choiv


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
        
    results = minimize( fun, t0, constraints = ({'type': 'eq', 'fun': con }), method = 'SLSQP'  ) # , bounds = tuple(len(t0)*[(-1,1)])
    t = results.x
        
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
    results = least_squares( fun, t0, bounds= (-1,1) )  
    B = B_t(results.x)
  
    State    = B@State
    Detector = la.inv(B).conj().T@Detector
    Gates    = B@Gates@la.inv(B)
    
    for j in range(N_Gates):
        Gates[j] = Dim * Gates[j] / np.trace( Process2Choi( Gates[j] ) )
    
    return State, Detector, Gates
    
def UnitaryProjection(Z):
    U = Z@la.fractional_matrix_power( Z.conj().T@Z , -0.5 )
    return U
    
def Counts_GQT(t,Dim,N_outcomes,N_Gates):
    t = t.reshape(-1,Dim**2)
    t_state    = t[0,:]
    t_Detector = t[1:1+N_outcomes,:]
    t_Gates    = t[1+N_outcomes:,:].reshape(-1,Dim**4)
    
    rho      = CholeskyVector2PositiveMatrix(t_state,1).flatten()
    Detector =  np.array([ CholeskyVector2PositiveMatrix(t_Detector[k,:]).flatten() for k in range(N_outcomes)]).T
    Gates    = np.array( [ Process2Choi(CholeskyVector2PositiveMatrix(t_Gates[k,:],Dim)) for k in range(N_Gates) ] )
    
    Counts   = np.array( [[[ Detector.conj().T@Gates[j,:,:]@Gates[k,:,:]@Gates[i,:,:]@rho for j in range(N_Gates)]for i in range(N_Gates)]for k in range(N_Gates)] )
    
    return np.real( Counts )

def Constraints_GSQT(t,Dim,N_outcomes,N_Gates):
    t = t.reshape(-1,Dim**2)
    t_state = t[0,:]
    t_Detector = t[1:1+N_outcomes,:]
    t_Gates = t[1+N_outcomes:,:].reshape(-1,Dim**4)
    f1 = la.norm( np.sum(np.array([ CholeskyVector2PositiveMatrix(t_Detector[k,:]).flatten() for k in range(N_outcomes) ]),0) - np.eye(Dim).flatten() )**2
    f2 = la.norm( np.array([ PartialTrace( CholeskyVector2PositiveMatrix(t_Gates[k,:]) ,[Dim,Dim], 0).flatten() - np.eye(Dim).flatten() for k in range(N_Gates) ]) )**2
    return np.append(f2,f1)

def GaugeFix_Fun(t,rho,Pi,Gamma,rho_tarjet,Pi_tarjet,Gamma_tarjet,B_t,gauge='gate_set'):
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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




