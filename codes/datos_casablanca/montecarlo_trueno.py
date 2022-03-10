from qiskit import IBMQ
from main import *
from joblib import Parallel, delayed
from time import time

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-csic', group='internal', project='iff-csic')
backend  = provider.get_backend('ibm_lagos')

circs_all, circs_pkg, pkg_idx, parall_qubits = device_process_measurement_tomography_circuits( backend )

results   = [backend.retrieve_job('616f5990cea63358bd144f6c').result(),
             backend.retrieve_job('616f5997dd30e90910e61666').result()]

mc = 100

def fun():
    choi_single, choi_double, gateset  = device_process_measurement_tomography_fitter( results, circs_all, circs_pkg, pkg_idx, out=1, resampling=2**10, paralell=False )
    num_qubits = len(choi_single)
    
    quantities = []
    for k in range(num_qubits):
        quantities.append( Quantities( choi_single[k][0], choi_single[k][1] ) )
    
    quantities_2 = []
    cros_quantities = []
    for i in range(len(parall_qubits)):
        for j in range(len(parall_qubits[i])):
            k = parall_qubits[i][j][0]
            l = parall_qubits[i][j][1]
            cros_quantities.append( Cross_Quantities( choi_single[k][0], choi_single[k][1],
                                                      choi_single[l][0], choi_single[l][1],
                                                      choi_double[i][j][0], choi_double[i][j][1]
                                                    )  )
            quantities_2.append( Quantities( choi_double[i][j][0], choi_double[i][j][1] ) )
    
    return quantities, quantities_2, cros_quantities


t1 = time()
datos = Parallel(n_jobs=-1)( delayed( fun )() for _ in range(mc) ) 
t2 = time()
print(t2-t1)

np.save( 'datos_montecarlo', np.array( datos, dtype=object ) )





