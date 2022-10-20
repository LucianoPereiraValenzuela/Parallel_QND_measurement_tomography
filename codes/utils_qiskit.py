
import json
import copy 
from qiskit.result import Result

def dict2results( dictionary ):

    results = Result( backend_name = dictionary['backend_name'],
                        backend_version = dictionary['backend_version'],
                        qobj_id = dictionary['qobj_id'],
                        job_id = dictionary['job_id'],
                        success = dictionary['success'],
                        results = dictionary['results'] )

    results = results.from_dict( dictionary )

    return results

def CombineResults( results ):
    
    combined_result = copy.deepcopy(results[0])
    for idx in range(1, len(results)):
            combined_result.results.extend(results[idx].results)
    return combined_result


def results2dict( results ):

    dic  = results.to_dict()
    date = dic['date']
    dic['date'] = date.__str__()

    return dic

def save_results( results, name=None , folder=None):

    dic = results2dict(results)

    if name is None:
        name = dic['job_id']

    if folder is None:
        folder = ''
    else:
        folder = folder+'/'

    with open(folder+name+'.json', 'w') as f:
        json.dump(dic, f)


def load_results( name, folder=None ):

    with open('{}/{}.json'.format(folder,name), 'r') as f:
        results_job = dict2results( json.load(f) )

    return results_job