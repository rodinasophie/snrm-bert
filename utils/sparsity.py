from utils.helpers import path_exists, load_file, list_files, join, estimate_sparsity
from datetime import datetime

VERY_LARGE_EPOCH = 1000
MAX_ITER = 60

def count_zeros(model, loader, batch_size, element_type):
    is_end = False
    print("Counting zeros for ", element_type)
    fn = loader.generate_queries if element_type == 'queries' else loader.generate_docs
    zeros = 0
    counter = 0
    all_zeros=0
    #k = 0
    while not is_end:
        #if k == MAX_ITER:
        #    break
        #k += 1
        ids, docs, is_end = fn(batch_size)
        repr = model.evaluate_repr(docs, input_type=element_type).cpu().numpy()
        counter += repr.shape[0]
        for i in range(repr.shape[0]):
            z, nans = estimate_sparsity(repr[i])
            if nans != 0:
                print("Nans for id {} = {}, zeros = {}".format(ids[i], nans, z), flush=True)
            zeros += z
            if z == repr.shape[1]:
                all_zeros += 1
                #print("All zeros({}, {}) for q: id-{}, text-{}".format(z, repr.shape[1], ids[i], docs[i]), flush=True) 
        if is_end:
            break
    print("All zeros: ", all_zeros)
    loader.reset()
    return float(zeros)/float(counter) # return mean zeros
        

def check_sparsity(model, loader, batch_size):
    query_mean_zeros = count_zeros(model, loader, batch_size, 'queries')
    docs_mean_zeros = count_zeros(model, loader, batch_size, 'docs')

    return query_mean_zeros, docs_mean_zeros
