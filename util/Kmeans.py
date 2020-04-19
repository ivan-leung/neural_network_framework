import numpy as np
import types
from Statistics import col_mean
from ErrorHandle import validate_input
from DataFrame import DataFrame
from Model import Model

def el2(e1,e2):
    if e1 is None or e2 is None:
        return 0

    else:
        return pow(e1-e2,2) 

class kmeans(Model):
    def __init__(self, data, dist_fn=el2):
        Model.__init__(self)
        self.data = data
        self.prediction_fn = lambda x: 1
        self.dist_fn = dist_fn

    def train(self, num_clusters, max_iters):
        """
        Performs k-means clustering.
        
        Key Arguments:
            entries - numpy matrix that contains the data entries to be clustered.
            num_clusters - integer of number of clusters sought.
            max_iters - maximum number of iterations before algorithm terminates.
            dist_fn - distance function/ functions.
                  Defaults to applying L2 distance to every attribute in df.
                  if provided a different function, this function is the 
                  distance measurement applied across all attributes.
        """
        entries = self.data.get_entries(mat=True)
        validate_input(isinstance(num_clusters, int), "num_clusters must be an integer")
        validate_input(isinstance(max_iters, int), "max_iters must be an integer")
        validate_input(isinstance(self.dist_fn, types.FunctionType), "dist_fns must be a function")

        centroids = self._init_centroids(entries, num_clusters)
        it = 0
        prev_assignments = [None for _ in range(entries.shape[0])]
        while it < max_iters:
            assignments, cost = self._cluster_points(entries, centroids)
            print "iter", it, 'Cost=', cost
            if prev_assignments == assignments:
                print "="*8 + "converged" + "="*8
                break
            prev_assignments = assignments
            centroids = self._calculate_centroids(entries, assignments, num_clusters)
            it += 1
        self.centroids = centroids

    def summary(self):
        summary = {}
        cdata = np.matrix(np.vstack((self.data.get_col_names(), self.centroids)))
        df = DataFrame(cdata, row_names=False)        
        summary["centroids"] = df 
        entries = self.data.get_entries(mat=True)
        assignments, cost = self._cluster_points(entries, self.centroids)
        a_list = sorted(zip(assignments, \
                [i for i in range(len(self.data.get_row_names()))]))

        summary["assignments"] = dict()
        for a in a_list:
            cluster_num, row_name = a
            if cluster_num not in summary["assignments"]:
                summary["assignments"][cluster_num] = []
            summary["assignments"][cluster_num].append(row_name) 
        summary["cost"] = cost       
        return summary
        

    def _cluster_points(self, entries, centroids):
        n_row = entries.shape[0]
        n_col = entries.shape[1]
        dist_vecfn = np.vectorize(self.dist_fn)
        assignment_cost =  [self._assign_entry(entry, centroids, dist_vecfn) for entry in entries]
        assignments = [elem[1] for elem in assignment_cost]
        cost = sum([elem[0] for elem in assignment_cost])
        return (assignments, cost)

    def _assign_entry(self, entry, centroids, dist_vecfn):
        all_dists =  np.sum(dist_vecfn(entry, centroids), 1)
        return (min(all_dists), np.argmin(all_dists))
        
    def _calculate_centroids(self, entries, assignments, k):
        new_centroids = [np.zeros((0,entries.shape[1])) for _ in range(k)]
        counts = np.array([0. for _ in range(k)]).reshape((k,1))
        for i, entry in enumerate(entries):
            cluster = assignments[i]
            #print 'cluster', cluster
            new_centroids[cluster] = np.vstack((new_centroids[cluster],entry))
            counts[cluster] += 1
        for c, nc in enumerate(new_centroids):
            new_centroids[c] = col_mean(nc)
        for i in range(k):
            if counts[i] == 0:
                new_centroids[i] = [0 for _ in range(entries.shape[1])] 
        return new_centroids
        
    def _init_centroids(self, entries, k):
        validate_input(k <= entries.shape[0], "cluster number exceed data pts")
        chosen_points = np.random.choice(entries.shape[0], k).tolist()
        return np.array(entries[chosen_points,:], dtype=object)        



