from concurrent import futures

import numpy as np
import nifty
import nifty.graph.opt.multicut as nmc
import nifty.graph.opt.lifted_multicut as nlmc


# this returns a 2d array with the all the indices of matching rows for a and b
# cf. http://stackoverflow.com/questions/20230384/find-indexes-of-matching-rows-in-two-2-d-arrays
def find_matching_row_indices(x, y):
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    # using a dictionary, this is faster than the pure np variant
    indices = []
    rows_x = {tuple(row): i for i, row in enumerate(x)}
    for i, row in enumerate(y):
        if tuple(row) in rows_x:
            indices.append([rows_x[tuple(row)], i])
    return np.array(indices)


def compute_mc_superpixels(affinities, n_threads):
    segmenter = McSuperpixel(stacked_2d=True, n_threads=n_threads)
    return segmenter(affinities)


def compute_long_range_mc_superpixels(affinities, offsets,
                                      only_repulsive_lr, n_threads,
                                      stacked_2d=True):
    segmenter = LongRangeMulticutSuperpixel(offsets=offsets, only_repulsive_lr=only_repulsive_lr,
                                            stacked_2d=stacked_2d, n_threads=n_threads)
    return segmenter(affinities)


def compute_lmc_superpixels(affinities, offsets, n_threads, stacked_2d=True):
    segmenter = LmcSuperpixel(offsets=offsets, n_threads=n_threads, stacked_2d=stacked_2d)
    return segmenter(affinities)


def size_filter(hmap, seg, threshold):
    import vigra
    segments, counts = np.unique(seg, return_counts=True)
    mask = np.ma.masked_array(seg, np.in1d(seg, segments[counts < threshold])).mask
    filtered = seg.copy()
    filtered[mask] = 0
    filtered, _ = vigra.analysis.watershedsNew(hmap, seeds=filtered.astype("uint32"))
    filtered, max_label, _ = vigra.analysis.relabelConsecutive(filtered, start_label=1)
    return filtered, max_label


def superpixel_stacked_from_affinities(affinities, sp2d_fu, n_threads):
    segmentation = np.zeros(affinities.shape[1:], dtype='uint32')

    def run_sp_2d(z):
        seg, off = sp2d_fu(affinities[:, z])
        segmentation[z] = seg
        return off + 1

    with futures.ThreadPoolExecutor(max_workers=n_threads) as tp:
        tasks = [tp.submit(run_sp_2d, z) for z in range(len(segmentation))]
        offsets = [t.result() for t in tasks]

    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    offsets = np.cumsum(offsets).astype('uint32')
    segmentation += offsets[:, None, None]
    return segmentation, segmentation.max()


def multicut(n_nodes, uvs, costs, time_limit=None):
    graph = nifty.graph.UndirectedGraph(n_nodes)
    graph.insertEdges(uvs)
    obj = nmc.multicutObjective(graph, costs)
    solver = obj.kernighanLinFactory(warmStartGreedy=True).create(obj)
    if time_limit is not None:
        visitor = obj.verboseVisitor(visitNth=100000000, timeLimitSolver=time_limit)
        node_labels = solver.optimize(visitor=visitor)
    else:
        node_labels = solver.optimize()
    return node_labels


def lifted_multicut(n_nodes, local_uvs, local_costs, lifted_uvs, lifted_costs, time_limit=None):
    graph = nifty.graph.UndirectedGraph(n_nodes)
    graph.insertEdges(local_uvs)

    lifted_obj = nlmc.liftedMulticutObjective(graph)
    lifted_obj.setCosts(local_uvs, local_costs)
    lifted_obj.setCosts(lifted_uvs, lifted_costs)
    # visitor = lifted_obj.verboseVisitor(100)
    solver_ehc = lifted_obj.liftedMulticutGreedyAdditiveFactory().create(lifted_obj)
    node_labels = solver_ehc.optimize()
    solver_kl = lifted_obj.liftedMulticutKernighanLinFactory().create(lifted_obj)
    node_labels = solver_kl.optimize(node_labels)
    return node_labels


def probs_to_costs(probs, beta=.5):
    p_min = 0.001
    p_max = 1. - p_min
    costs = (p_max - p_min) * probs + p_min
    # probabilities to energies, second term is boundary bias
    costs = np.log((1. - costs) / costs) + np.log((1. - beta) / beta)
    return costs


class WatershedBase(object):
    def __init__(self, grower):
        # check that this is callable
        self.grower = grower

    def __call__(self, affinities):
        return self.grower(affinities)

    @staticmethod
    def get_2d_from_3d_offsets(offsets):
        # only keep in-plane channels
        keep_channels = [ii for ii, off in enumerate(offsets) if off[0] == 0]
        offsets = [off[1:] for ii, off in enumerate(offsets) if ii in keep_channels]
        return keep_channels, offsets


class McSuperpixel(WatershedBase):
    def __init__(self, beta=.5, min_segment_size=0, stacked_2d=False, n_threads=1):
        self.beta = beta
        self.min_segment_size = min_segment_size
        self.stacked_2d = stacked_2d
        self.n_threads = n_threads

    def mc_superpixel(self, affinities):
        shape = affinities.shape[1:]
        grid_graph = nifty.graph.undirectedGridGraph(shape)
        costs = grid_graph.affinitiesToEdgeMap(affinities)
        assert len(costs) == grid_graph.numberOfEdges
        costs = probs_to_costs(costs, beta=self.beta)
        segmentation = multicut(
            grid_graph.numberOfNodes,
            grid_graph.uvIds(),
            costs
        ).reshape(shape)
        if self.min_segment_size > 0:
            affinities = np.sum(affinities, axis=0)
            segmentation, max_label = size_filter(affinities, segmentation, self.min_segment_size)
        else:
            max_label = segmentation.max()
        return segmentation, max_label

    def __call__(self, affinities):
        if self.stacked_2d:
            assert affinities.shape[0] >= 3
            affinities_ = np.require(affinities[1:3], requirements='C')
            segmentation, _ = superpixel_stacked_from_affinities(affinities_, self.mc_superpixel, self.n_threads)

        else:
            if affinities.shape[0] > 3:
                affinities_ = np.require(affinities[:3], requirements='C')
            else:
                affinities_ = affinities
            segmentation, _ = self.mc_superpixel(affinities_)
        return segmentation


class LongRangeMulticutSuperpixel(WatershedBase):
    def __init__(self,
                 offsets,
                 beta=.5,
                 only_repulsive_lr=False,
                 min_segment_size=0,
                 stacked_2d=False,
                 n_threads=1):
        self.stacked_2d = stacked_2d
        assert isinstance(offsets, list)
        if self.stacked_2d:
            self.keep_channels, self.offsets = self.get_2d_from_3d_offsets(offsets)
        else:
            self.offsets = offsets
        self.beta = beta
        self.min_segment_size = min_segment_size
        self.only_repulsive_lr = only_repulsive_lr
        self.n_threads = n_threads

    def lr_mc_superpixel(self, affinities):
        shape = affinities.shape[1:]
        grid_graph = nifty.graph.undirectedGridGraph(shape)
        edge_map = grid_graph.liftedProblemFromLongRangeAffinities(affinities,
                                                                   self.offsets)
        uvs = np.array([key for key in edge_map.keys()], dtype='uint32')
        costs = np.array([val for val in edge_map.values()], dtype='float64')

        # filter out the long range non-repulsive edges
        if self.only_repulsive_lr:
            print("Only repulsive")
            assert self.beta == .5
            local_uvs = grid_graph.uvIds()
            # compare to local connectivity
            local_lr_edges = find_matching_row_indices(uvs, local_uvs)[:, 0]
            assert len(local_lr_edges) == len(local_uvs)

            # invert indices to get the mask only containing long range edges
            lr_edge_mask = np.ones(len(uvs), dtype='bool')
            lr_edge_mask[local_lr_edges] = False

            # mask for repulsive edges
            repulsive_mask = costs > .5
            assert len(repulsive_mask) == len(lr_edge_mask)

            # FIXME thsi shouldn't be so complicated...
            x = repulsive_mask
            y = lr_edge_mask
            final_mask = np.logical_not(np.logical_xor(y, np.logical_and(x, y)))
            uvs = uvs[final_mask]
            costs = costs[final_mask]

        # FIXME this should be the other way round, but I am suffering from the usual confusion with the sign
        costs = probs_to_costs(costs, beta=self.beta)
        assert len(costs) == len(uvs)
        assert uvs.shape[1] == 2
        assert uvs.max() + 1 == grid_graph.numberOfNodes, "%i, %i" % (uvs.max(), grid_graph.numberOfNodes)

        segmentation = multicut(grid_graph.numberOfNodes, uvs, costs).reshape(shape)
        if self.min_segment_size > 0:
            affinities_ = np.sum(affinities, axis=0)
            segmentation, max_label = size_filter(affinities_, segmentation, self.min_segment_size)
        else:
            max_label = segmentation.max()
        return segmentation, max_label

    def __call__(self, affinities):
        assert affinities.shape[0] == len(self.offsets), "%i, %i" % (affinities.shape[0], len(self.offsets))
        if self.stacked_2d:
            affinities_ = np.require(affinities[self.keep_channels], requirements='C')
            segmentation, _ = superpixel_stacked_from_affinities(affinities_, self.lr_mc_superpixel, self.n_threads)

        else:
            segmentation, _ = self.lr_mc_superpixel(affinities)
        return segmentation


class LmcSuperpixel(WatershedBase):
    def __init__(self, offsets,
                 beta=.5, beta_lifted=.5,
                 cost_weight=1., min_segment_size=0,
                 stacked_2d=False, n_threads=1):
        self.stacked_2d = stacked_2d
        # if we calculate stacked 2d superpixels from 3d affinity
        # maps, we must adjust the offsets by excludig all offsets
        # with z coordinates and make the rest 2d
        if self.stacked_2d:
            self.keep_channels, self.offsets = self.get_2d_from_3d_offsets(offsets)
        else:
            self.offsets = offsets
        self.beta = beta
        self.beta_lifted = beta_lifted
        self.cost_weight = cost_weight
        self.min_segment_size = min_segment_size
        self.n_threads = n_threads

    def lmc_superpixel(self, affinities, dim):
        shape = affinities.shape[1:]
        grid_graph = nifty.graph.undirectedGridGraph(shape)
        edge_map = grid_graph.liftedProblemFromLongRangeAffinities(affinities,
                                                                   self.offsets)
        uvs = np.array([key for key in edge_map.keys()] , dtype='uint32')
        costs = np.array([val for val in edge_map.values()], dtype='float64')

        # split uv-ids and costs into local and lifted uv-ids
        local_uvs = grid_graph.uvIds()
        # find edges of the local connectivity in our uvs
        local_edges = find_matching_row_indices(uvs, local_uvs)[:, 0]
        assert len(local_edges) == len(local_uvs)
        # invert indices to get the mask only containing long range edges
        lr_edge_mask = np.ones(len(uvs), dtype='bool')
        lr_edge_mask[local_edges] = False
        # split into local and lifted
        lifted_uvs = uvs[lr_edge_mask]
        lifted_costs = probs_to_costs(costs[lr_edge_mask], beta=self.beta_lifted)
        local_costs = probs_to_costs(costs[local_edges], beta=self.beta_lifted)
        # weight the local costs with lifted-to-local weight
        local_costs *= self.cost_weight
        segmentation = lifted_multicut(grid_graph.numberOfNodes,
                                       local_uvs,
                                       local_costs,
                                       lifted_uvs,
                                       lifted_costs).reshape(shape)
        if self.min_segment_size > 0:
            hmap = np.sum(affinities[:dim], axis=0) / dim
            segmentation, max_label = size_filter(hmap, segmentation, self.min_segment_size)
        else:
            max_label = segmentation.max()
        return segmentation, max_label

    def __call__(self, affinities):
        if self.stacked_2d:
            affinities_ = np.require(affinities[self.keep_channels], requirements='C')
            segmentation, _ = superpixel_stacked_from_affinities(affinities_,
                                                                 partial(self.lmc_superpixel, dim=2),
                                                                 self.n_threads)
        else:
            segmentation, _ = self.lmc_superpixel(affinities, dim=3)
        return segmentation
