#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include <math.h>


#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xadapt.hpp"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyvectorize.hpp"

#include <iostream>
#include <numeric>
#include <cmath>
#include <map>
#include <set>
#include <unordered_set>
#include <random>
//#include <boost/functional/hash.hpp>

#include <iterator>
//#include <boost/graph/adjacency_list.hpp>

namespace py = pybind11;

struct MutexWatershed {
    // A Watershed Segmentation Execution Engine
    // Given a ground truth segmentation the constrained Segmentation can be computed in parallel.

    MutexWatershed(
        xt::pytensor<int64_t, 1> image_shape, 
        xt::pytensor<int64_t, 2> offsets,
        uint64_t num_attractive_channels, 
        xt::pytensor<uint64_t, 1> dam_stride
    )
     : 
        image_shape(image_shape), 
        offsets(offsets), 
        num_attractive_channels(num_attractive_channels), 
        dam_stride(dam_stride)
     {

        n_points = xt::prod(image_shape)(0);
        ndims = image_shape.size();
        action_counter = 0;
        directions = offsets.shape()[0];

        strides = xt::zeros<int64_t>({directions});
        for (uint64_t d = 0; d < directions; ++d) {
            int64_t this_stride = 0;
            int64_t total_stride = 1;
            for (int64_t n = ndims-1; n >= 0; --n) {
                this_stride += offsets(d, n) * total_stride;
                total_stride *= image_shape(n);
            }
            strides(d) = this_stride;
        }

        // reset all arrays to initial values
        clear_all();
        set_bounds();
     }

    // TODO: Could be made constant
    uint64_t n_points;
    int64_t ndims;
    uint64_t connectivity;
    uint64_t directions;

    xt::pytensor<int64_t,1> image_shape;
    xt::pytensor<int64_t,2> offsets;
    xt::pytensor<int64_t,1> strides;
    uint64_t num_attractive_channels;
    xt::pytensor<uint64_t, 1> dam_stride;
    xt::xarray<bool> bounds;

    // pointers to union find data structures
    // can be switched between unconstrained and constrained arrays
    // (indicated by uc and c respectively)
    xt::pytensor<uint64_t,1> * parent;
    xt::pytensor<uint64_t,1> * rank;
    xt::pytensor<uint64_t,1> * region_size;
    xt::pytensor<int64_t,1> * actions;

    // union find and bookkeeping arrays
    xt::pytensor<uint64_t,1> uc_parent;
    xt::pytensor<uint64_t,1> uc_rank;
    xt::pytensor<uint64_t,1> uc_region_size;

    // bookkeeping variables
    uint64_t action_counter;
    bool finished;
    xt::pyarray<bool> seen_actions;
    int64_t error_count;

    // constrained data
    xt::pytensor<uint64_t,1> c_parent;
    xt::pytensor<uint64_t,1> c_rank;
    xt::pytensor<uint64_t,1> c_region_size;
    xt::pytensor<int64_t,1> c_actions;
    xt::pytensor<int64_t,1> label_image;
    xt::pytensor<int64_t,1> region_gt_label;
    xt::pytensor<int64_t,1> uc_actions;

    // one directional adjacency list
    // maps root node to random pixel in connected component
    // this makes checking for an existing link more expensive,
    // but saves resources since it does not require relabeling on 
    // graph contraction
    std::vector<std::vector<int64_t>> dam_graph;
    bool active_constraints;
    bool has_gt_labels;
    bool use_tollerant_check;

    // buffer for merged dams
    std::vector<int64_t> merged_dams_buffer_;

    void clear() {
        // reset minimal spanning tree
        action_counter = 0;
        
        uc_parent = xt::arange(n_points);
        uc_rank = xt::zeros<uint64_t>({n_points});
        uc_region_size = xt::ones<uint64_t>({n_points});

        c_region_size = xt::ones<uint64_t>({n_points});
        uc_actions = xt::zeros<int64_t>({n_points * (directions)});
        seen_actions = xt::zeros<bool>({n_points * (directions)});
        has_gt_labels = false;
        use_tollerant_check = false;
        finished = false;
        error_count = 0;
    }

    void c_clear() {
        // reset constrained minimal spanning tree
        c_parent = xt::arange(n_points);
        c_rank = xt::zeros<uint64_t>({n_points});
        c_actions = xt::zeros<int64_t>({n_points * (directions)});

        region_size = &uc_region_size;
        active_constraints = false;
        dam_graph = std::vector<std::vector<int64_t>>(n_points);
    }

    void clear_all() {
        set_uc();
        clear();
        c_clear();
    }

    void set_bounds(){
    	std::vector<uint64_t> s;
        for (uint64_t n = 0; n < ndims; ++n)
           s.push_back(image_shape(n));
        s.push_back(directions);
        bounds = xt::zeros<bool>(s);

        if (ndims == 2){
            fast_2d_set_bounds();
        }
        else if(ndims == 3){
            fast_3d_set_bounds();
        }
        else{
            std::cout << "WARNING: fallback to slow bound computation because image dimensions " << ndims << " != 2 or 3" << std::endl;
            bounds.reshape({n_points, directions});
            slow_set_bounds();
            return;
        }
        bounds.reshape({n_points, directions});
    }

    void fast_2d_set_bounds(){
        xt::view(bounds, xt::range(0, image_shape(0)-1, int64_t(dam_stride(0))),
                         xt::range(0, image_shape(1)-1, int64_t(dam_stride(1))),
                         xt::all()) = 1;

        xt::view(bounds, xt::all(),
                         xt::all(),
                         xt::range(0, num_attractive_channels, int64_t(1))) = 1;

        for (uint64_t d = 0; d < directions; ++d) {
            if (offsets(d, 0) > 0)
                xt::view(bounds, xt::range(image_shape(0)-1, image_shape(0)-offsets(d, 0)-1, int64_t(-1)), xt::all(), d) = 0;
            else if (offsets(d, 0) < 0)
                xt::view(bounds, xt::range(0., -offsets(d, 0), 1), xt::all(), d) = 0;
            if (offsets(d, 1) > 0)
                xt::view(bounds, xt::all(), xt::range(image_shape(1)-1, image_shape(1)-offsets(d, 1)-1, int64_t(-1)), d) = 0;
            else if (offsets(d, 1) < 0)
                xt::view(bounds, xt::all(), xt::range(0., -offsets(d, 1), 1), d) = 0;
        }
    }

    void fast_3d_set_bounds(){

        xt::view(bounds, xt::range(0, image_shape(0)-1, int64_t(dam_stride(0))),
                         xt::range(0, image_shape(1)-1, int64_t(dam_stride(1))),
                         xt::range(0, image_shape(2)-1, int64_t(dam_stride(2))),
                         xt::all()) = 1;

        xt::view(bounds, xt::all(),
                         xt::all(),
                         xt::all(),
                         xt::range(0, num_attractive_channels, int64_t(1))) = 1;

        for (uint64_t d = 0; d < directions; ++d) {
            if (offsets(d, 0) > 0)
                xt::view(bounds, xt::range(image_shape(0)-1, image_shape(0)-offsets(d, 0)-1, int64_t(-1)), xt::all(), xt::all(), d) = 0;
            else if (offsets(d, 0) < 0)
                xt::view(bounds, xt::range(0., -offsets(d, 0), 1.), xt::all(), xt::all(), d) = 0;
            if (offsets(d, 1) > 0)
                xt::view(bounds, xt::all(), xt::range(image_shape(1)-1, image_shape(1)-offsets(d, 1)-1, int64_t(-1)), xt::all(), d) = 0;
            else if (offsets(d, 1) < 0)
                xt::view(bounds, xt::all(), xt::range(0., -offsets(d, 1), 1.), xt::all(), d) = 0;
            if (offsets(d, 2) > 0)
                xt::view(bounds, xt::all(), xt::all(), xt::range(image_shape(2)-1, image_shape(2)-offsets(d, 2)-1, int64_t(-1)), d) = 0;
            else if (offsets(d, 2) < 0)
                xt::view(bounds, xt::all(), xt::all(), xt::range(0., -offsets(d, 2), 1.), d) = 0;
        }
    }

    void slow_set_bounds(){
        for (uint64_t i = 0; i < n_points; ++i) {
            for (uint64_t ds = 0; ds < directions; ++ds) {
                bounds(i, ds) = is_valid_edge(i, ds);
            }
        }
    }

    void compute_randomized_bounds(){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> rng;
        uint64_t stride_product = 1;
        for (int64_t n = 0; n < ndims; ++n)
            stride_product *= dam_stride(n);

        // NOTE: we have to substract 1 from the node product, because
        // uniform_distributions range is inclusive on both sides
        rng = std::uniform_int_distribution<>(0, stride_product - 1);

        // set dam stride to 1 temporarily since it is replaced by rng
        auto real_stride = dam_stride;
        for (auto& d : dam_stride)
            d = 1;

        for (uint64_t i = 0; i < n_points; ++i) {
            for (uint64_t d = num_attractive_channels; d < directions; ++d){
                if (is_valid_edge(i, d)){
                    bounds(i, d) = (rng(gen) == 0);
                }
            }
        }
        dam_stride = real_stride;
    }

    uint64_t find(uint64_t i) {
        if ((*parent)(i) == i) {
            return i;
        }
        else{
            (*parent)(i) = find((*parent)(i));
            return (*parent)(i);
        }
    }

    inline uint64_t _get_direction( uint64_t e) {
        return e / n_points;
    }

    inline uint64_t _get_position( uint64_t e) {
        return e % n_points;
    }

    auto get_flat_label_projection() {
        //  this function generates an id-invariant projection of the current segmentation
        //  which can be used as an input for a neural network

        // set dam stride to 1 temporarily to avoid image artifacts
        auto real_stride = dam_stride;
        for (auto& d : dam_stride)
            d = 1;

        xt::pytensor<float,1> region_projection = xt::zeros<float>({n_points* directions});

        for (uint64_t i = 0; i < n_points; ++i) {
            uint64_t root_i = find(i);
            for (uint64_t d = 0; d < directions; ++d) {
                if (check_bounds(i, d))
                    if (root_i == find(i+strides(d)))
                        region_projection(d*n_points + i) = 1.;
            }
        }
        dam_stride = real_stride;
        return region_projection;
    }

    void set_state( const MutexWatershed & ouf) {
        // copies segmentation state from other MutexWatershed object
        ndims = ouf.ndims;
        action_counter = ouf.action_counter;
        strides = ouf.strides;
        n_points = ouf.n_points;
        
        uc_rank = ouf.uc_rank;
        c_rank = ouf.c_rank;
        // uc_rank = &rank;

        uc_region_size =  ouf.uc_region_size;
        c_region_size =  ouf.c_region_size;
        
        uc_parent =  ouf.uc_parent;
        c_parent =  ouf.c_parent;
        // uc_parent =  &parent;
        region_gt_label = ouf.region_gt_label;
        
        image_shape = ouf.image_shape;
        c_actions = ouf.c_actions;
        uc_actions = ouf.uc_actions;
        // actions = &uc_actions;
        has_gt_labels = ouf.has_gt_labels;
        seen_actions = ouf.seen_actions;
        dam_graph = ouf.dam_graph;

        set_uc();
    }

    inline uint64_t is_valid_edge(uint64_t i, uint64_t k) {
        int64_t index = i;
        for (int64_t n = ndims-1; n >= 0; --n)
        {
            if (k >= num_attractive_channels)
                if (index % int64_t(dam_stride(n)) != 0)
                    return false;
            if (offsets(k, n) != 0) {

                int64_t coord = index % image_shape(n);
                if (offsets(k, n) < 0) {
                    if ( coord < -offsets(k, n))
                        return false;
                }
                else{
                    if (coord  >= image_shape(n)-offsets(k, n))
                        return false;
                }
            }
            index /= image_shape(n);
        }
        return true;
    }

    inline bool check_bounds(uint64_t i, uint64_t d){
        return bounds(i, d);
    }

    void set_c() {
        parent = &c_parent;
        rank = &c_rank;
        region_size = &c_region_size;
        active_constraints = true;
    }

    void set_uc() {
        parent = &uc_parent;
        rank = &uc_rank;
        region_size = &uc_region_size;
        active_constraints = false;
    }

    inline void merge_dams(uint64_t root_from, uint64_t root_to) {
        if (dam_graph[root_from].size() == 0)
            return;

        if (dam_graph[root_to].size() == 0){
            dam_graph[root_to] = dam_graph[root_from];
            return;
        }

        merged_dams_buffer_.resize(0);
        merged_dams_buffer_.reserve(std::max(dam_graph[root_from].size(), dam_graph[root_to].size()));

        std::merge(dam_graph[root_from].begin(), dam_graph[root_from].end(),
            dam_graph[root_to].begin(), dam_graph[root_to].end(), std::back_inserter(merged_dams_buffer_));

        dam_graph[root_to] = merged_dams_buffer_;
        dam_graph[root_from].clear();
    }

    inline bool is_dam_constrained(uint64_t root_i, uint64_t root_j){
        auto de_i = dam_graph[root_i].begin();
        auto de_j = dam_graph[root_j].begin();

        while (de_i != dam_graph[root_i].end() && de_j != dam_graph[root_j].end()) {
            if (*de_i < *de_j) {
                ++de_i;
            } else  {
                if (!(*de_j < *de_i)) {
                    return true;
                }
                ++de_j;
            }
        }
        return false;
    }

    inline bool merge_roots(uint64_t root_i, uint64_t root_j) {
        if (!active_constraints) {
            if (is_dam_constrained(root_i, root_j)) return false;
            merge_dams(root_i, root_j);
        }
        else{
            throw std::runtime_error("not implemented");
        }

        // merge regions
        (*parent)(root_i) = root_j;
        (*region_size)(root_j) += (*region_size)(root_i);
        return true;
    }
    

    inline bool dam_constrained_merge(uint64_t i, uint64_t j) {
        uint64_t root_i = find(i);
        uint64_t root_j = find(j);
    
        if (root_i != root_j) {
            if ((*rank)(root_i) < (*rank)(root_j)) {
                return merge_roots(root_i, root_j);
            }
            else if ((*rank)(root_i) > (*rank)(root_j)) {
                return merge_roots(root_j, root_i);
            }
            else{
                if (merge_roots(root_i, root_j)) {   
                    (*rank)(root_j) += 1;
                    return true;
                }
                return false;
            }
        }
        else{
            return false;
        }
    }

    inline bool add_dam_edge(uint64_t i, uint64_t j, uint64_t dam_edge) {
        uint64_t root_i = find(i);
        uint64_t root_j = find(j);
        if (root_i != root_j) {
            if (!is_dam_constrained(root_i, root_j)){
                dam_graph[root_i].insert(std::upper_bound(dam_graph[root_i].begin(), dam_graph[root_i].end(), dam_edge), dam_edge);
                dam_graph[root_j].insert(std::upper_bound(dam_graph[root_j].begin(), dam_graph[root_j].end(), dam_edge), dam_edge);
                return true;
            }
        }
        return false;
    }

    void repulsive_mst_cut(const xt::pyarray<long> & edge_list) {
        for (auto& e : edge_list) {
            uint64_t i = _get_position(e);
            uint64_t d = _get_direction(e);
            if (check_bounds(i, d)) {
                int64_t j = int64_t(i) + strides(d);
                if (d < num_attractive_channels) {
                    bool a = dam_constrained_merge(i, j);
                    uc_actions(e) = a;
                }
                else{
                    uc_actions(e) = add_dam_edge(i, j, e);
                }
            }
        }
    }

    void repulsive_ucc_mst_cut(const xt::pyarray<long> & edge_list, uint64_t num_iterations) {
        finished = true;
        action_counter = 0;
        error_count = 0;
        for (auto& e : edge_list) {
            if (num_iterations != 0 and num_iterations <= action_counter)
                finished = false;

            if (!seen_actions(e)) {
                seen_actions(e) = true;
                uint64_t i = _get_position(e);
                uint64_t d = _get_direction(e);


                if (check_bounds(i, d)) {
                    if (d < num_attractive_channels) {
                        int64_t j = int64_t(i) + strides(d);

                        // unconstrained merge
                        set_uc();
                        bool a = dam_constrained_merge(i, j);
                        if (a) {
                            ++action_counter;
                        }
                        uc_actions(e) = a;
                    }
                    else{
                        int64_t j = int64_t(i) + strides(d);
                        set_uc();
                        uc_actions(e) = add_dam_edge(i, j, e);
                    }
                }
            }
        }
    }

    bool is_finised() {
        return finished;
    }

    /////////////// get functions ////////////////////

    auto get_seen_actions() {
        return seen_actions;
    }

    auto get_flat_label_image_only_merged_pixels() {
        set_uc();
        xt::pyarray<uint64_t> label = xt::zeros<uint64_t>({n_points});
        for (uint64_t i = 0; i < n_points; ++i)
        {
            uint64_t root_i = find(i);
            uint64_t size = (*region_size)(root_i);
            if (size > 1) {
                label(i) = root_i+1;
            }
            else{
                label(i) = 0;
            }
        }
        return label;
    }

    auto get_flat_uc_label_image_only_merged_pixels() {
        set_uc();
        auto lab_img = get_flat_label_image_only_merged_pixels();
        return lab_img;
    }

    auto get_flat_c_label_image_only_merged_pixels() {
        set_c();
        auto lab_img = get_flat_label_image_only_merged_pixels();
        set_uc();
        return lab_img;
    }

    auto get_flat_label_image() {
        xt::pyarray<long> label = xt::zeros<long>({n_points});
        for (uint64_t i = 0; i < n_points; ++i)
        {
            label(i) = find(i)+1;
        }
        return label;
    }

    auto get_action_counter() {
        return action_counter;
    }

    auto get_flat_c_label_image() {
        set_c();
        auto lab_img = get_flat_label_image();
        set_uc();
        return lab_img;
    }

    auto get_flat_uc_label_image() {
        set_uc();
        return get_flat_label_image();
    }

    auto get_flat_applied_action() {   
        return actions;
    }

    auto get_flat_applied_uc_actions() {
        return uc_actions;
    }

    auto get_flat_applied_c_actions() {
        return c_actions;
    }
};


// Python Module and Docstrings

PYBIND11_PLUGIN(mutex_watershed)
{
    xt::import_numpy();

    py::module m("mutex_watershed", R"docu(
        Mutex watershed

        .. currentmodule:: mutex_watershed

        .. autosummary:
           :toctree: _generate

        MutexWatershed
    )docu");

    py::class_<MutexWatershed>(m, "MutexWatershed")
        .def(py::init<xt::pytensor<int64_t,1> , xt::pytensor<int64_t,2> , uint64_t, xt::pytensor<uint64_t,1>>())
        .def("clear_all", &MutexWatershed::clear_all)
        .def("get_flat_label_image_only_merged_pixels", &MutexWatershed::get_flat_label_image_only_merged_pixels)
        .def("get_flat_uc_label_image_only_merged_pixels", &MutexWatershed::get_flat_uc_label_image_only_merged_pixels)
        .def("get_flat_c_label_image_only_merged_pixels", &MutexWatershed::get_flat_c_label_image_only_merged_pixels)
        .def("get_flat_label_image", &MutexWatershed::get_flat_label_image)
        .def("get_flat_uc_label_image", &MutexWatershed::get_flat_uc_label_image)
        .def("get_flat_c_label_image", &MutexWatershed::get_flat_c_label_image)
        .def("get_flat_label_projection", &MutexWatershed::get_flat_label_projection)
        .def("set_state",  &MutexWatershed::set_state)
        .def("set_bounds",  &MutexWatershed::set_bounds)
        .def("check_bounds",  &MutexWatershed::check_bounds)
        .def("compute_randomized_bounds",  &MutexWatershed::compute_randomized_bounds)
        .def("repulsive_mst_cut",  &MutexWatershed::repulsive_mst_cut)
        .def("repulsive_ucc_mst_cut",  &MutexWatershed::repulsive_ucc_mst_cut)
        .def("set_uc", &MutexWatershed::set_uc)
        .def("check_bounds", &MutexWatershed::check_bounds)
        .def("is_finised", &MutexWatershed::is_finised)
        .def("get_seen_actions", &MutexWatershed::get_seen_actions)

        .def("get_action_counter", &MutexWatershed::get_action_counter)
        .def("get_flat_applied_action", &MutexWatershed::get_flat_applied_action)
        .def("get_flat_applied_uc_actions", &MutexWatershed::get_flat_applied_uc_actions)
        .def("get_flat_applied_c_actions", &MutexWatershed::get_flat_applied_c_actions);

    return m.ptr();
}


