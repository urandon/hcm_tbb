#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <limits>
#include <functional>

#include <tbb/tbb.h>



typedef std::vector<double> vtype;
typedef std::vector<vtype> pointstype;
typedef std::vector<std::vector<bool>> mtype;

typedef std::pair<double, vtype::const_iterator> argmintype;


void get_data(pointstype& data, std::istream& in = std::cin)
{
    int n_points, n_dim;
    in >> n_points >> n_dim;
    data.resize(n_points);

	for (auto & point : data) {
		point.resize(n_dim);
		for (int idim = 0; idim < n_dim; ++idim) {
			in >> point[idim];
		}
	}
}

void take_m_matrix(const mtype& m_matrix, std::ostream& out = std::cout)
{
    for (const auto & point : m_matrix)
    {
        for (const auto & belong_to_cluster : point) {
            out << belong_to_cluster << " ";
        }
	out << std::"\n";
    }
    out << std::flush;
}



enum SOLVE_HCM_STATUS {
	eOK,
	eDataWrong,
	eNotConverged
};

std::vector<std::string> SOLVE_HCM_STATUS_MSG = {
    "HCM Converged",
    "Data corrupt or internal problems",
    "HCM didn't converged"
};


int solve_hcm(mtype& m_matrix, const pointstype& data, int n_clusters, int max_iters, double eps)
{
   	int n_points = data.size();
	int n_dim = (data.size()) ? data[0].size() : 0;
	if (n_points == 0 || n_dim == 0) {
		return SOLVE_HCM_STATUS::eDataWrong;
	}
    m_matrix.resize(n_clusters, std::vector<bool>(n_points, false));

    // STEP 1: init centers
    pointstype centers(n_clusters, vtype(n_dim));
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> pts_distr(0, n_points - 1);

        tbb::parallel_for(0, n_clusters, [&](int cidx) {
            int pidx = pts_distr(gen);
            std::copy(data[pidx].begin(), data[pidx].end(), centers[cidx].begin());
        });
    }

    // main optimization cycle
    std::vector<int> cluster_mapping(n_points);
    std::vector<int> old_cluster_mapping(n_points);
    bool converged = false;
    double J_old = 0;
    for (int iter = 0; iter < max_iters && !converged; ++iter) {
        // STEP 2: find distances and minimum
        tbb::enumerable_thread_specific<vtype> tls_d(n_clusters);
        tbb::enumerable_thread_specific<double> J_tls(0);        
        tbb::parallel_for(0, n_points, [&](int pidx){
            vtype& d = tls_d.local();
            for (int cidx = 0; cidx < n_clusters; ++cidx) {
                d[cidx] = 0;
                for (int didx = 0; didx < n_dim; ++didx) {
                    double distance = data[pidx][didx] - centers[cidx][didx];
                    d[cidx] += distance * distance;
                }
            }

            // find min and argmin
            argmintype argmin = tbb::parallel_reduce(
                tbb::blocked_range<vtype::const_iterator>(d.begin(), d.end()),
                argmintype(DBL_MAX, d.end()),
                [](const tbb::blocked_range<vtype::const_iterator>& r, argmintype init) -> argmintype {
                    for (vtype::const_iterator a = r.begin(); a != r.end(); ++a) {
                        if (init.first > *a) {
                            init = std::make_pair(*a, a);
                        }                    
                    }
                    return init;
                },
                [](argmintype a, argmintype b) -> argmintype {
                    return (a.first < b.first) ? a : b;
                }
            );

            // update thread-local J and cluster mapping
            cluster_mapping[pidx] = argmin.second - d.begin();
            J_tls.local() += argmin.first;
        });

        // update M
        tbb::parallel_for(0, n_points, [&](int pidx) {
            m_matrix[old_cluster_mapping[pidx]][pidx] = false;
            m_matrix[cluster_mapping[pidx]][pidx] = true;
        });


        // STEP 3: evaluate quality functional (join thread locals)
        double J = J_tls.combine(std::plus<double>());

        // STEP 4: update cluster centers
        std::vector<int> cluster_counters(n_clusters, 0);
        tbb::parallel_for(0, n_clusters, [&](int cidx){
            int n_points_in_cluster = 0;
            const std::vector<bool>& m_matrix_cidx = m_matrix[cidx];
            vtype& center = centers[cidx];
            std::fill(center.begin(), center.end(), 0);

            tbb::parallel_for(0, n_points, [&](int pidx) {
                if (m_matrix_cidx[pidx]) {
                    const vtype& point = data[pidx];
                    n_points_in_cluster += 1;
                    tbb::parallel_for(0, n_dim, [&](int didx) {
                        center[didx] += point[didx];
                    });
                }
            });

            tbb::parallel_for(0, n_dim, [&](int didx) {
                center[didx] /= n_points_in_cluster;
            });
        });

        // check for convergency
        if (abs(J_old - J) < eps)
            converged = true;
        J_old = J;

        std::swap(cluster_mapping, old_cluster_mapping);
    }


	return (converged) ? SOLVE_HCM_STATUS::eOK : SOLVE_HCM_STATUS::eNotConverged;
}



int main(int argc, char ** argv)
{
    pointstype data;
    mtype m_matrix;

    int n_clusters;
    int max_iters = 20;
    double eps = 1e-3;
    int n_tbb_threads = -1;

    std::ios_base::sync_with_stdio(0);

    if (argc < 3-1 && argc > 5+1) {
        std::cout << "Usage: " << argv[0] << "nclusters infile ofile [nthreads] [max_iters] [eps]" << std::endl;
    }

    std::istringstream(argv[1]) >> n_clusters;
    if (argc > 4) std::istringstream(argv[4]) >> n_tbb_threads;
    if (argc > 5) std::istringstream(argv[5]) >> max_iters;
    if (argc > 6) std::istringstream(argv[6]) >> eps;

    tbb::task_scheduler_init task_scheduler;
    if (n_tbb_threads != -1) {
        task_scheduler.initialize(n_tbb_threads);
    }

    get_data(data, std::ifstream(argv[2]));
    int status = solve_hcm(m_matrix, data, n_clusters, max_iters, eps);
    std::cout << SOLVE_HCM_STATUS_MSG[status] << std::endl;
    take_m_matrix(m_matrix, std::ofstream(argv[3]));

    return 0;
}
