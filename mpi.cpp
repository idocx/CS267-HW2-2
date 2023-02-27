#include "common.h"
#include <cmath>
#include <fstream> 
#include <cstring> 
#include <mpi.h>
#include <vector>


// Put any static global variables here that you will use throughout the simulation.
typedef struct grid_class { 
    std::vector<particle_t> members; 
} grid_class;

// x is the row and y is the column.
int ngrid, nblock_x, nblock_y;

// the number of grids on X-axis and Y-axis for current processor
int ngrid_per_block_x, ngrid_per_block_y;

// the grid that each processor has
grid_class* grids;

MPI_Datatype GRID;
// the MPI request for sending and receiving the ghost grids

std::vector<particle_t> parts_to_send;

particle_t* recv_ls;
particle_t* parts_recv;
int* recv_counts;
int* recv_displs;

int get_part_grid_index(int gx, int gy, int rank) {
    // gx = (int)floor(part.x / cutoff) - block_x * ngrid_per_block_x_global + 1;
    // gy = (int)floor(part.y / cutoff) - block_y * ngrid_per_block_y_global + 1;
    int lx = (int)(gx / (ngrid_per_block_y - 2));
    int ly = (int)(gy / (ngrid_per_block_x - 2));
    int grid_id = lx * nblock_y + ly;
    if (rank != grid_id) return -1;
    lx = gx % (ngrid_per_block_y - 2) + 1;
    ly = gy % (ngrid_per_block_x - 2) + 1;
    return lx * ngrid_per_block_x + ly;
}

int get_part_ghost_grid_index(int gx, int gy, int rank) {
    int lx = (int)(gx / (ngrid_per_block_y - 2));
    int ly = (int)(gy / (ngrid_per_block_x - 2));
    int ghost_id = lx * nblock_y + ly;

    lx = gx % (ngrid_per_block_y - 2) + 1;
    ly = gy % (ngrid_per_block_x - 2) + 1;
    if (rank == ghost_id) {
        return lx * ngrid_per_block_x + ly;
    } else {
        int block_left = rank - 1;
        int block_top = rank - nblock_y;
        int block_right = rank + 1;
        int block_down = rank + nblock_y;

        int block_top_left = block_top - 1;
        int block_top_right = block_top + 1;
        int block_down_left = block_down - 1;
        int block_down_right = block_down + 1;
        if (ghost_id == block_left) {
            ly = 0;
        } else if (ghost_id == block_top) {
            lx = 0;
        } else if (ghost_id == block_right) {
            ly = ngrid_per_block_x - 1;
        } else if (ghost_id == block_down) {
            lx = ngrid_per_block_y - 1;
        } else if (ghost_id == block_top_left) {
            lx = ly = 0;
        } else if (ghost_id == block_top_right) {
            lx = 0;
            ly = ngrid_per_block_x - 1;
        } else if (ghost_id == block_down_right) {
            lx = ngrid_per_block_y - 1;
            ly = ngrid_per_block_x - 1;
        } else if (ghost_id == block_down_left) {
            lx = ngrid_per_block_y - 1;
            ly = 0;
        } else {
            return -1;
        }
    }
    return lx * ngrid_per_block_x + ly;
}

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, const particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    if (min_r * min_r > r2)
        r2 = min_r * min_r;

    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves enerly better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    if (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    if (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

static inline void apply_force_all(int gx, int gy, int rank, particle_t& part) {
    // for (int px = gx; px < gx + 3; px++) {
    //     for (int py = gy; py < gy + 3; py++) {
    //         // If the neighbor is outside the grid, let's abondon it.

    //         // Assign the grid for this neighbor.
    //         grid_class* grid = &grids[get_part_grid_id(px, py)];

    //         if (grid->num_p != 0) {
    //             for (int j = 0; j < MAX_P; j++) {
    //                 if (grid->members[j].id == 0) {
    //                     continue;
    //                 }
    //                 // If the neighbor is the particle itself, we don't need to consider the force.
    //                 if (part.id == grid->members[j].id) {
    //                     continue;
    //                 } 
    //                 apply_force(part, grid->members[j]);
    //             }
    //         }
    //     }
    // }

    for (int px = gx; px < gx + 3; ++px) {
        for (int py = gy; py < gy + 3; ++py) {
            // If the neighbor is outside the grid, let's abondon it.

            // Assign the grid for this neighbor.
            if (px >= ngrid || py >= ngrid) continue;

            int ind = get_part_ghost_grid_index(px, py, rank);

            for (const particle_t& particle : grids[ind].members) {
                // If the neighbor is the particle itself, we don't need to consider the force.
                if (part.id == particle.id)
                    continue;
                apply_force(part, particle);
            }
        }
    }
}


/*
This function will update the ghost grids on the boundary of the 
block so that we can compute the force without communicating with other blocks.
*/
void update_ghost_grid(int rank, int num_procs) {
    if (rank < nblock_y * nblock_x) {
        // update grid
        for (int i = 0; i < ngrid_per_block_y; i++) {
            for (int j = 0; j < ngrid_per_block_x; j++) {
                int ind = i * ngrid_per_block_x + j;

                // clear the boundary since we will add the ghosts
                if (i == 0 || i == ngrid_per_block_y - 1 || j == 0 || j == ngrid_per_block_x - 1) {
                    grids[ind].members.clear();
                    continue;
                }

                // for the current grid location
                grid_class* local_cell = &grids[ind];

                for (auto part_new = local_cell->members.begin(); part_new != local_cell->members.end(); part_new++) {
                    // search through the particles inside it
                    particle_t& part = *part_new;

                    int gx = (int)(part.x / cutoff);
                    int gy = (int)(part.y / cutoff);
                    // get the local grid id of the particle in this process
                    int new_ind = get_part_grid_index(gx, gy, rank);
                    if (ind == new_ind) {
                        if (1 == i || 1 == j || ngrid_per_block_y - 2 == i || ngrid_per_block_x - 2 == j) {
                            // if it is at the ghost boundary of the other processes, we will send it to other processes.
                            parts_to_send.push_back(part);
                        }
                        // we don't need to update it if it is at the same location as before
                        continue;
                    }

                    if (new_ind >= 0) {
                        // if it is at the current process's local grid, we will add it
                        int lx = gx % (ngrid_per_block_y - 2) + 1;
                        int ly = gy % (ngrid_per_block_x - 2) + 1;

                        if (1 == lx || 1 == ly || ngrid_per_block_y - 2 == lx || ngrid_per_block_x - 2 == ly) {
                            // if it is at the ghost boundary of the other processes, we will send it to other processes.
                            parts_to_send.push_back(part);
                        }
                        // otherwise, we update the part location
                        grids[new_ind].members.push_back(part);
                    } else {
                        // if it goes the ghost area of current process, we will also send it away
                        parts_to_send.push_back(part);
                    }
                    // eliminate the sent away particle from the current process
                    local_cell->members.erase(part_new--);
                }
            }
        }
    }

    memset(recv_counts, 0, num_procs * sizeof(int));
    memset(recv_displs, 0, num_procs * sizeof(int));

    // I decided to try the allgather here for the communication between different processes
    int num_parts_to_send = parts_to_send.size();
    // we first gather the number of sent particles of each process
    MPI_Allgather(&num_parts_to_send, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);
    // record the location of the shared particles with cumulative sum
    recv_displs[0] = 0;
    for (int i = 1; i < num_procs; i++)
        recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
    // then we gather the sent list of the particles from each process
    MPI_Allgatherv(&parts_to_send[0], parts_to_send.size(), PARTICLE, recv_ls, recv_counts,
                   recv_displs, PARTICLE, MPI_COMM_WORLD);
    if (rank < nblock_y * nblock_x) {
        for (int i = 0; i < recv_displs[num_procs - 1] + recv_counts[num_procs - 1]; ++i) {
            particle_t& part = recv_ls[i];
            if (part.id == 0) {
                continue;
            }

            int gx = (int)(part.x / cutoff);
            int gy = (int)(part.y / cutoff);
            int new_ind = get_part_ghost_grid_index(gx, gy, rank);
            bool update = true;
            if (new_ind >= 0)
                for (particle_t& p : grids[new_ind].members)
                    if (p.id == part.id) {
                        update = false;
                    }
            // if the current particle doesn't belong to this grid, we will add it to this grid
            if (new_ind >= 0 && update)
                grids[new_ind].members.push_back(part);
        }
    }
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here

    ngrid = (int)ceil(size / cutoff);
    nblock_y = nblock_x = floor(sqrt(num_procs));

    // int sqrt_num_procs = (int)ceil(sqrt(num_procs));
    // for (int i = sqrt_num_procs; i >= 1; i--) {
    //     if (num_procs % i == 0) {
    //         nblock_x = i;
    //         nblock_y = num_procs / i;
    //         break;
    //     }
    // }

    // add 2 for the boundary of the ghost
    ngrid_per_block_x = (int)ceil((double) ngrid / (double) nblock_x) + 2;
    ngrid_per_block_y = (int)ceil((double) ngrid / (double) nblock_y) + 2;

    recv_counts = (int*)malloc(num_procs * sizeof(int));
    recv_displs = (int*)malloc(num_procs * sizeof(int));
    grids = (grid_class*)calloc(ngrid_per_block_y * ngrid_per_block_x, sizeof(grid_class));


    // for (int i = 0; i < num_parts; i++) {
    //     int gx, gy;
    //     get_part_grid_index(parts[i], gx, gy);

    //     if (gx >= 1 && gx < ngrid_per_block_x + 1 && gy >= 1 && gy < ngrid_per_block_y + 1) {
    //         // note that we have ghost grids on the edges, so we need to add 1 to the index
    //         int grid_id = flatten_index(gx, gy, ngrid_per_block_y + 2);
    //         grids[grid_id].members[grids[grid_id].num_p] = parts[i];
    //         grids[grid_id].num_p++;
    //     }
    // }

    for (int i = 0; i < num_parts; ++i) {
        int gx = (int)(parts[i].x / cutoff);
        int gy = (int)(parts[i].y / cutoff);
        int ind = get_part_grid_index(gx, gy, rank);

        if (ind >= 0)
            grids[ind].members.push_back(parts[i]);
    }
    recv_ls = parts;

    // MPI_Barrier(MPI_COMM_WORLD);
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function
    parts_to_send.clear();
    update_ghost_grid(rank,num_procs);

    // MPI_Barrier(MPI_COMM_WORLD);

    // Compute Forces
    if (rank < nblock_y * nblock_x) {
        for (int i = 1; i < ngrid_per_block_y - 1; i++) {
            for (int j = 1; j < ngrid_per_block_x - 1; j++) {
                int ind = i * ngrid_per_block_x + j;
                grid_class* local_cell = &grids[ind];
                for (particle_t& part : local_cell->members) {
                    part.ax = part.ay = 0;
                    int gx = (int)(part.x / cutoff) - 1;
                    int gy = (int)(part.y / cutoff) - 1;
                    if (gx < 0) gx = 0;
                    if (gy < 0) gy = 0;
                    apply_force_all(gx, gy, rank, part);
                }
            }
        }

        // Move Particles
        for (int i = 1; i < ngrid_per_block_y - 1; i++) {
            for (int j = 1; j < ngrid_per_block_x - 1; j++) {
                int ind = i * ngrid_per_block_x + j;
                grid_class* local_cell = &grids[ind];
                for (particle_t& part : local_cell->members) {
                    move(part, size);
                }
            }
        }
    }
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
    
    // we will first send the number of particles to each processor
    // then we will send the particles to each processor
    // finally, we will gather the particles from each processor

    // int num_parts_to_send = 0;

    // for (int i = 0; i < ngrid_per_block_x; i++) {
    //     for (int j = 0; j < ngrid_per_block_y; j++) {
    //         int grid_id = get_part_grid_id(i + 1, j + 1);
    //         num_parts_to_send += grids[grid_id].num_p;
    //     }
    // }

    // particle_t* parts_to_send = (particle_t*)calloc(num_parts_to_send, sizeof(particle_t));

    parts_to_send.clear();
    if (0 == rank && parts_recv == NULL) {
        parts_recv = (particle_t*)malloc(num_parts * sizeof(particle_t));
        recv_ls = parts_recv;
    }
    memset(recv_counts, 0, num_procs * sizeof(int));
    memset(recv_displs, 0, num_procs * sizeof(int));

    // // collect the particles from each grid
    // int index = 0;
    // for (int i = 0; i < ngrid_per_block_x; i++) {
    //     for (int j = 0; j < ngrid_per_block_y; j++) {
    //         int grid_id = get_part_grid_id(i + 1, j + 1);
    //         for (int k = 0; k < MAX_P; k++) {
    //             if (grids[grid_id].members[k].id != 0) {
    //                 parts_to_send[index++] = grids[grid_id].members[k];
    //             }
    //         }
    //     }
    // }

    // int* recv_countss = NULL;
    // int* recv_displs = NULL;
    // particle_t* parts_recv = NULL;

    if (rank < nblock_y * nblock_x) {
        for (int i = 1; i < ngrid_per_block_y - 1; i++) {
            for (int j = 1; j < ngrid_per_block_x - 1; j++) {
                int ind = i * ngrid_per_block_x + j;
                grid_class* local_cell = &grids[ind];
                for (particle_t& part : local_cell->members) {
                    parts_to_send.push_back(part);
                }
            }
        }
    }
    int num_parts_to_send = parts_to_send.size();
    MPI_Gather(&num_parts_to_send, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);


    if (rank == 0) {
        for (int i = 0; i < num_procs; i++) {
            if (i == 0) {
                recv_displs[i] = 0;
            } else {
                recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
            }
        }
    }
    
    MPI_Gatherv(&parts_to_send[0], parts_to_send.size(), PARTICLE, recv_ls, recv_counts,
                recv_displs, PARTICLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < recv_displs[num_procs - 1] + recv_counts[num_procs - 1]; ++i) {
            particle_t* part = &recv_ls[i];
            if (part->id != 0) {
                parts[part->id - 1] = *part;
            }
        }
    }

    // if (rank == 0) {
    //     for (int i = 0; i < num_parts; i++) {
    //         uint64_t id = parts_recv[i].id;
    //         parts[id - 1] = parts_recv[i];
    //     }

    //     free(recv_counts);
    //     free(recv_displs);
    //     free(parts_recv);
    // }

    // free(parts_to_send);

}