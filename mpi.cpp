#include "common.h"
#include <mpi.h>
#include <map>
#include <cstddef>
#include <chrono>
#include <cmath>
#include <vector>
#include <cstdio>

# define MAX_P 5

int nblock_x, nblock_y;
int ngrid;
// the number of grids on X-axis and Y-axis for current processor
int ngrid_per_block_x, ngrid_per_block_y;
// the number of grids on X-axis and Y-axis for all processors
int ngrid_per_block_x_global, ngrid_per_block_y_global;

int block_x, block_y;

// Put any static global variables here that you will use throughout the simulation.
typedef struct grid_class {
    int num_p;
    particle_t members[MAX_P]; 
} grid_class;


// the grid that each processor has
grid_class* grids;

MPI_Datatype GRID;
// the MPI request for sending and receiving the ghost grids

int flatten_index(int x, int y, int y_size) {
    return x * y_size + y;
}

int* unflatten_index(int index, int y_size) {
    int* result = (int*)malloc(2 * sizeof(int));
    result[0] = index / y_size;
    result[1] = index % y_size;
    return result;
}

// get particle's grid position based on its position in current process
void get_part_grid_index(particle_t& part, int& gx, int& gy) {
    gx = (int)floor(part.x / cutoff) - block_x * ngrid_per_block_x_global + 1;
    gy = (int)floor(part.y / cutoff) - block_y * ngrid_per_block_y_global + 1;
}

// get particle's grid index based on its position in current process
int get_part_grid_id(int gx, int gy) {
    int grid_id = flatten_index(gx, gy, ngrid_per_block_y + 2);
    return grid_id;
}

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

static inline void apply_force_all(int gx, int gy, particle_t& part) {
    for (int px = gx; px < gx + 3; px++) {
        for (int py = gy; py < gy + 3; py++) {
            // If the neighbor is outside the grid, let's abondon it.
            if (px < 0 || px >= ngrid_per_block_x + 2 || py < 0 || py >= ngrid_per_block_y + 2) 
                // printf("Error: the neighbor is outside the grid\n");
                printf("Error: px: %d, py: %d is out of the bound.\n", px, py);

            // Assign the grid for this neighbor.
            grid_class* grid = &grids[flatten_index(px, py, ngrid_per_block_y + 2)];

            if (grid->num_p > 0) {
                for (int j = 0; j < MAX_P; j++) {
                    if (grid->members[j].id == 0) {
                        continue;
                    }
                    // If the neighbor is the particle itself, we don't need to consider the force.
                    if (part.id == grid->members[j].id){
                        continue;
                    } 
                    apply_force(part, grid->members[j]);
                }
            }
        }
    }
}

void send_grid(grid_class* grid, int length, int dest, MPI_Request* request, int tag) {
    MPI_Isend(grid, length, GRID, dest, tag, MPI_COMM_WORLD, request);
}

void receive_grid(grid_class* grid, int length, int source, MPI_Request* request, int tag) {
    MPI_Irecv(grid, length, GRID, source, tag, MPI_COMM_WORLD, request);
}

/*
This function will update the ghost grids on the boundary of the 
block so that we can compute the force without communicating with other blocks.
*/
void update_ghost_grid() {
    int cnt = 0;
    MPI_Request* request = (MPI_Request*)calloc(12 + ngrid_per_block_x * 4, sizeof(MPI_Request));

    // Send and receive the grids on the top and bottom boundary
    if (block_y != 0) {
        for (int i = 0; i < ngrid_per_block_x; i++) {
            send_grid(grids + flatten_index(i + 1, 1, ngrid_per_block_y + 2), 1, flatten_index(block_x, block_y - 1, nblock_y), request+cnt++, i+1);
            receive_grid(grids + flatten_index(i + 1, 0, ngrid_per_block_y + 2), 1,flatten_index(block_x, block_y - 1, nblock_y), request+cnt++, i+1);
        }
    }
    if (block_y != nblock_y - 1) {
        for (int i = 0; i < ngrid_per_block_x; i++) {
            send_grid(grids + flatten_index(i + 1, ngrid_per_block_y, ngrid_per_block_y + 2), 1, flatten_index(block_x, block_y + 1, nblock_y), request+cnt++, i+1);
            receive_grid(grids + flatten_index(i + 1, ngrid_per_block_y + 1, ngrid_per_block_y + 2), 1, flatten_index(block_x, block_y + 1, nblock_y), request+cnt++, i+1);
        }
    }

    // Send and receive the grids on the left and right boundary
    if (block_x != 0) {
        send_grid(grids + flatten_index(1, 1, ngrid_per_block_y + 2), ngrid_per_block_y, flatten_index(block_x - 1, block_y, nblock_y), request+cnt++, 0);
        receive_grid(grids + flatten_index(0, 1, ngrid_per_block_y + 2), ngrid_per_block_y, flatten_index(block_x - 1, block_y, nblock_y), request+cnt++, 0);
    }

    if (block_x != nblock_x - 1) {
        send_grid(grids + flatten_index(ngrid_per_block_x, 1, ngrid_per_block_y + 2), ngrid_per_block_y, flatten_index(block_x + 1, block_y, nblock_y), request+cnt++, 0);
        receive_grid(grids + flatten_index(ngrid_per_block_x + 1, 1, ngrid_per_block_y + 2), ngrid_per_block_y, flatten_index(block_x + 1, block_y, nblock_y), request+cnt++, 0);
    }

    // Send and receive the grids on the corner
    if (block_x != 0 && block_y != 0) {
        send_grid(grids + flatten_index(1, 1, ngrid_per_block_y + 2), 1, flatten_index(block_x - 1, block_y - 1, nblock_y), request+cnt++, 0);
        receive_grid(grids + flatten_index(0, 0, ngrid_per_block_y + 2), 1, flatten_index(block_x - 1, block_y - 1, nblock_y), request+cnt++, 0);
    }
    if (block_x != nblock_x - 1 && block_y != 0) {
        send_grid(grids + flatten_index(ngrid_per_block_x, 1, ngrid_per_block_y + 2), 1, flatten_index(block_x + 1, block_y - 1, nblock_y), request+cnt++, 0);
        receive_grid(grids + flatten_index(ngrid_per_block_x + 1, 0, ngrid_per_block_y + 2), 1, flatten_index(block_x + 1, block_y - 1, nblock_y), request+cnt++, 0);
    }
    if (block_x != 0 && block_y != nblock_y - 1) {
        send_grid(grids + flatten_index(1, ngrid_per_block_y, ngrid_per_block_y + 2), 1, flatten_index(block_x - 1, block_y + 1, nblock_y), request+cnt++, 0);
        receive_grid(grids + flatten_index(0, ngrid_per_block_y + 1, ngrid_per_block_y + 2), 1, flatten_index(block_x - 1, block_y + 1, nblock_y), request+cnt++, 0);
    }
    if (block_x != nblock_x - 1 && block_y != nblock_y - 1) {
        send_grid(grids + flatten_index(ngrid_per_block_x, ngrid_per_block_y, ngrid_per_block_y + 2), 1, flatten_index(block_x + 1, block_y + 1, nblock_y), request+cnt++, 0);
        receive_grid(grids + flatten_index(ngrid_per_block_x + 1, ngrid_per_block_y + 1, ngrid_per_block_y + 2), 1, flatten_index(block_x + 1, block_y + 1, nblock_y), request+cnt++, 0);
    }

    // Wait for all the grids to be received
    MPI_Waitall(cnt, request, MPI_STATUSES_IGNORE);

    // Free the requests
    free(request);
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here
    // Create MPI Grid Type
    const int nitems = 2;
    int blocklengths[2] = {1, MAX_P};
    MPI_Datatype types[2] = {MPI_INT, PARTICLE};
    MPI_Aint offsets[2];
    offsets[0] = offsetof(grid_class, num_p);
    offsets[1] = offsetof(grid_class, members);
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &GRID);
    MPI_Type_commit(&GRID);

    ngrid = (int)ceil(size / cutoff);

    // we will divide the space into block_size[0] * block_size[1] blocks
    int sqrt_num_procs = (int)ceil(sqrt(num_procs));
    for (int i = sqrt_num_procs; i >= 1; i--) {
        if (num_procs % i == 0) {
            nblock_x = i;
            nblock_y = num_procs / i;
            break;
        }
    }

    // each block will have ngrid_per_block_x * ngrid_per_block_y grids
    ngrid_per_block_x_global = (int)ceil((double) ngrid / (double) nblock_x);
    ngrid_per_block_y_global = (int)ceil((double) ngrid / (double) nblock_y);

    // the position of the block in the block grid
    int* block_pos = unflatten_index(rank, nblock_y);
    block_x = block_pos[0];
    block_y = block_pos[1];

    if ((block_x + 1) * ngrid_per_block_x_global > ngrid) {
        ngrid_per_block_x = ngrid - block_x * ngrid_per_block_x_global;
    } else {
        ngrid_per_block_x = ngrid_per_block_x_global;
    }

    if ((block_y + 1) * ngrid_per_block_y_global > ngrid) {
        ngrid_per_block_y = ngrid - block_y * ngrid_per_block_y_global;
    } else {
        ngrid_per_block_y = ngrid_per_block_y_global;
    }

    // printf("ngrid = %d, nblock_x = %d, nblock_y = %d, ngrid_per_block_x_global = %d, ngrid_per_block_y_global = %d, block_x = %d, block_y = %d, ngrid_per_block_x = %d, ngrid_per_block_y = %d\n", ngrid, nblock_x, nblock_y, ngrid_per_block_x_global, ngrid_per_block_y_global, block_x, block_y, ngrid_per_block_x, ngrid_per_block_y);

    grids = (grid_class*)calloc((ngrid_per_block_x + 2) * (ngrid_per_block_y + 2), sizeof(grid_class));

    // TODO : parallelize this assigning process
    // go through all the particles, if they are in this block, assign them to the grids
    // the ghost grids are left empty
    for (int i = 0; i < num_parts; i++) {
        int gx, gy;
        get_part_grid_index(parts[i], gx, gy);

        if (gx >= 1 && gx < ngrid_per_block_x + 1 && gy >= 1 && gy < ngrid_per_block_y + 1) {
            // note that we have ghost grids on the edges, so we need to add 1 to the index
            int grid_id = flatten_index(gx, gy, ngrid_per_block_y + 2);
            grids[grid_id].members[grids[grid_id].num_p] = parts[i];
            grids[grid_id].num_p++;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // // Exchange the ghost atoms with the neighboring blocks
    // update_ghost_grid();

    // Compute the forces on each particle
    for (int i = 0; i < ngrid_per_block_x; i++) {
        for (int j = 0; j < ngrid_per_block_y; j++) {
            int grid_id = get_part_grid_id(i+1, j+1);
            if (0 != grids[grid_id].num_p) {
                for (int k = 0; k < MAX_P; k++) {
                    if (grids[grid_id].members[k].id == 0) continue;
                    particle_t part = grids[grid_id].members[k];
                    apply_force_all(i, j, part);
                }
            }
        }
    }

    std::map<int, std::vector<particle_t>> moved_particles;

    for (int i = 0; i < ngrid_per_block_x; i++) {
        for (int j = 0; j < ngrid_per_block_y; j++) {
            int grid_id = get_part_grid_id(i+1, j+1);
            grid_class grid = grids[grid_id];
            
            if (0 != grid.num_p) {
                for (int k = 0; k < MAX_P; k++) {
                    if (grid.members[k].id == 0) continue;
                    move(grid.members[k], size);

                    // If the particle is out of the block, we need to move it to the moved_particles array
                    int gx_global, gy_global;
                    int gx, gy;

                    get_part_grid_index(grid.members[k], gx, gy);

                    gx_global = block_x * ngrid_per_block_x_global + gx - 1;
                    gy_global = block_y * ngrid_per_block_y_global + gy - 1;

                    // calculate the block index of the particle
                    int bx = (int)floor((double) gx_global / (double) ngrid_per_block_x_global);
                    int by = (int)floor((double) gy_global / (double) ngrid_per_block_y_global);

                    // if the particle is out of the block, we need to move it to the moved_particles array
                    if (bx != block_x || by != block_y) {
                        int block_id = flatten_index(bx, by, nblock_y);
                        particle_t part = grid.members[k];
                        moved_particles[block_id].push_back(part);  
                        grid.members[k].id = 0;
                    } 
                    // if the particle is still in the block, we need to update the grid
                    else {
                        int grid_id = get_part_grid_id(gx, gy);
                        
                        if (gx == i + 1 && gy == j + 1) continue;

                        for (int kk = 0; kk < MAX_P; kk++) {
                            if (grids[grid_id].members[kk].id == 0) {
                                grids[grid_id].members[kk] = grid.members[k];
                                grids[grid_id].num_p++;
                                grid.members[k].id = 0;
                                grid.num_p--;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // redistribute the moved particles to its correct grid
    // use MPI_alltoall to send the particle number,
    // then use MPI_alltoallv to send the particles
    int* send_counts = (int*)calloc(num_procs, sizeof(int));
    int* recv_counts = (int*)calloc(num_procs, sizeof(int));
    int* send_displs = (int*)calloc(num_procs, sizeof(int));
    int* recv_displs = (int*)calloc(num_procs, sizeof(int));

    for (int i = 0; i < num_procs; i++) {
        if (moved_particles.find(i) != moved_particles.end()) {
            send_counts[i] = moved_particles[i].size();
        } else {
            send_counts[i] = 0;
        }

        if (i == 0) {
            send_displs[i] = 0;
        } else {
            send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
        }
    }

    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

    for (int i = 0; i < num_procs; i++) {
        if (i == 0) {
            recv_displs[i] = 0;
        } else {
            recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
        }
    }

    int num_parts_sent = send_displs[num_procs - 1] + send_counts[num_procs - 1];
    int num_parts_recv = recv_displs[num_procs - 1] + recv_counts[num_procs - 1];

    particle_t* parts_sent = (particle_t*)calloc(num_parts_sent, sizeof(particle_t));
    particle_t* parts_recv = (particle_t*)calloc(num_parts_recv, sizeof(particle_t));

    for (auto it = moved_particles.begin(); it != moved_particles.end(); it++) {
        int block_id = it->first;
        int num_parts = it->second.size();
        for (int i = 0; i < num_parts; i++) {
            parts_sent[send_displs[block_id] + i] = it->second[i];
        }
    }

    MPI_Alltoallv(
        parts_sent, 
        send_counts, 
        send_displs, 
        PARTICLE, 
        parts_recv, 
        recv_counts, 
        recv_displs, 
        PARTICLE, 
        MPI_COMM_WORLD
    );

    // move the received particles to the correct grid
    for (int i = 0; i < num_parts_recv; i++) {
        int gx, gy;
        get_part_grid_index(parts_recv[i], gx, gy);
        int grid_id = get_part_grid_id(gx, gy);
        for (int k = 0; k < MAX_P; k++) {
            if (grids[grid_id].members[k].id == 0) {
                grids[grid_id].members[k] = parts_recv[i];
                grids[grid_id].num_p++;
                break;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    free(send_counts);
    free(recv_counts);
    free(send_displs);
    free(recv_displs);
    free(parts_sent);
    free(parts_recv);
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
    
    // we will first send the number of particles to each processor
    // then we will send the particles to each processor
    // finally, we will gather the particles from each processor
}