#include "common.h"
#include <mpi.h>

# define MAX_P 5

int nblock_x, nblock_y;
int ngrid;
int ngrid_per_block_x, ngrid_per_block_y;

int block_x, block_y;

// the grid that each processor has
grid_class* grids;

particle_t* moved_particles;

// Put any static global variables here that you will use throughout the simulation.
typedef struct grid_class {
    int num_p;
    particle_t members[MAX_P]; 
} grid_class;

int flatten_index(int x, int y, int y_size) {
    return x * y_size + y;
}

int* unflatten_index(int index, int y_size) {
    int* result = (int*)malloc(2 * sizeof(int));
    result[0] = index / y_size;
    result[1] = index % y_size;
    return result;
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

void apply_force_all(int gx, int gy, particle_t& part) {
    for (int px = gx; px < gx + 3; px++) {
        for (int py = gy; py < gy + 3; py++) {
            // If the neighbor is outside the grid, let's abondon it.
            if (px >= ngrid_per_block_x || py >= ngrid_per_block_y) continue;

            // Assign the grid for this neighbor.
            grid_class* grid = &grids[flatten_index(px, py, ngrid_per_block_y)];

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

/*
This function will update the ghost grids on the boundary of the 
block so that we can compute the force without communicating with other blocks.
*/
void update_ghost_grid() {
}

/*
This function will move the particles from one block to another

MPI_alltoallv is used to send the particles to the correct block
*/
void redistribute_parts() {}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here
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
    ngrid_per_block_x = (int)ceil(ngrid / nblock_x);
    ngrid_per_block_y = (int)ceil(ngrid / nblock_y);

    // the position of the block in the block grid
    int* block_pos = unflatten_index(rank, nblock_y);
    block_x = block_pos[0];
    block_y = block_pos[1];

    grids = (grid_class*)calloc((ngrid_per_block_x + 2) * (ngrid_per_block_y + 2), sizeof(grid_class));
    moved_particles = (particle_t*)calloc(nblock_x * nblock_y * MAX_P * 10, sizeof(particle_t));

    // TODO : parallelize this assigning process
    // go through all the particles, if they are in this block, assign them to the grids
    // the ghost grids are left empty
    for (int i = 0; i < num_parts; i++) {
        int grid_id_x = (int)floor(parts[i].x / cutoff) - block_id_x * ngrid_per_block_x;
        int grid_id_y = (int)floor(parts[i].y / cutoff) - block_id_y * ngrid_per_block_y;

        if (grid_id_x >= 0 && grid_id_x < ngrid_per_block_x && grid_id_y >= 0 && grid_id_y < ngrid_per_block_y) {
            int grid_id = flatten_index(grid_id_x + 1, grid_id_y + 1, ngrid_per_block_y);
            grids[grid_id].members[grids[grid_id].num_p] = parts[i];
            grids[grid_id].num_p++;
        }
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Exchange the ghost atoms with the neighboring blocks
    update_ghost_grids();

    // Compute the forces on each particle
    for(int i = 0; i < ngrid * ngrid; i++) {
        if (0 != grids[i].num_p) {
            for (int j = 0; j < MAX_P; j++) {
                if (grids[i].members[j] == NULL) continue;
                particle_t part = grids[i].members[j];
                part.ax = part.ay = 0;
                int gx = (int)(part.x / cutoff) - 1;
                int gy = (int)(part.y / RANGE) - 1;
                if(gx < 0) {
                    gx = 0;
                }
                if(gy < 0) {
                    gy = 0;
                }
                apply_force_all(gx, gy, part);
            }
        }
    }

    for (int i = 0; i < ngrids_per_block_x * ngrids_per_block_y; i++) {
        if (0 != grids[i].num_p) {
            for (int j = 0; j < MAX_P; j++) {
                if (grids[i].members[j].id == 0) continue;
                move(grids[i].members[j], size);
            }
        }
    }

    redistribute_parts()
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
}