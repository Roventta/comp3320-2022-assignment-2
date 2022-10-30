#include "./cloth_code.h"
#include "./SoAVectorLib.cpp"
#include "./experiments/bitMagicExperiment.cpp"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void initMatrix(int n, double UNUSED(mass), double UNUSED(fcon),
                int UNUSED(delta), double UNUSED(grav), double sep,
                double rball, double offset, double UNUSED(dt), double **x,
                double **y, double **z, double **cpx, double **cpy,
                double **cpz, double **fx, double **fy, double **fz,
                double **vx, double **vy, double **vz, double **oldfx,
                double **oldfy, double **oldfz) {
  int i, nx, ny;

  // Free any existing
  free(*x);
  free(*y);
  free(*z);
  free(*cpx);
  free(*cpy);
  free(*cpz);

  // allocate arrays to hold locations of nodes
  *x = (double *)malloc(n * n * sizeof(double));
  *y = (double *)malloc(n * n * sizeof(double));
  *z = (double *)malloc(n * n * sizeof(double));
  // This is for opengl stuff
  *cpx = (double *)malloc(n * n * sizeof(double));
  *cpy = (double *)malloc(n * n * sizeof(double));
  *cpz = (double *)malloc(n * n * sizeof(double));

  // initialize coordinates of cloth
  for (nx = 0; nx < n; nx++) {
    for (ny = 0; ny < n; ny++) {
      (*x)[n * nx + ny] = nx * sep - (n - 1) * sep * 0.5 + offset;
      (*z)[n * nx + ny] = rball + 1;
      (*y)[n * nx + ny] = ny * sep - (n - 1) * sep * 0.5 + offset;
      (*cpx)[n * nx + ny] = 0;
      (*cpz)[n * nx + ny] = 1;
      (*cpy)[n * nx + ny] = 0;
    }
  }

  // Throw away existing arrays
  free(*fx);
  free(*fy);
  free(*fz);
  free(*vx);
  free(*vy);
  free(*vz);
  free(*oldfx);
  free(*oldfy);
  free(*oldfz);
  // Alloc new
  *fx = (double *)malloc(n * n * sizeof(double));
  *fy = (double *)malloc(n * n * sizeof(double));
  *fz = (double *)malloc(n * n * sizeof(double));
  *vx = (double *)malloc(n * n * sizeof(double));
  *vy = (double *)malloc(n * n * sizeof(double));
  *vz = (double *)malloc(n * n * sizeof(double));
  *oldfx = (double *)malloc(n * n * sizeof(double));
  *oldfy = (double *)malloc(n * n * sizeof(double));
  *oldfz = (double *)malloc(n * n * sizeof(double));
  for (i = 0; i < n * n; i++) {
    (*vx)[i] = 0.0;
    (*vy)[i] = 0.0;
    (*vz)[i] = 0.0;
    (*fx)[i] = 0.0;
    (*fy)[i] = 0.0;
    (*fz)[i] = 0.0;
  }
}

void loopcode(int n, double mass, double fcon, int delta, double grav,
              double sep, double rball, double xball, double yball,
              double zball, double dt, double *x, double *y, double *z,
              double *fx, double *fy, double *fz, double *vx, double *vy,
              double *vz, double *oldfx, double *oldfy, double *oldfz,
              double *pe, double *ke, double *te) {
  int i, j;
  double xdiff, ydiff, zdiff, vmag, damp;


  // update position as per MD simulation
  for (j = 0; j < n; j++) {
    for (i = 0; i < n; i++) {
      x[j * n + i] += dt * (vx[j * n + i] + dt * fx[j * n + i] * 0.5 / mass);
      oldfx[j * n + i] = fx[j * n + i];
      
      y[j * n + i] += dt * (vy[j * n + i] + dt * fy[j * n + i] * 0.5 / mass);
      oldfy[j * n + i] = fy[j * n + i];
     
      z[j * n + i] += dt * (vz[j * n + i] + dt * fz[j * n + i] * 0.5 / mass);
      oldfz[j * n + i] = fz[j * n + i];

        xdiff = x[j * n + i] - xball;
	ydiff = y[j * n + i] - yball;
	zdiff = z[j * n + i] - zball;
	vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
	
	long mask = (EBH_dtl(vmag-rball)>>63);
	x[j*n+i] = EBH_ltd((EBH_dtl(x[j*n+i]) & ~mask) | (EBH_dtl(xball+xdiff*rball/vmag) & mask));
	y[j*n+i] = EBH_ltd((EBH_dtl(y[j*n+i]) & ~mask) | (EBH_dtl(yball+ydiff*rball/vmag) & mask));
	z[j*n+i] = EBH_ltd((EBH_dtl(z[j*n+i]) & ~mask) | (EBH_dtl(zball+zdiff*rball/vmag) & mask));

	//get the normal of the ball contacting surface
	struct singleVector surfaceNormal = vectorSub(xball, yball, zball, x[j*n+i], y[j*n+i], z[j*n+i]);
	struct singleVector eliminated_velocity = vectorProj(vx[j*n+i], vy[j*n+i], vz[j*n+i], surfaceNormal.x, surfaceNormal.y, surfaceNormal.z);
	struct singleVector newVelocity = vectorSub(vx[j*n+i], vy[j*n+i], vz[j*n+i],eliminated_velocity.x, eliminated_velocity.y, eliminated_velocity.z);
	//friction
	newVelocity = vectorScale(newVelocity.x, newVelocity.y, newVelocity.z, 0.1);
	
	vx[j*n+i] = EBH_ltd((EBH_dtl(vx[j*n+i]) & ~mask) | (EBH_dtl(newVelocity.x) & mask));
	vy[j*n+i] = EBH_ltd((EBH_dtl(vy[j*n+i]) & ~mask) | (EBH_dtl(newVelocity.y) & mask));
	vz[j*n+i] = EBH_ltd((EBH_dtl(vz[j*n+i]) & ~mask) | (EBH_dtl(newVelocity.z) & mask));   
    }
  }

  *pe = eval_pef(n, delta, mass, grav, sep, fcon, x, y, z, fx, fy, fz);

  // Add a damping factor to eventually set velocity to zero
  damp = 0.995;
  *ke = 0.0;
  for (j = 0; j < n; j++) {
    for (i = 0; i < n; i++) {
      vx[j * n + i] = (vx[j * n + i] +
                       dt * (fx[j * n + i] + oldfx[j * n + i]) * 0.5 / mass) *
                      damp;
      vy[j * n + i] = (vy[j * n + i] +
                       dt * (fy[j * n + i] + oldfy[j * n + i]) * 0.5 / mass) *
                      damp;
      vz[j * n + i] = (vz[j * n + i] +
                       dt * (fz[j * n + i] + oldfz[j * n + i]) * 0.5 / mass) *
                      damp;
      *ke += vx[j * n + i] * vx[j * n + i] + vy[j * n + i] * vy[j * n + i] +
             vz[j * n + i] * vz[j * n + i];
    }
  }
  *ke = *ke / 2.0;
  *te = *pe + *ke;
}

double eval_pef(int n, int delta, double mass, double grav, double sep,
                double fcon, double *x, double *y, double *z, double *fx,
                double *fy, double *fz) {
  double pe, xdiff, ydiff, zdiff, vmag;
  int nx, ny, dx, dy;

  int adjacents[1+4*delta*delta+4*delta] = {0};
  double rlen[1+4*delta*delta+4*delta] = {0};

  pe = 0.0;
  // loop over particles
  for (nx = 0; nx < n; nx++) {
    for (ny = 0; ny < n; ny++) {
      fx[nx * n + ny] = 0.0;
      fy[nx * n + ny] = 0.0;
      fz[nx * n + ny] = -mass * grav;
      // loop over displacements
	int adjacentIndex = 0;
/*		for(dx = MAX(nx-delta, 0); dx< MIN(nx+delta+1, n); dx++) {
		for(dy = MAX(ny-delta, 0); dy<MIN(ny+delta+1, n); dy++){
			if(dx!=nx||dy!=ny){adjacents[adjacentIndex] = dx*n+dy;
			rlen[adjacentIndex] = sqrt((double)((nx-dx)*(nx-dx)+(ny-dy)*(ny-dy)))*sep;
			adjacentIndex++;}
		}
	} 
*/
	
		for(dx = MAX(nx-delta, 0); dx< nx; dx++) {
		for(dy = MAX(ny-delta, 0); dy<MIN(ny+delta+1, n); dy++){
			adjacents[adjacentIndex] = dx*n+dy;
			rlen[adjacentIndex] = sqrt((double)((nx-dx)*(nx-dx)+(ny-dy)*(ny-dy)))*sep;
			adjacentIndex++;
		}
	} 

	for(dx = nx+1; dx< MIN(nx+delta+1, n); dx++) {
		for(dy = MAX(ny-delta, 0); dy<MIN(ny+delta+1, n); dy++){
			adjacents[adjacentIndex] = dx*n+dy;
			rlen[adjacentIndex] = sqrt((double)((nx-dx)*(nx-dx)+(ny-dy)*(ny-dy)))*sep;
			adjacentIndex++;
		}
	}
	
	dx = nx;

	for(dy = MAX(ny-delta,0); dy<ny; dy++){
			adjacents[adjacentIndex] = dx*n+dy;
			rlen[adjacentIndex] = sqrt((double)((nx-dx)*(nx-dx)+(ny-dy)*(ny-dy)))*sep;
			adjacentIndex++;
	}
	for(dy = ny+1; dy<MIN(ny+delta+1, n); dy++){
			adjacents[adjacentIndex] = dx*n+dy;
			rlen[adjacentIndex] = sqrt((double)((nx-dx)*(nx-dx)+(ny-dy)*(ny-dy)))*sep;
			adjacentIndex++;
	}
	

	
	
	for (int i = 0; i<adjacentIndex; i++) {
	    // compute reference distance
            // compute actual distance
            xdiff = x[adjacents[i]] - x[nx * n + ny];
            ydiff = y[adjacents[i]] - y[nx * n + ny];
            zdiff = z[adjacents[i]] - z[nx * n + ny];
            vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
            // potential energy and force
            pe += fcon * (vmag - rlen[i]) * (vmag - rlen[i]);
            fx[nx * n + ny] += fcon * xdiff * (vmag - rlen[i]) / vmag;
            fy[nx * n + ny] += fcon * ydiff * (vmag - rlen[i]) / vmag;
            fz[nx * n + ny] += fcon * zdiff * (vmag - rlen[i]) / vmag;
 
	}
    }
  }
  return 0.5 * pe;
}
