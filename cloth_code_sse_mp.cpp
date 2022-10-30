#include "./cloth_code.h"
#include "./SoAVectorLib.cpp"
#include "./SIMDlib.cpp"
#include "./experiments/bitMagicExperiment.cpp"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

__m256d xballF;
__m256d yballF;
__m256d zballF;
__m256d rballF;

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
  *x = (double *)aligned_alloc(32, n * n * sizeof(double));
  *y = (double *)aligned_alloc(32, n * n * sizeof(double));
  *z = (double *)aligned_alloc(32, n * n * sizeof(double));
  // This is for opengl stuff
  *cpx = (double *)aligned_alloc(32, n * n * sizeof(double));
  *cpy = (double *)aligned_alloc(32, n * n * sizeof(double));
  *cpz = (double *)aligned_alloc(32, n * n * sizeof(double));

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
  *fx = (double *)aligned_alloc(32, n * n * sizeof(double));
  *fy = (double *)aligned_alloc(32, n * n * sizeof(double));
  *fz = (double *)aligned_alloc(32, n * n * sizeof(double));
  *vx = (double *)aligned_alloc(32, n * n * sizeof(double));
  *vy = (double *)aligned_alloc(32, n * n * sizeof(double));
  *vz = (double *)aligned_alloc(32, n * n * sizeof(double));
  *oldfx = (double *)aligned_alloc(32, n * n * sizeof(double));
  *oldfy = (double *)aligned_alloc(32, n * n * sizeof(double));
  *oldfz = (double *)aligned_alloc(32, n * n * sizeof(double));
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
              double zball, double dt, double * __restrict__ x, double * __restrict__ y, double * __restrict__ z,
              double * __restrict__ fx, double * __restrict__ fy, double * __restrict__ fz, double * __restrict__ vx, double * __restrict__ vy,
              double * __restrict__ vz, double * __restrict__ oldfx, double * __restrict__ oldfy, double * __restrict__ oldfz,
              double * __restrict__ pe, double * __restrict__ ke, double * __restrict__ te) {
  //_assume_aligned(var, 32);

  int chunk_size = n/omp_get_num_threads();
  int i, j;
  double xdiff, ydiff, zdiff, vmag, damp;
  damp = 0.995;
  __m256d dampF = _mm256_setr_pd(damp, damp, damp, damp);
  __m256d dt_half_mass = _mm256_setr_pd(dt*0.5/mass,dt*0.5/mass,dt*0.5/mass,dt*0.5/mass);
  __m256d dtS_half_mass = _mm256_setr_pd(dt*dt*0.5/mass,dt*dt*0.5/mass,dt*dt*0.5/mass,dt*dt*0.5/mass);
  __m256d dtF = _mm256_setr_pd(dt, dt, dt, dt); 
  __m256d friction = _mm256_setr_pd(0.1,0.1,0.1,0.1); 

  xballF = _mm256_setr_pd(xball, xball, xball, xball);
  yballF = _mm256_setr_pd(yball, yball, yball, yball);
  zballF = _mm256_setr_pd(zball, zball, zball, zball);
  rballF = _mm256_setr_pd(rball, rball, rball, rball);
 	
	#pragma omp for schedule(static, chunk_size)
  for(j=0; j<n; j++) {
  	for(i=0; i<n-n%4; i=i+4){
	// load
	__m256d fxF = loadFourD(&fx[j*n+i]);
	__m256d fxF_clone = fxF;
	__m256d vxF = loadFourD(&vx[j*n+i]);
	__m256d xF = loadFourD(&x[j*n+i]);

	__m256d fyF = loadFourD(&fy[j*n+i]);
	__m256d fyF_clone = fyF;
	__m256d vyF = loadFourD(&vy[j*n+i]);
	__m256d yF = loadFourD(&y[j*n+i]);

	__m256d fzF = loadFourD(&fz[j*n+i]);
	__m256d fzF_clone = fzF;
	__m256d vzF = loadFourD(&vz[j*n+i]);
	__m256d zF = loadFourD(&z[j*n+i]);

	//alu
	__m256d temp_vxF = mulFourD(vxF, dtF);
	fxF = mulFourD(fxF, dtS_half_mass);
	xF = addFourD(xF, temp_vxF);
	xF = addFourD(xF, fxF);

	__m256d temp_vyF = mulFourD(vyF, dtF);
	fyF = mulFourD(fyF, dtS_half_mass);
	yF = addFourD(yF, temp_vyF);
	yF = addFourD(yF, fyF);

	__m256d temp_vzF = mulFourD(vzF, dtF);
	fzF = mulFourD(fzF, dtS_half_mass);
	zF = addFourD(zF, temp_vzF);
	zF = addFourD(zF, fzF);
	
		//alu
	__m256d xdiffF = subFourD(xF, xballF);
	__m256d ydiffF = subFourD(yF, yballF);
	__m256d zdiffF = subFourD(zF, zballF);
		
	__m256d vmagF = addFourD(mulFourD(xdiffF, xdiffF), mulFourD(ydiffF, ydiffF));
	vmagF = addFourD(vmagF, mulFourD(zdiffF,zdiffF));
	vmagF = _mm256_sqrt_pd(vmagF);

	//mask
	__m256d maskF = _mm256_cmp_pd(vmagF,rballF,1);
	__m256d neg_maskF = _mm256_cmp_pd(vmagF,rballF,5);
	
	//calc new position
	__m256d newxF = addFourD(xballF, _mm256_div_pd(mulFourD(xdiffF, rballF),vmagF));
	__m256d newyF = addFourD(yballF, _mm256_div_pd(mulFourD(ydiffF, rballF),vmagF));
	__m256d newzF = addFourD(zballF, _mm256_div_pd(mulFourD(zdiffF, rballF),vmagF));

	//calc new velocity
	//proj calc
	
	//surfaceNormal
	__m256d snx = subFourD(xballF, newxF); 
	__m256d sny = subFourD(yballF, newyF); 
	__m256d snz = subFourD(zballF, newzF);
	
	// the scaler for projection
	__m256d projScaleF = _mm256_div_pd(dotFourD(vxF, vyF, vzF,snx, sny, snz), dotFourD(snx, sny, snz, snx, sny, snz));
	__m256d subVxF = mulFourD(snx, projScaleF);
	__m256d subVyF = mulFourD(sny, projScaleF);
	__m256d subVzF = mulFourD(snz, projScaleF);

	__m256d newVxF = subFourD(vxF, subVxF);
	__m256d newVyF = subFourD(vyF, subVyF);
	__m256d newVzF = subFourD(vzF, subVzF);

	// friction
	newVxF = mulFourD(newVxF, friction);
	newVyF = mulFourD(newVyF, friction);
	newVzF = mulFourD(newVzF, friction);
		
	xF = _mm256_and_pd(xF, neg_maskF);
	yF = _mm256_and_pd(yF, neg_maskF);
	zF = _mm256_and_pd(zF, neg_maskF);

	vxF = _mm256_and_pd(vxF, neg_maskF);
	vyF = _mm256_and_pd(vyF, neg_maskF);
	vzF = _mm256_and_pd(vzF, neg_maskF);

	__m256d maskedxF = _mm256_and_pd(newxF, maskF); 
	__m256d maskedyF = _mm256_and_pd(newyF, maskF); 
	__m256d maskedzF = _mm256_and_pd(newzF, maskF); 
	
	__m256d maskedVxF = _mm256_and_pd(newVxF, maskF); 
	__m256d maskedVyF = _mm256_and_pd(newVyF, maskF); 
	__m256d maskedVzF = _mm256_and_pd(newVzF, maskF);

	maskedxF = addFourD(maskedxF, xF);
	maskedyF = addFourD(maskedyF, yF);
	maskedzF = addFourD(maskedzF, zF);

	maskedVxF = addFourD(maskedVxF, vxF);
	maskedVyF = addFourD(maskedVyF, vyF);
	maskedVzF = addFourD(maskedVzF, vzF);

	storeFourD(maskedxF,&x[j*n+i]);	
	storeFourD(maskedyF,&y[j*n+i]);	
	storeFourD(maskedzF,&z[j*n+i]);	
	storeFourD(maskedVxF,&vx[j*n+i]);	
	storeFourD(maskedVyF,&vy[j*n+i]);	
	storeFourD(maskedVzF,&vz[j*n+i]);	

	_mm256_storeu_pd(&oldfx[j*n+i], fxF_clone);
	_mm256_storeu_pd(&oldfy[j*n+i], fyF_clone);
	_mm256_storeu_pd(&oldfz[j*n+i], fzF_clone);

	}

	for (i = n-n%4; i < n; i++) {
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
  
  *ke = 0.0;
  static double KE;
  KE = 0.0;
 	#pragma omp for schedule(static, chunk_size) reduction(+:KE) 
	  for (j=0;j<n;j++) {
		for(i=0;i<n-n%4;i=i+4) {
  			double keArray[] = {0,0,0,0};
  			__m256d keF = _mm256_setr_pd(0,0,0,0);
			__m256d vxF = loadFourD(&vx[j*n+i]);
			__m256d fxF = loadFourD(&fx[j*n+i]);
			__m256d oldfxF = loadFourD(&oldfx[j*n+i]);
		
			__m256d vyF = loadFourD(&vy[j*n+i]);
			__m256d fyF = loadFourD(&fy[j*n+i]);
			__m256d oldfyF = loadFourD(&oldfy[j*n+i]);
		
			__m256d vzF = loadFourD(&vz[j*n+i]);
			__m256d fzF = loadFourD(&fz[j*n+i]);
			__m256d oldfzF = loadFourD(&oldfz[j*n+i]);

			//alu
			vxF = addFourD(vxF, mulFourD(addFourD(fxF, oldfxF), dt_half_mass));
		        vxF = mulFourD(vxF, dampF);
			
			vyF = addFourD(vyF, mulFourD(addFourD(fyF, oldfyF), dt_half_mass));
		        vyF = mulFourD(vyF, dampF);
			
			vzF = addFourD(vzF, mulFourD(addFourD(fzF, oldfzF), dt_half_mass));
		        vzF = mulFourD(vzF, dampF);
			
			keF = addFourD(keF, mulFourD(vxF, vxF));
			keF = addFourD(keF, mulFourD(vyF, vyF));
			keF = addFourD(keF, mulFourD(vzF, vzF));	

			storeFourD(vxF, &vx[j*n+i]);	
			storeFourD(vyF, &vy[j*n+i]);	
			storeFourD(vzF, &vz[j*n+i]);	
  
  			storeFourD(keF, keArray);
			
			KE += keArray[0]+keArray[1]+keArray[2]+keArray[3];
		}
		for(i=n-n%4;i<n;i++) {
			vx[j * n + i] = (vx[j * n + i] + dt * (fx[j * n + i] + oldfx[j * n + i]) * 0.5 / mass) * damp;
      			vy[j * n + i] = (vy[j * n + i] + dt * (fy[j * n + i] + oldfy[j * n + i]) * 0.5 / mass)*damp;
     			vz[j * n + i] = (vz[j * n + i] + dt * (fz[j * n + i] + oldfz[j * n + i]) * 0.5 / mass) * damp;
      			KE += vx[j * n + i] * vx[j * n + i] + vy[j * n + i] * vy[j * n + i] + vz[j * n + i] * vz[j * n + i];
		}
	}
  *ke = KE / 2.0;
  *te = *pe + *ke;
}

double eval_pef(int n, int delta, double mass, double grav, double sep,
                double fcon, double * __restrict__ x, double * __restrict__ y, double * __restrict__ z, double * __restrict__ fx,
                double * __restrict__ fy, double * __restrict__ fz) {
double  xdiff, ydiff, zdiff, vmag;
  int nx, ny, dx, dy;

  int chunk_size = n/omp_get_num_threads();
 
  static double pe;
  pe = 0;
  __m256d fconF = _mm256_setr_pd(fcon, fcon, fcon, fcon);
  // loop over particles

  #pragma omp for schedule(static, chunk_size) reduction(+:pe) 
  for (nx = 0; nx < n; nx++) {
  // initialize objects for one rows 
  int adjacents[1+4*delta*delta+4*delta] = {0}; 
  double rlen[1+4*delta*delta+4*delta] = {0};
  double fourPE[] = {0,0,0,0};
  __m256d peF = _mm256_setr_pd(0,0,0,0);
  __m256d placeHolder;
  __m256d pIncrement; 
  for (ny = 0; ny < n; ny++) {
      fx[nx * n + ny] = 0.0;
      fy[nx * n + ny] = 0.0;
      fz[nx * n + ny] = -mass * grav;
	//collect all the info of adjacent nodes.	
      int adjacentIndex = 0;
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
 

	double fourFx[] = {0,0,0,0};
  	double fourFy[] = {0,0,0,0};
 	double fourFz[] = {0,0,0,0};
	// sse part, process four nodes at once
	for(int i = 0; i<(adjacentIndex/4)*4; i = i+4) {
		__m256d xF = _mm256_setr_pd(x[nx*n+ny],x[nx*n+ny],x[nx*n+ny],x[nx*n+ny]);
		__m256d xAdjF = _mm256_setr_pd(x[adjacents[i]],x[adjacents[i+1]],x[adjacents[i+2]],x[adjacents[i+3]]);
		__m256d yF = _mm256_setr_pd(y[nx*n+ny],y[nx*n+ny],y[nx*n+ny],y[nx*n+ny]);
		__m256d yAdjF = _mm256_setr_pd(y[adjacents[i]],y[adjacents[i+1]],y[adjacents[i+2]],y[adjacents[i+3]]);
		__m256d zF = _mm256_setr_pd(z[nx*n+ny],z[nx*n+ny],z[nx*n+ny],z[nx*n+ny]);
		__m256d zAdjF = _mm256_setr_pd(z[adjacents[i]],z[adjacents[i+1]],z[adjacents[i+2]],z[adjacents[i+3]]);

		__m256d rlenF = _mm256_setr_pd(rlen[i], rlen[i+1], rlen[i+2], rlen[i+3]);

		__m256d xdiff = subFourD(xAdjF, xF);
		__m256d ydiff = subFourD(yAdjF, yF);
		__m256d zdiff = subFourD(zAdjF, zF);

		__m256d vmagF = addFourD(mulFourD(xdiff, xdiff), mulFourD(ydiff, ydiff));
		vmagF = addFourD(vmagF, mulFourD(zdiff,zdiff));
		vmagF = _mm256_sqrt_pd(vmagF);

		placeHolder = subFourD(vmagF, rlenF);
		pIncrement = mulFourD(fconF,mulFourD(placeHolder, placeHolder));
		peF = addFourD(peF, pIncrement);

		__m256d fxF = loadFourD(fourFx);
		__m256d fyF = loadFourD(fourFy);
		__m256d fzF = loadFourD(fourFz);

		fxF = addFourD(fxF,_mm256_div_pd(mulFourD(fconF,mulFourD(placeHolder,xdiff)), vmagF));
		fyF = addFourD(fyF,_mm256_div_pd(mulFourD(fconF,mulFourD(placeHolder,ydiff)), vmagF));
		fzF = addFourD(fzF,_mm256_div_pd(mulFourD(fconF,mulFourD(placeHolder,zdiff)), vmagF));

		storeFourD(fxF, fourFx);
		storeFourD(fyF, fourFy);
		storeFourD(fzF, fourFz);

	}
	
	fx[nx*n+ny] += fourFx[0]+fourFx[1]+fourFx[2]+fourFx[3]; 
	fy[nx*n+ny] += fourFy[0]+fourFy[1]+fourFy[2]+fourFy[3]; 
	fz[nx*n+ny] += fourFz[0]+fourFz[1]+fourFz[2]+fourFz[3]; 

	for (int i = 4*(adjacentIndex/4); i<adjacentIndex; i++) {
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
  
  storeFourD(peF, fourPE);
  pe += fourPE[0]+fourPE[1]+fourPE[2]+fourPE[3];
  }

  return 0.5 * pe;
}
