#include <math.h>
#include <stdio.h>

struct singleVector {
	double x;
	double y;
	double z;
} SV;

void SVPrinter(struct singleVector in) {
	printf("x is %f\n", in.x);
	printf("y is %f\n", in.y);
	printf("z is %f\n", in.z);
}

singleVector vectorAdd(double ax, double ay, double az, double bx, double by, double bz){
	struct singleVector out = {ax+bx, ay+by, az+bz};
	return out;
}

singleVector vectorSub(double ax, double ay, double az, double bx, double by, double bz){
	struct singleVector out = {ax-bx, ay-by, az-bz};
	return out;
}

double vectorDot(double ax, double ay, double az, double bx, double by, double bz) {
	double out = ax*bx + ay*by + az*bz;
	return out;
}

singleVector vectorScale(double ax, double ay, double az, double scale) {
	struct singleVector out = {ax*scale, ay*scale, az*scale};
	return out;
}

singleVector vectorProj(double ax, double ay, double az, double bx, double by, double bz) {
	double scale = vectorDot(ax, ay, az, bx, by,bz)/vectorDot(bx,by,bz,bx,by,bz);
	return vectorScale(bx,by,bz, scale);	
}

void loadSingle(double* X, double* Y, double* Z, int i,struct singleVector v){
		X[i] = v.x;
		Y[i] = v.y;
		Z[i] = v.z;
}

/*
int main() {
	SVPrinter(vectorProj(1,2,3,9,8,7));
}
*/

