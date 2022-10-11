#include <immintrin.h>
#include <stdio.h>

__m256d loadFourD(double* rsi) {
	__m256d fourD = _mm256_loadu_pd(rsi);
	return fourD;	
}

__m256d addFourD(__m256d x, __m256d y) {
	 return _mm256_add_pd(x,y);
}

__m256d mulFourD(__m256d x, __m256d y) {
	return _mm256_mul_pd(x,y);
}

void storeFourD(__m256d sourc, double* dest) {
	_mm256_store_pd(dest, sourc);
}

/*
int main() {
	double x[4] = {10,20,30,40};
	double y[4] = {30, 30, 30, 30};
	double out[4] = {1,2,3,4};

	__m256d awns = addFourD(loadFourD(x), loadFourD(y));
	storeFourD(awns, out);
	printf("%f, %f, %f, %f\n", out[0], out[1], out[2], out[3]);

}
*/
