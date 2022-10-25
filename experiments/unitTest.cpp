#include "../SIMDlib.cpp"
#include "bitMagicExperiment.cpp"

int main() {
	double aS[] = {100,100,100,100};
	double bS[] = {101,99,100,99};
	double cS[] = {1,1,1,1};
	double result[] = {2,2,2,2};
	double resultM[] = {2,2,2,2};
	__m256d A = loadFourD(aS);
	__m256d B = loadFourD(bS);
	__m256d C = loadFourD(cS);

	storeFourD(compareFourD(A,B), result);

	printf("%s\n",toBinary(EBH_dtl(result[0]), 64));
	printf("%s\n",toBinary(EBH_dtl(result[1]), 64));
	printf("%s\n",toBinary(EBH_dtl(result[2]), 64));
	printf("%s\n",toBinary(EBH_dtl(result[3]), 64));

	__m256d Ax = _mm256_setr_pd(1,1,1,1);
	__m256d Ay = _mm256_setr_pd(2,1,1,1);
	__m256d Az = _mm256_setr_pd(3,1,1,1);
	__m256d Bx = _mm256_setr_pd(4,1,1,1);
	__m256d By = _mm256_setr_pd(5,1,1,1);
	__m256d Bz = _mm256_setr_pd(6,1,1,1);

	double dotResult[4] = {0,0,0,0};
	storeFourD(dotFourD(Ax, Ay, Az, Bx, By,Bz), dotResult);
	printf("%f, %f, %f, %f\n", dotResult[0], dotResult[1], dotResult[2], dotResult[3]);

	// mask load test
	//__mmask8 msk = _mm256_cmp_pd_mask(A,B,0);
	
	__m256d old = loadFourD(resultM);

	__m256d mask = _mm256_cmp_pd(A, B, 1);
	__m256d maskNeg = _mm256_cmp_pd(A,B, 5);
	__m256d maskedC =  _mm256_and_pd(C, mask);
	__m256d maskedOld = _mm256_and_pd(old, maskNeg);

	maskedC = addFourD(maskedC,maskedOld);

	storeFourD(maskedC, resultM);
	printf("%f, %f, %f, %f\n",resultM[0], resultM[1], resultM[2], resultM[3]);	

	return 0;
}
