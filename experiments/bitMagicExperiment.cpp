#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double EBH_ltd(long number) {
	return *(double*) &number;
}

long EBH_dtl(double number) {
	return *(long*) &number;
}

char* toBinary(int n, int len)
{
    char* binary = (char*)malloc(sizeof(char) * len);
    int k = 0;
    for (unsigned i = (1 << len - 1); i > 0; i = i / 2) {
        binary[k++] = (n & i) ? '1' : '0';
    }
    binary[k] = '\0';
    return binary;
}
/*
int main() {
	double contact = 100 - 1110;
	double nocontact = 1021 - 100;

	double c1 = 666; // take while no contact
	double c2 = 222; // take while contact

	long NCmask = (EBH_dtl(nocontact)>>63);
	long Cmask = (EBH_dtl(contact)>>63);

	char* ncB = toBinary(NCmask, 64);
	printf("NCmask %s\n", ncB);
	char* cB = toBinary(Cmask, 64);
	printf("Cmask %s\n", cB);
	printf("%f\n", EBH_ltd(~NCmask&EBH_dtl(c1) | EBH_dtl(c2)&NCmask));
	printf("%f\n", EBH_ltd(~Cmask&EBH_dtl(c1) | EBH_dtl(c2)&Cmask));

}
*/

