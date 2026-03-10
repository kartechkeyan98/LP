#include<cblas.h>
#include<cstdio>

int main(void){
    double x[]= {1, 2, 3};
    double y[]= {4, 5, 6};
    double res= cblas_ddot(3, x, 1, y, 1);
    printf("Dot Product: %f\n", res);
    return 0;
}