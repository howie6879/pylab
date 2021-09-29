#include <stdio.h>

int main(void)
{
    int a, b;
    double c;

    printf("整数A:");
    scanf("%d", &a);
    printf("整数B:");
    scanf("%d", &b);
    c = (a / (b * 1.0)) * 100;
    printf("A的值是B的 %d%% \n", (int)c);
}