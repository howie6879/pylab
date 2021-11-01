#include <stdio.h>

int main(void)
{
    int n1, n2;
    puts("请输入两个整数");
    printf("整数1：");
    scanf("%d", &n1);
    printf("整数2：");
    scanf("%d", &n2);
    printf("他们的和是：%d \n", n1 + n2);
}