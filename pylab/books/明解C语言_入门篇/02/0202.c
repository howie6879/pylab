#include <stdio.h>

int main(void)
{
    int no;

    printf("请输入一个整数：");
    scanf("%d", &no);

    printf("整数的最后一位是：%d \n", no % 10);
}