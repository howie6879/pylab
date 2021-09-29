#include <stdio.h>

#define NUMBER 5

int main(void)
{
    int i, tensu[NUMBER], sum = 0;

    puts("请输入学生的分数");
    for (i = 0; i < NUMBER; i++)
    {
        printf("%2d号：", i + 1);
        scanf("%d", &tensu[i]);
        sum += tensu[i];
    }

    printf("总分：%5d\n", sum);
    printf("平均分：%5.1f\n", (double)sum / NUMBER);
}