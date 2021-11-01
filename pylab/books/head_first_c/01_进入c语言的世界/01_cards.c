#include <stdio.h>
#include <stdlib.h>

int main()
{
    char card_name[3];
    puts("输入牌名: ");
    scanf("%2s", card_name);
    int val = 0;
    if (card_name[0] == 'k')
    {
        val = 10;
    }
    else if (card_name[0] == 'Q')
    {
        val = 10;
    }
    else if (card_name[0] == 'J')
    {
        val = 10;
    }
    else if (card_name[0] == 'A')
    {
        val = 11;
    }
    else
    {
        val = atoi(card_name);
    }
    if (val >= 3 && val <= 6)
    {
        puts("计数增加");
    }
    else if (val == 10 || card_name[0] == 'J' || card_name[0] == 'Q' || card_name[0] == 'K'){
        puts("计数减少");
    }
    printf("这张牌的点数是：%i\n", val);
    return 0;
}
