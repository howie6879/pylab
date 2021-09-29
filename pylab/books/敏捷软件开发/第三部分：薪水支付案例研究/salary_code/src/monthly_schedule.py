#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/19.
"""
import calendar
import datetime

from src.payment_schedule import PaymentSchedule


class MonthlySchedule(PaymentSchedule):
    def is_pay_day(self, date: datetime.date):
        return calendar.monthrange(date.year, date.month)[1] == date.day

    def get_pay_period_start_date(self, pay_date: datetime.date):
        return pay_date.replace(day=1)


if __name__ == "__main__":
    date = datetime.date(2019, 4, 22)

    print(
        calendar.monthrange(
            datetime.date(2019, 4, 22).year, datetime.date(2019, 4, 22).month
        )[1]
    )

    date = date.replace(day=1)
    print(date)
