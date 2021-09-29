#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/19.
"""
import datetime

from src.payment_schedule import PaymentSchedule


class BlweeklySchedule(PaymentSchedule):
    def __init__(self, value):
        self.value = value

    def is_pay_day(self, date: datetime.date):
        return date.weekday() == self.value - 1

    def get_pay_period_start_date(self, pay_date: datetime.date):
        return pay_date + datetime.timedelta(days=-11)
