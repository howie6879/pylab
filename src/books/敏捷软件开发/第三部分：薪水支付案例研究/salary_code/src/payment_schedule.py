#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/19.
"""

import datetime


class PaymentSchedule:
    def is_pay_day(self, date: datetime.date):
        raise NotImplementedError

    def get_pay_period_start_date(self, pay_date: datetime.date):
        raise NotImplementedError
