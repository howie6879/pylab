#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/19.
"""

import datetime


class Paycheck:
    _gross_pay: float = None
    _deductions: float = None
    _net_pay: float = None
    _disposition: str = ""

    def __init__(self, pay_period_start_date: datetime.date, pay_date: datetime.date):
        self.pay_period_start_date = pay_period_start_date
        self.pay_date = pay_date

    @property
    def gross_pay(self):
        return self._gross_pay

    @gross_pay.setter
    def gross_pay(self, value):
        self._gross_pay = value

    @property
    def deductions(self):
        return self._deductions

    @deductions.setter
    def deductions(self, value):
        self._deductions = value

    @property
    def net_pay(self):
        return self._net_pay

    @net_pay.setter
    def net_pay(self, value):
        self._net_pay = value

    @property
    def disposition(self):
        return self._disposition

    @disposition.setter
    def disposition(self, value):
        self._disposition = value
