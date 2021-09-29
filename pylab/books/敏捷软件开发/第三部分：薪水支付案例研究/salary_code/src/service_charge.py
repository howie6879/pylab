#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/22.
"""
import datetime


class ServiceCharge:
    def __init__(self, date: datetime.date, amount: float):
        self._date = date
        self._amount = amount

    def get_date(self):
        return self._date

    def get_amount(self):
        return self._amount
