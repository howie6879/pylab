#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/20.
"""
import datetime

from src.hourly_classification import HourlyClassification
from src.payroll_database import PayrollDatabase
from src.time_card import TimeCard
from src.transaction import Transaction


class TimeCardTransaction(Transaction):
    def __init__(self, date: datetime.date, hours: float, emp_id: int):
        self._date = date
        self._hours = hours
        self._emp_id = emp_id

    def execute(self):
        e = PayrollDatabase.get_employee(self._emp_id)

        if e is not None:
            hc = e.classification
            if isinstance(hc, HourlyClassification):
                hc.add_time_card(TimeCard(self._date, self._hours))
            else:
                raise ValueError("Tried to add time_card to non-hourly employee")
        else:
            raise ValueError("No such employee.")
