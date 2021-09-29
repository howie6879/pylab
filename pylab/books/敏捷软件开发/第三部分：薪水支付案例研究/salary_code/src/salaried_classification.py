#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/19.
"""
from src.paycheck import Paycheck
from src.payment_classification import PaymentClassification


class SalariedClassification(PaymentClassification):
    def __init__(self, salary):
        self.salary = salary

    def calculate_pay(self, paycheck: Paycheck):
        return self.salary
