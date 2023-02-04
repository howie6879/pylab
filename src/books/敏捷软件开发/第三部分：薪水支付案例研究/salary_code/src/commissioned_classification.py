#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/19.
"""
from src.paycheck import Paycheck
from src.payment_classification import PaymentClassification


class CommissionedClassification(PaymentClassification):
    def __init__(self, salary, commission_rate):
        self.salary = salary
        self.commission_rate = commission_rate
        self.sales_receipts = []

    def add_sales_receipts(self, sales_receipt):
        self.sales_receipts.append(sales_receipt)

    def get_sales_receipt(self, day):
        pass

    def calculate_pay(self, paycheck: Paycheck):
        return 0
