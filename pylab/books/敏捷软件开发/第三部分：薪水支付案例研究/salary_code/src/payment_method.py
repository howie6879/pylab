#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/19.
"""

from src.paycheck import Paycheck


class PaymentMethod:
    def pay(self, paycheck: Paycheck):
        raise NotImplementedError
