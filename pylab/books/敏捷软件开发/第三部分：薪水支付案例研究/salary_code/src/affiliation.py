#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/22.
"""

from src.paycheck import Paycheck


class Affiliation:
    def calculate_deductions(self, paycheck: Paycheck):
        raise NotImplementedError
