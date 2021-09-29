#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/22.
"""
import datetime
import typing

from src.affiliation import Affiliation
from src.paycheck import Paycheck
from src.service_charge import ServiceCharge


class UnionAffiliation(Affiliation):
    CHARGE_DAY_OF_WEEK: int = 5

    def __init__(self, member_id: int, dues: float):
        self.member_id = member_id
        self.dues = dues
        self.service_charges = []

    def add_service_charge(self, service_charge: ServiceCharge):
        self.service_charges.append(service_charge)

    def get_service_charge(
        self, date: datetime.date
    ) -> typing.Union[ServiceCharge, None]:
        for service_charge in self.service_charges:
            if service_charge.get_date() == date:
                return service_charge
        return None

    def calculate_deductions(self, paycheck: Paycheck):
        fridays = self.number_of_friday_in_pay_period(
            paycheck.pay_period_start_date, paycheck.pay_date
        )
        total_dues = self.dues * fridays
        for service_charge in self.service_charges:
            if (
                paycheck.pay_period_start_date
                <= service_charge.get_date()
                <= paycheck.pay_date
            ):
                total_dues += service_charge.get_amount()
        return total_dues

    def number_of_friday_in_pay_period(self, start: datetime.date, end: datetime.date):
        friday = 0

        for i in range((end - start).days + 1):
            day = start + datetime.timedelta(days=i)
            if day.weekday() == self.CHARGE_DAY_OF_WEEK - 1:
                friday += 1

        return friday
