#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/19.
"""
import datetime
import typing

from src.paycheck import Paycheck
from src.payment_classification import PaymentClassification
from src.time_card import TimeCard


class HourlyClassification(PaymentClassification):
    FRIDAY = 5
    WORK_HOUR_OF_DAY = 8.0
    MORE_PAY_RATE = 1.5

    def __init__(self, hour_salary):
        self.hour_salary = hour_salary
        self.time_cards = []

    def add_time_card(self, time_card: TimeCard):
        self.time_cards.append(time_card)

    def get_time_card(self, date: datetime.date) -> typing.Union[TimeCard, None]:
        for time_card in self.time_cards:
            if time_card.get_date() == date:
                return time_card
        return None

    def calculate_pay(self, paycheck: Paycheck):
        pay = 0.0
        for time_card in self.time_cards:
            if self.is_in_pay_periods(time_card, paycheck.pay_date):
                pay += self.calculate_pay_for_time_card(time_card)
        return pay

    def calculate_pay_for_time_card(self, time_card: TimeCard):
        hours = time_card.get_hours()
        over_time = max(0.0, hours - self.WORK_HOUR_OF_DAY)
        straight_time = hours - over_time
        return (
            straight_time * self.hour_salary
            + over_time * self.hour_salary * self.MORE_PAY_RATE
        )

    def is_in_pay_periods(self, time_card: TimeCard, pay_day: datetime.date):
        start_day = pay_day + datetime.timedelta(days=-5)
        return start_day <= time_card.get_date() <= pay_day


if __name__ == "__main__":
    a = datetime.date(2019, 4, 1)
    b = datetime.date(2019, 4, 1)
    c = datetime.date(2019, 4, 22)
    print(c > b >= a)
