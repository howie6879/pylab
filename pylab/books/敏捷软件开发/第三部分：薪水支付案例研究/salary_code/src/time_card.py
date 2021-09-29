#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/20.
"""

import datetime


class TimeCard:
    def __init__(self, date: datetime.date, hours: float):
        self._date = date
        self._hours = hours

    def get_date(self):
        return self._date

    def get_hours(self):
        return self._hours


if __name__ == "__main__":
    print(type(datetime.date.today()))
    print(datetime.date.today())

    print(datetime.date(2017, 3, 22))
    print(datetime.date(2019, 4, 22).month)

    print(datetime.date.today() - datetime.timedelta(days=1))
