# #!/usr/bin/env python
# """
#  Created by howie.hu at 2019/4/19.
# """
#
# import datetime
#
# from src.add_salaried_employee import AddSalariedEmployee
# from src.add_commissioned_employee import AddCommissionedEmployee
# from src.add_hourly_empoylee import AddHourlyEmployee
# from src.change_member_transaction import ChangeMemberTransaction
# from src.change_name_transaction import ChangeNameTransaction
# from src.change_hourly_transaction import ChangeHourlyTransaction
# from src.delete_employee_transaction import DeleteEmployeeTransaction
# from src.hold_method import HoldMethod
# from src.hourly_classification import HourlyClassification
# from src.monthly_schedule import MonthlySchedule
# from src.paycheck import Paycheck
# from src.payday_transaction import PaydayTransaction
# from src.payroll_database import PayrollDatabase
# from src.salaried_classification import SalariedClassification
# from src.serveice_charge_transaction import ServiceChargeTransaction
# from src.time_card_transaction import TimeCardTransaction
# from src.union_affiliation import UnionAffiliation
# from src.weekly_schedule import WeeklySchedule
#
#
# def hourly_union_member_service_charge():
#     emp_id = 1
#     member_id = 7734
#
#     t = AddHourlyEmployee(emp_id, "Bill", "Home", 15.24)
#     t.execute()
#
#     cmt = ChangeMemberTransaction(emp_id, member_id, 9.42)
#     cmt.execute()
#
#     pay_date = datetime.date(2001, 11, 9)
#
#     sct = ServiceChargeTransaction(member_id, pay_date, 19.42)
#     sct.execute()
#
#     pt = PaydayTransaction(pay_date)
#     pt.execute()
#
#     pc: Paycheck = pt.get_paycheck(emp_id)
#
#     assert pc is not None
#     assert pay_date == pc.pay_date
#     assert 8 * 15.24 == pc.gross_pay
#     assert 9.42 * 19.42 == pc.deductions
#     assert ((8 * 15.42) - (9.42 + 19.42)) == pc.net_pay
#
# if __name__ == '__main__':
#     hourly_union_member_service_charge()
