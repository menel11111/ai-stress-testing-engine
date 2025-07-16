import pandas as pd

class StressTester:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def simulate_revenue_drop(self, drop_pct=0.4):
        latest_cash = self.df['cash_balance'].iloc[-1]
        latest_expenses = self.df['expenses'].iloc[-1]
        latest_loan = self.df['loan_payment'].iloc[-1]
        stressed_revenue = self.df['revenue'].iloc[-1] * (1 - drop_pct)

        cash = latest_cash
        survival_months = 0

        while cash > 0:
            monthly_burn = latest_expenses + latest_loan - stressed_revenue
            cash -= monthly_burn
            survival_months += 1
            if survival_months > 120:
                break

        return {
            "stressed_revenue": int(stressed_revenue),
            "survival_months": survival_months,
            "monthly_burn": int(monthly_burn)
        }
