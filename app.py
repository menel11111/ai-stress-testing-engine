from engine.model import RevenueForecaster
from engine.stress_test import StressTester

def main():
    data_file = "my_big_financial_data.csv"

    print("\n📊 Running AI Revenue Forecast...")
    model = RevenueForecaster(data_file)
    model.train()
    forecast = model.forecast_next()
    print(f"📈 Forecasted next-month revenue: AED {forecast}")

    print("\n⚠️ Running Stress Test (40% Revenue Drop)...")
    stress = StressTester(data_file)
    result = stress.simulate_revenue_drop(drop_pct=0.4)
    print(f"  - New revenue: AED {result['stressed_revenue']}")
    print(f"  - Monthly burn: AED {result['monthly_burn']}")
    print(f"  - Survival period: {result['survival_months']} months")

if __name__ == "__main__":
    main()
