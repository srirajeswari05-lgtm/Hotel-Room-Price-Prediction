from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

with open("lightgbm_room_price_model.sav", "rb") as f:
    model = pickle.load(f)

try:
    training_columns = model.feature_name_
except:
    training_columns = model.booster_.feature_name()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None

    if request.method == "POST":
        hotel = request.form["hotel"]
        lead_time = int(request.form["lead_time"])
        arrival_date_year = int(request.form["arrival_date_year"])
        arrival_date_month = request.form["arrival_date_month"]
        arrival_date_day_of_month = int(request.form["arrival_date_day_of_month"])
        stays_in_weekend_nights = int(request.form["stays_in_weekend_nights"])
        stays_in_week_nights = int(request.form["stays_in_week_nights"])
        adults = int(request.form["adults"])
        children = int(request.form["children"])
        babies = int(request.form["babies"])
        meal = request.form["meal"]
        country = request.form["country"]
        market_segment = request.form["market_segment"]
        reserved_room_type = request.form["reserved_room_type"]
        customer_type = request.form["customer_type"]

        arrival_date = pd.to_datetime(
            f"{arrival_date_year}-{arrival_date_month}-{arrival_date_day_of_month}",
            errors="coerce"
        )

        total_stay = stays_in_weekend_nights + stays_in_week_nights
        total_guests = adults + children + babies
        arrival_month_num = arrival_date.month
        arrival_day_name = arrival_date.day_name()

        weekend_flag = 1 if arrival_day_name in ["Saturday", "Sunday"] else 0
        holiday_flag = 1 if arrival_month_num in [5, 6, 7, 8, 12] else 0

        if arrival_month_num in [12, 1, 2]:
            season = "Winter"
        elif arrival_month_num in [3, 4, 5]:
            season = "Summer"
        elif arrival_month_num in [6, 7, 8, 9]:
            season = "Monsoon"
        else:
            season = "Autumn"

        occupancy_rate = min((total_guests / (total_stay + 1)) * 100, 100)

        input_df = pd.DataFrame([{
            "hotel": hotel,
            "lead_time": lead_time,
            "arrival_date_year": arrival_date_year,
            "arrival_date_month": arrival_date_month,
            "arrival_date_week_number": arrival_date.isocalendar().week,
            "arrival_date_day_of_month": arrival_date_day_of_month,
            "stays_in_weekend_nights": stays_in_weekend_nights,
            "stays_in_week_nights": stays_in_week_nights,
            "adults": adults,
            "children": children,
            "babies": babies,
            "meal": meal,
            "country": country,
            "market_segment": market_segment,
            "distribution_channel": "TA/TO",
            "is_repeated_guest": 0,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 0,
            "reserved_room_type": reserved_room_type,
            "assigned_room_type": reserved_room_type,
            "booking_changes": 0,
            "deposit_type": "No Deposit",
            "agent": 0,
            "company": 0,
            "days_in_waiting_list": 0,
            "customer_type": customer_type,
            "required_car_parking_spaces": 0,
            "total_of_special_requests": 0,
            "reservation_status": "Check-Out",
            "arrival_day": arrival_date.day,
            "arrival_month_num": arrival_month_num,
            "arrival_year": arrival_date.year,
            "arrival_day_name": arrival_day_name,
            "weekend_flag": weekend_flag,
            "total_stay": total_stay,
            "total_guests": total_guests,
            "season": season,
            "holiday_flag": holiday_flag,
            "room_changed": 0,
            "base_price": 100,
            "price_difference": 0,
            "competitor_price": 105,
            "occupancy_rate": occupancy_rate
        }])

        input_encoded = pd.get_dummies(input_df, drop_first=True)
        input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)

        predicted_price = model.predict(input_encoded)
        prediction = round(float(predicted_price[0]), 2)

    return render_template("predict.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)