import pandas as pd
import tensorflow as tf

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def get_training_dataset(solar_path, sunspot_path, label_path, sample_size=100000):

    solar_data = pd.read_csv(solar_path, nrows=sample_size)
    sunspot_data = pd.read_csv(sunspot_path, nrows=sample_size)
    label_data = pd.read_csv(label_path, nrows=sample_size)

    def process_timedelta(df):
        df["timedelta"] = pd.to_timedelta(df["timedelta"])
        df["day_number"] = df["timedelta"].dt.days
        return df

    solar_data = process_timedelta(solar_data)
    sunspot_data = process_timedelta(sunspot_data)
    label_data = process_timedelta(label_data)

    merged_data = pd.merge(
        solar_data,
        sunspot_data,
        on="day_number",
        how="inner"
    )

    merged_data = pd.merge(
        merged_data,
        label_data,
        on="day_number",
        how="inner"
    )

    feature_columns = [
        "bx_gse",
        "by_gse",
        "bz_gse",
        "theta_gse",
        "phi_gse",
        "bx_gsm",
        "by_gsm",
        "bz_gsm",
        "theta_gsm",
        "phi_gsm",
        "bt",
        "density",
        "speed",
        "temperature",
        "source"
    ]

    X = merged_data[feature_columns]
    y = merged_data["dst"]

    return X, y


solar_file = "/content/drive/MyDrive/aurora/Train_on_this/solar_wind.csv"
sunspot_file = "/content/drive/MyDrive/aurora/Train_on_this/sunspots_smooth.csv"
label_file = "/content/drive/MyDrive/aurora/Train_on_this/labels(dst).csv"

X, y = get_training_dataset(
    solar_file,
    sunspot_file,
    label_file
)

X = X.select_dtypes(include=["number"])

imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()

X = imputer.fit_transform(X)
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = Sequential()

model.add(
    Dense(
        128,
        activation="relu",
        input_shape=(X_train.shape[1],)
    )
)

model.add(
    Dense(
        64,
        activation="relu"
    )
)

model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.20,
    verbose=1
)

model_path = "/content/drive/MyDrive/aurora/trained_model1.h5"

model.save(model_path)

print("Training completed successfully.")
print("Model saved at:", model_path)

predictions = model.predict(X_test)
predictions = predictions.flatten()

rmse = mean_squared_error(
    y_test,
    predictions,
    squared=False
)

print("RMSE:", rmse)
