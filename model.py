import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

dataset_path = "C:\\Users\\khura\\OneDrive\\Desktop\\MLLabEval\\Fuel_cell_performance_data-Full.csv"
df = pd.read_csv(dataset_path)

print("Dataset Info:")
print(df.info())

if df.isnull().sum().sum() > 0:
    print("\nHandling missing values...")
    df = df.dropna() 


roll_number = "102203355"  
last_digit = int(roll_number[-1])

target_map = {
    0: "Target1", 5: "Target1",
    1: "Target2", 6: "Target2",
    2: "Target3", 7: "Target3",
    3: "Target4", 8: "Target4",
    4: "Target5", 9: "Target5"
}
selected_target = target_map[last_digit]
print(f"\nSelected Target: {selected_target}")

df = df[[selected_target, *df.columns.difference([selected_target])]]

X = df.drop(columns=[selected_target])
y = df[selected_target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    return predictions

lr = LinearRegression()
evaluate_model(lr, X_train, X_test, y_train, y_test)

dt = DecisionTreeRegressor(random_state=42)
evaluate_model(dt, X_train, X_test, y_train, y_test)

rf = RandomForestRegressor(random_state=42)
evaluate_model(rf, X_train, X_test, y_train, y_test)

results = {
    "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
    "MSE": [mean_squared_error(y_test, lr.predict(X_test)),
            mean_squared_error(y_test, dt.predict(X_test)),
            mean_squared_error(y_test, rf.predict(X_test))],
    "R2 Score": [r2_score(y_test, lr.predict(X_test)),
                 r2_score(y_test, dt.predict(X_test)),
                 r2_score(y_test, rf.predict(X_test))]
}
result_df = pd.DataFrame(results)
print("\nFinal Results:")
print(result_df)
