"""Experiment 4: Create a Pandas DataFrame with student names and grades."""
import pandas as pd

data = {
    "Student Name": ["Arun", "Priya", "Kavin", "Meena"],
    "Grade": ["A", "B", "A+", "B+"],
}
df = pd.DataFrame(data)
print(df)
