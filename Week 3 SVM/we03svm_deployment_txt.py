import pickle
import pandas as pd

# Uncomment the following snippet of code to debug problems with finding the .pkl file path
# This snippet of code will exit the progrintam and print the current working directory.
# import os
# exit(os.getcwd())

mower_model = pickle.load(open(r'C:/Users/Meghanjali/Desktop/Data science programming/WE03 ASGT/best_svm_poly.pkl',"rb"))

print("\n*****************************************************")
print("* The riding mower prediction model *")
print("*****************************************************\n")
Income= float(input("Enter the income "))
Lot_Size=float(input("Enter the lot size of individual"))
df = pd.DataFrame({'Income': [Income],'Lot_Size':[Lot_Size]})
result = mower_model.predict(df)
probability = mower_model.predict_proba(df)
ownership = ('Nonowner', 'Owner')
print(f"\nThe mower model indicates probability of ownership at {probability[0][1]:.4f}, therefore it's indicated that we should {ownership[result[0]]}.\n")