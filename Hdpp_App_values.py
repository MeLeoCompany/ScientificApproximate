import pandas as pd

def lists_to_excel(list1, list2, file_name):
    # Create a DataFrame from the lists with specified column names
    df = pd.DataFrame({'dHpp': list1, 'App': list2})
    
    # Write the DataFrame to an Excel file
    # The 'index=False' parameter prevents pandas from writing row indices
    df.to_excel(file_name, index=False)

output_file_name = 'output.xlsx'

# G = 1.0
# hm = 0.05
# DeltaH_pp = 2*G/(3)**0.5
# dHpp = []
# App = []
# while (hm < 16.0):
#     dHpp.append(DeltaH_pp*(((hm/DeltaH_pp)**2) + 5 - 2*(4 + (hm/DeltaH_pp)**2)**0.5)**0.5)
#     App.append((4 *((1/DeltaH_pp)**2)*(hm/DeltaH_pp))/(3*((hm / DeltaH_pp)**2) + 8 + (((hm / DeltaH_pp)**2) + 4 )**(3/2))**(0.5))
#     hm += 0.05

# lists_to_excel(dHpp, App, output_file_name)
# print(f"Lists have been written to {output_file_name}")