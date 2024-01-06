import pandas as pd


def read_points_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # First line contains the number of points
    num_points = int(lines[0].strip())

    # Initialize arrays to hold the two columns of data
    column1 = []
    column2 = []

    # Read the data from the remaining lines
    for line in lines[1:num_points+1]:  # Ensuring we only read 'num_points' lines
        data = line.strip().split('\t')  # Splitting the line by tab
        column1.append(float(data[0]))   # Assuming the data is of type float
        column2.append(float(data[1]))

    return column1, column2


def lists_to_excel(
    hm_list,
    qt_list,
    Gt_list,
    H_0t_list,
    msn_list
):
    # Create a DataFrame from the lists with specified column names
    df = pd.DataFrame({
        'hm': hm_list, 'qt': qt_list,
        'Gt': Gt_list, 'H_0': H_0t_list,
        'MSN': msn_list
    })

    # Write the DataFrame to an Excel file
    # The 'index=False' parameter prevents pandas from writing row indices
    df.to_excel("results.xlsx", index=False)
