
import pandas as pd
import os

base_path = r'c:\Users\tdamelio\Desktop\campeones_analysis\data\sourcedata\xdf\sub-27'
file_path = os.path.join(base_path, 'order_matrix_27_A_block1_VR.xlsx')

try:
    df = pd.read_excel(file_path)
    print("Unique Dimension:", df['dimension'].unique())
    print("Unique Order:", df['order_emojis_slider'].unique())
except Exception as e:
    print(e)
