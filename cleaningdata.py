from mysklearn.mypytable import MyPyTable 


file_path = "/home/FinalProject/neo.csv"
output_file = "/home/FinalProject/neo_clean.csv"
neo_table = MyPyTable()
neo_table.load_from_file(file_path)

removed_attributes =["id", "name", "orbiting_body", "sentry_object"]

columns_to_keep = [col for col in neo_table.column_names if col not in removed_attributes]

cleaned_data = [[row[neo_table.column_names.index(col)] for col in columns_to_keep] for row in neo_table.data]

cleaned_table = MyPyTable(column_names=columns_to_keep, data=cleaned_data)
cleaned_table.save_to_file(output_file)
