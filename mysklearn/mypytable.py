from mysklearn import myutils

##############################################
# Programmer: Leo Jia
# Class: CPSC 322, Fall 2024
# Programming Assignment #2
# 9/25/24
# 
# Description: this program does pa2
##############################################

import copy
import csv
from tabulate import tabulate

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names) # TODO: fix this

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        if isinstance(col_identifier, str):
            try:
                col_idx = self.column_names.index(col_identifier)
            except ValueError:
                raise ValueError(f"Column '{col_identifier}' does not exist in table.")
        elif isinstance(col_identifier, int):
            col_idx = col_identifier
        else:
            raise ValueError("Column Identifier should be a string or integer.")
        
        column = [row[col_idx] for row in self.data if include_missing_values or row[col_idx] != "NA"]

        return column

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for i, row in enumerate(self.data):
            for j, value in enumerate(row):
                try:
                    self.data[i][j] = float(value)
                except ValueError:
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        self.data = [row for idx, row in enumerate(self.data) if idx not in row_indexes_to_drop]

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            self.column_names = next(reader)
            self.data = [row for row in reader]
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.column_names)
            writer.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        key_indices = [self.column_names.index(key) for key in key_column_names]

        seen_rows = set()
        duplicate_indicies = []

        for i, row in enumerate(self.data):
            key_values = tuple(row[idx] for idx in key_indices)
            if key_values in seen_rows:
                duplicate_indicies.append(i)
            else:
                seen_rows.add(key_values)

        return duplicate_indicies

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        self.data = [row for row in self.data if "NA" not in row]

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col_idx = self.column_names.index(col_name)
        valid_values = [float(row[col_idx]) for row in self.data if row[col_idx] != "NA"]
        if valid_values:
            avg = sum(valid_values) / len(valid_values)
            for row in self.data:
                if row[col_idx] == "NA":
                    row[col_idx] = avg

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        stats_table = MyPyTable(["attribute", "min", "max","mid", "avg", "median"])
        
        def calculate_median(data):
            sorted_data = sorted(data)
            n = len(sorted_data)
            m = n // 2
            if n % 2 == 0:
                return (sorted_data[m - 1] + sorted_data[m]) / 2
            else:
                return sorted_data[m]
        
        for col_name in col_names:
            column_data = self.get_column(col_name, include_missing_values = False)

            if not column_data:
                continue

            col_min = min(column_data)
            col_max = max(column_data)
            col_mid = (col_min + col_max) / 2
            col_avg = sum(column_data) / len(column_data)
            col_median = calculate_median(column_data)

            stats_table.data.append([col_name, col_min, col_max, col_mid, col_avg, col_median])


        return stats_table

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        self_key_indicies = [self.column_names.index(key) for key in key_column_names]
        other_key_indicies = [other_table.column_names.index(key) for key in key_column_names]

        joined_column_names = self.column_names[:]
        for col_name in other_table.column_names:
            if col_name not in key_column_names:
                joined_column_names.append(col_name)

        joined_data = []

        for self_row in self.data:
            self_key_values = tuple([self_row[idx] for idx in self_key_indicies])
            for other_row in other_table.data:
                other_key_values = tuple([other_row[idx] for idx in other_key_indicies])
                if self_key_values == other_key_values:
                    joined_row = self_row[:]
                    for i, value in enumerate(other_row):
                        if i not in other_key_indicies:
                            joined_row.append(value)
                    joined_data.append(joined_row)

        return MyPyTable(joined_column_names, joined_data)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        self_key_indices = [self.column_names.index(key) for key in key_column_names]
        other_key_indices = [other_table.column_names.index(key) for key in key_column_names]

        combined_column_names = self.column_names + [col for col in other_table.column_names if col not in key_column_names]

        joined_data = []
        other_matched = [False] * len(other_table.data)

        for self_row in self.data:
            self_key_values = [self_row[idx] for idx in self_key_indices]
            match_found = False
            for i, other_row in enumerate(other_table.data):
                other_key_values = [other_row[idx] for idx in other_key_indices]
                if self_key_values == other_key_values:
                    match_found = True
                    other_matched[i] = True
                    joined_row = self_row + [other_row[other_table.column_names.index(col)] for col in other_table.column_names if col not in key_column_names]
                    joined_data.append(joined_row)

            if not match_found:
                joined_row = self_row[:] + ["NA"] * (len(other_table.column_names) - len(key_column_names))
                joined_data.append(joined_row)
                print(f"Unmatched row from self: {joined_row}")

        for i, other_row in enumerate(other_table.data):
            if not other_matched[i]:
                other_keys = [other_row[idx] for idx in other_key_indices]
                joined_row = ["NA"] * len(self.column_names)
                for idx, key in enumerate(key_column_names):
                    joined_row[self.column_names.index(key)] = other_keys[idx]
                joined_row += [other_row[other_table.column_names.index(col)] for col in other_table.column_names if col not in key_column_names]
                joined_data.append(joined_row)
                    
        
        print(f"Final joined data: {joined_data}")

        return MyPyTable(column_names = combined_column_names, data=joined_data)
