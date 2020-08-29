import liwc
import numpy as np

from collections import Counter


class RIDCountsExtractor():
    def __init__(self, dictionary_path):
        self.categories = self.read_dic_categories(dictionary_path)
        self.parser, _ = liwc.load_token_parser(dictionary_path)


    def read_dic_categories(self, dictionary_path):
        with open(dictionary_path) as lines:
            for line in lines:
                if line.strip() == "%":
                    break
            categories = dict(self.parse_categories(lines))
        return categories


    def parse_categories(self, lines):
        for line in lines:
            line = line.strip()
            if line == "%":
                return
            if "\t" in line:
                category_id, category_name = line.split("\t", 1)
                yield category_id, category_name


    def get_rid_counts_list(self, tokenized_text_list):
        detailed_list_of_list = []
        count_list_of_list = []
        percentaje_list_of_list = []
        for tokenized_text in tokenized_text_list:

            detailed_list, count_list, percentaje_list = self.get_rid_counts(tokenized_text)

            detailed_list_of_list.append(detailed_list)
            count_list_of_list.append(count_list)
            percentaje_list_of_list.append(percentaje_list)


        return detailed_list_of_list, np.array(count_list_of_list), np.array(percentaje_list_of_list)


    def get_rid_counts(self, tokenized_text):
        counts = Counter(category for token in tokenized_text for category in self.parser(token))
        total_counts = 0
        for category_id, category_name in self.categories.items():
            if len(category_id) == 1:
                category_count = counts.get(category_name)
                if category_count is None:
                    category_count = 0
                total_counts = total_counts + category_count

        detailed_list = []
        count_list = []
        percentaje_list = []
        for category_id, category_name in self.categories.items():
            category_count = counts.get(category_name)
            if category_count is None:
                category_count = 0

            category_percentage = 0
            if total_counts > 0:
                category_percentage = round(category_count / total_counts * 100, 2)

            level = 1
            parent_categories = ""
            if len(category_id) > 1:
                level = 2
                parent_categories = self.categories.get(category_id[0])
                if len(category_id) > 3:
                    level = 3
                    parent_categories = parent_categories + " " + self.categories.get(category_id[0:3])

            count_list.append(category_count)
            percentaje_list.append(category_percentage)

            category_detail = {}
            category_detail["category_id"] = category_id
            category_detail["category_name"] = category_id
            category_detail["level"] = level
            category_detail["parent_categories"] = parent_categories
            category_detail["cat_count"] = category_count
            category_detail["cat_perc"] = category_percentage
            detailed_list.append(category_detail)

        return detailed_list, np.array(count_list), np.array(percentaje_list)
