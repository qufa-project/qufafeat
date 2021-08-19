import random

from featuretools.primitives import utils


class OperatorManager:
    """
    1.
    현재 설계 상으로는 특징 추출시 연산자를 사용자가 지정하도록 되어 있으나, 사용자가 연산자를 지정하지 않는 방식으로 특징 추출이 구현될 수 있음.
    이러한 경우, 전체 operator를 반환하여 특징 추출을 지원하는 용도로 이 클래스를 활용함.
    2.
    지금 구현에서는 사용자가 단순히 operator 이름만 지정하는데, featuretool의 dfs 입력은 transform이나 aggregation 타입을 구분하여 operator를 지정해야 함.
    이를 위해 사용자가 지정한 operator에 해당하는 유형을 반환함.
    """
    def __init__(self, operators: list):
        if not operators:
            self.operators = None
        else:
            self.operators = set(str.lower(op) for op in operators)
        self.transforms = utils.get_transform_primitives().keys()
        self.aggregations = utils.get_aggregation_primitives().keys()
        self.premium_transform = {
            "absolute_diff", "age_over_18", "age_over_25", "age_over_65",
            "age_under_18",
            "age_under_65",
            "natural_language_to_year",
            "nth_week_of_month",
            "part_of_day",
            "percent_change",
            "phone_number_to_country",
            "polarity_score",
            "punctuation_count",
            "quarter",
            "same_as_previous",
            "savgol_filter",
            "score_percentile",
            "season",
            "sign",
            "stopword_count",
            "sub_region_code_to_region",
            "title_word_count",
            "upper_case_count",
            "upper_case_word_count",
            "url_to_protocol",
            "zip_code_to_state",
            "count_string",
            "cumulative_time_since_last_false",
            "cumulative_time_since_last_true",
            "date_to_time_zone",
            "day_name",
            "greater_than_previous",
            "is_first_occurrence",
            "is_last_occurrence",
            "is_max_so_far",
            "is_min_so_far",
            "is_whole_number",
            "is_zero",
            "lag",
            "less_than_previous",
            "mean_characters_per_word"
        }
        self.premium_aggregation = {
            "autocorrelation", "correlation", "n_most_common_frequency", "n_unique_days",
            "n_unique_days_of_calendar_year", "n_unique_days_of_month", "n_unique_months", "n_unique_weeks",
            "num_consecutive_greater_mean", "num_consecutive_less_mean",
            "num_false_since_last_true", "num_peaks", "num_true_since_last_false",
            "num_zero_crossings", "path_length", "percent_unique", "time_since_last_false",
            "time_since_last_max", "time_since_last_min", "time_since_last_true", "variance",
            "count_above_mean", "count_below_mean", "count_greater_than", "count_inside_range",
            "count_less_than", "count_outside_range", "count_inside_nth_std", "count_outside_nth_std",
            "date_first_event", "has_no_duplicates", "is_monotonically_decreasing",
            "is_monotonically_increasing", "is_unique", "kurtosis", "max_consecutive_false",
            "max_consecutive_negatives", "max_consecutive_positives", "max_consecutive_true",
            "max_consecutive_zeros", "max_count", "max_min_delta", "median_count", "min_count"
        }

    def get_transform_operators(self):
        if self.operators is None:
            return random.sample(self.transforms - self.premium_transform, 15)
        return set(self.operators & self.transforms)

    def get_aggregation_operators(self):
        if self.operators is None:
            return random.sample(self.aggregations - self.premium_aggregation, 10)
        return set(self.operators & self.aggregations)
