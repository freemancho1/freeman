from datetime import datetime


def eta(num_of_remaining_count, current_processing_time):
    return datetime.now() + num_of_remaining_count * current_processing_time