def check_in(name, value, allowed_values):
    if value not in allowed_values:
        raise ValueError(
            "Invalid value for '{}': {}. Must be in {}.".format(
                name, value, allowed_values
            )
        )


def check_greater_equal(name, value, greater_equal_value):
    if value < greater_equal_value:
        raise ValueError(
            "Invalid value for '{}': {}. Must be >= {}.".format(
                name, value, greater_equal_value
            )
        )


def check_less_equal(name, value, less_equal_value):
    if value > less_equal_value:
        raise ValueError(
            "Invalid value for '{}': {}. Must be <= {}.".format(
                name, value, less_equal_value
            )
        )


def check_between(name, value, lower_bound, upper_bound):
    check_greater_equal(name, value, lower_bound)
    check_less_equal(name, value, upper_bound)


def check_not_none(name, value):
    if value is None:
        raise ValueError("'{}' cannot be None".format(name))
