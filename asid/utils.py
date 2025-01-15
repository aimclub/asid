from typing import Any


def validate_number(x: Any, num_type: type, num_cl: str):
    """
    Validates the type and classification of a numeric parameter.

    Parameters:
    - x (Any): The value to check.
    - num_type (type): The expected numeric type (e.g., int, float).
    - num_cl (str): The classification of the number, such as "positive", "non-negative", or "negative".

    Raises:
    - TypeError: If the parameter is not of the specified numeric type.
    - ValueError: If the parameter does not match the specified classification.
    """

    if isinstance(x, num_type):
        if num_cl == "positive" and x <= 0:
            raise ValueError("The parameter should be " + num_cl + ".")
        elif num_cl == "non-negative" and x < 0:
            raise ValueError("The parameter should be " + num_cl + ".")
        elif num_cl == "negative" and x >= 0:
            raise ValueError("The parameter should be " + num_cl + ".")
    else:
        raise TypeError("The parameter should be of " + str(num_type))
