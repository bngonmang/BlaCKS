# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from pyparsing import Regex, opAssoc, nums, Word, oneOf, operatorPrecedence, ParserElement, ParseException
import pandas as pd


"""
We define a parser with the infix operators: * / + - = <= >= < > and or.
isnull is a unary operator before the panda.
Strings are quoted and unquoted strings are row of the current dataframe.
"""


class NoColumnError(KeyError):
    """ Exception raised when a needed column is not found. """
    pass


class DSLTypeError(TypeError):
    """ Raised when we try to apply an op to an operand on which it's not supposed to work with. """
    pass


EPSILON = 1e-4

ParserElement.enablePackrat()
Series = pd.core.series.Series

OPERATOR_CD = '* /'
OPERATOR_PM = '+ -'
OPERATOR_EQ = '= <= >= < > !='
OPERATOR_BOOL = 'and or'


def equal(x, y):
    x_is_number = is_number(x)
    # TODO check that x and y are both number or non numbers
    if x_is_number:
        return abs(x - y) < EPSILON
    else:
        return x == y


def to_float(x, y):
    return convert_val(x, float), convert_val(y, float)


def to_bool(x, y):
    return convert_val(x, bool), convert_val(y, bool)


def type_equal(x, y):
    if is_number(x) != is_number(y):
        raise DSLTypeError
    if is_number(x) and is_number(y):
        return to_float(x, y)
    return x, y


def builtin_and(x, y):
    try:
        return x and y
    except ValueError:
        return x & y


def builtin_or(x, y):
    try:
        return x or y
    except ValueError:
        return x | y


ops = {
    '+': (to_float, lambda x, y: x + y),
    '-': (to_float, lambda x, y: x - y),
    '*': (to_float, lambda x, y: x * y),
    '/': (to_float, lambda x, y: x / y),
    '<=': (to_float, lambda x, y: x <= y),
    '!=': (to_float, lambda x, y: x != y),
    '>=': (to_float, lambda x, y: x >= y),
    '<': (to_float, lambda x, y: x < y),
    '>': (to_float, lambda x, y: x > y),
    '=': (type_equal, equal),
    'and': (to_bool, lambda x, y: x and y),
    'or': (to_bool, builtin_or),
}


def get_parser():
    symbol = Regex('(\'[^\']+\')|([a-zA-Z\'_][a-zA-Z0-9\'_]*)')  # sym
    number = Word(nums)
    floatNumber = Word(nums+'.'+nums)
    boolean = oneOf('True False')
    operator_cd = oneOf(OPERATOR_CD)
    operator_pm = oneOf(OPERATOR_PM)
    operator_eq = oneOf(OPERATOR_EQ)
    operator_bool = oneOf(OPERATOR_BOOL)
    operand = boolean | floatNumber | number| symbol
    expr = operatorPrecedence(operand,
                              [
                                  ('not', 1, opAssoc.RIGHT),
                                  ('-', 1, opAssoc.RIGHT),
                                  ('isnull', 1, opAssoc.RIGHT),
                                  (operator_cd, 2, opAssoc.RIGHT),
                                  (operator_pm, 2, opAssoc.RIGHT),
                                  (operator_eq, 2, opAssoc.RIGHT),
                                  (operator_bool, 2, opAssoc.RIGHT),
                              ])
    return expr


parser = get_parser()


def remove_stop_characters(string):
    """ Removes white space, ")" and "(" in expression (str).  """
    return string.replace(' ', '').replace('(', '').replace(')', '').replace('\t', '')


def parse(expr):
    """
    Parse expr to a list (tree) or to a string if string is True
    """
    parsed = parser.parseString(expr).asList()[0]
    folded_expr = fold(parsed)
    reconstructed_expr = remove_stop_characters(tree_to_string(folded_expr))
    if reconstructed_expr != remove_stop_characters(expr):
        raise ParseException('The expression isn\'t handled well by the system, received {} '
                             'transformed it to {}'.format(expr, tree_to_string(reconstructed_expr)))
    return folded_expr


def fold_string(expr):
    """ Returns a string with parenthesis. """
    return tree_to_string(parse(expr), parentheses=False)


def is_number(x):
    if isinstance(x, Series):
        if x.dtype == bool:
            return False
        if any(isinstance(e, (bool, basestring)) for e in x):
            return False
        return True
    else:
        if isinstance(x, bool):
            return False
        return isinstance(x, (int, long, float, complex))


def tree_to_string(expr, parentheses=False):
    """ Converts an AST to a str """
    if isinstance(expr, list):
        rv = ' '.join([tree_to_string(e, parentheses=True) for e in expr])
        return '({})'.format(rv) if parentheses else rv
    return expr


def evaluate_rec(expr, current_df):
    #print (expr)
    if not isinstance(expr, list):
        # If we have just a single value.
        result = evaluate_value(expr, current_df)

        return result

    # If we have an expression to evaluate.
    try:
        if len(expr) == 2:

            operator, rexpr = expr
            return evaluate_unary_op(operator,
                                     evaluate_rec(rexpr, current_df))
        elif len(expr) == 3:

            lexpr, operator, rexpr = expr
            result = evaluate_op(evaluate_rec(lexpr, current_df),
                               operator,
                               evaluate_rec(rexpr, current_df))

            return result
    except DSLTypeError:
        raise DSLTypeError('Type error on evaluating "{}"'.format(tree_to_string(expr)))
    except ValueError:
        raise ValueError('"{}" is not allowed in "{}"'.format(operator, tree_to_string(expr)))


def evaluate_unary_op(operator, val):
    """
    :param operator: - or isnull
    :param val:
    """
    if operator == '-':
        return evaluate_op(0, '-', val)
    if operator == 'isnull':
        try:
            return val.isnull()
        except Exception as e:
            print(e)
            raise e
    if operator == 'not':
        return ~val
    raise ValueError


def leaf_is_float(leaf):
    return not any(x not in '.0123456789' for x in leaf)


def evaluate_value(val, current_df=None):
    """ We check the type value as bool, string and number.
    :param val:
    :return:
    """
    if val == 'True':
        return True
    if val == 'False':
        return False
    if leaf_is_float(val):
        return float(val)
    if val[0] == val[-1] == "'":
        return unicode(val[1:-1])
    try:
        return current_df[val]
    except KeyError as e:
        raise NoColumnError('No column named {} in dataframe'.format(val))


def evaluate_rule_on_df(expr, current_df):
    """
    Evaluate the rule on the dataframe.
    If a column is given, metrics are generated.
    Returns a pd.Serie or np.array
    """
    parsed_expr = parse(expr)
    try:
        return evaluate_rec(parsed_expr, current_df)
    except DSLTypeError as e:
        return e.message
    except NoColumnError as e:
        return e.message
    except ValueError as e:
        return e.message


def convert_val(val, type_):
    """
    Tries to convert val to the given type.
    """
    try:
        if isinstance(val, Series):
            return val.apply(type_)
        return type_(val)
    except ValueError:
        raise DSLTypeError


def evaluate_op(left, op, right):
    """
    Evaluates the right and left operand using op.
    """
    # error checking
    type_converter = ops[op][0]
    left, right = type_converter(left, right)
    func = ops[op][1]
    return func(left, right)


def fold(expr):
    """folds the exp as a tree: eg:
    in : [['5', '+', '3', '-', '1'], '+', '1', '+', '1']
    out: [[[['5', '+', '3'], '-', '1'], '+', '1'], '+', '1']
    """
    if not isinstance(expr, list):
        return expr
    expr = map(fold, expr)
    while 3 < len(expr):
        expr = [expr[:3]] + expr[3:]
    return expr


def leaf_iterator(tree):
    """ Iterate over the leaves of an ast. """
    if not isinstance(tree, list):
        yield tree
    else:
        for subtree in tree:
            for leaf in leaf_iterator(subtree):
                yield leaf

def get_variables_names_from_tree(tree):
    return filter(is_column, set(leaf_iterator(tree)))


def is_column(leaf):
    """ Return if the leaf is a panda column """
    if leaf in ops.keys():
        return False
    if leaf in ['True', 'False', 'isnull']:
        return False
    if leaf[0] == leaf[-1] == "'":
        return False
    if leaf_is_float(leaf):
        return False
    return True


if __name__ == '__main__':
    a = [[[['5', '+', '3'], '-', '1'], '+', '1'], '+', '1']
    print([x for x in leaf_iterator(a)])
