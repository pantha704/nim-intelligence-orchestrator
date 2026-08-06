"""Safe arithmetic expression evaluator — AST-based, never uses eval().

Supports numbers, + - * / // % **, parentheses, unary signs, a whitelisted
set of math functions and the constants pi and e.
"""
import ast
import math

ALLOWED_FUNCTIONS = {
    "abs": abs,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "pow": math.pow,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "min": min,
    "max": max,
    "factorial": math.factorial,
}

ALLOWED_CONSTANTS = {"pi": math.pi, "e": math.e}


class ExpressionError(ValueError):
    """Raised when an expression is invalid, unsafe or uncomputable."""


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in ALLOWED_CONSTANTS:
            return ALLOWED_CONSTANTS[node.id]
        raise ExpressionError(f"unknown name '{node.id}'")
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ExpressionError("division by zero")
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise ExpressionError("division by zero")
            return left // right
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise ExpressionError("modulo by zero")
            return left % right
        if isinstance(node.op, ast.Pow):
            return left ** right
        raise ExpressionError("unsupported operator")
    if isinstance(node, ast.UnaryOp):
        value = _eval_node(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise ExpressionError("unsupported unary operator")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
            raise ExpressionError("unsupported function call")
        fn = ALLOWED_FUNCTIONS[node.func.id]
        args = [_eval_node(a) for a in node.args]
        if node.keywords:
            raise ExpressionError("keyword arguments not allowed")
        return float(fn(*args))
    raise ExpressionError(f"unsupported syntax: {type(node).__name__}")


def safe_eval_expression(expr: str) -> float:
    """Evaluate a numeric expression safely. Raises ExpressionError on any
    disallowed or uncomputable input (including zero division)."""
    expr = expr.strip()
    if not expr:
        raise ExpressionError("empty expression")
    # No assignment, lambdas, comprehensions, attribute access, etc.
    if any(tok in expr for tok in ("=", "lambda", "import", "__", "while", "for")):
        raise ExpressionError("unsupported token in expression")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"invalid expression: {e}") from e
    return _eval_node(tree.body)
