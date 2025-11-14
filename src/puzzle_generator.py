# src/puzzle_generator.py
import random
from typing import Tuple

class PuzzleGenerator:
    """
    Generates child-friendly arithmetic puzzles for three difficulty levels.
    Ensures integer answers for division by creating divisible pairs.
    """

    def __init__(self):
        self.operations = ['+', '-', '*', '/']

    def _int_pair_divisible(self, a_min, a_max, b_min, b_max) -> Tuple[int,int]:
        b = random.randint(b_min, b_max)
        quotient_min = max(1, a_min // max(1, b))
        quotient_max = max(1, a_max // max(1, b))
        quotient = random.randint(quotient_min, quotient_max)
        a = b * quotient
        return a, b

    def generate(self, level: str) -> Tuple[str, float]:
        """Return (question_str, answer_as_number)."""
        level = level.capitalize()
        op = random.choice(self.operations)

        if level == 'Easy':
            if op == '+':
                a, b = random.randint(1, 10), random.randint(1, 10)
            elif op == '-':
                a, b = random.randint(1, 10), random.randint(1, 10)
                if b > a: a, b = b, a
            elif op == '*':
                a, b = random.randint(1, 5), random.randint(1, 5)
            else:
                a, b = self._int_pair_divisible(1, 25, 1, 5)

        elif level == 'Medium':
            if op == '+':
                a, b = random.randint(10, 50), random.randint(1, 50)
            elif op == '-':
                a, b = random.randint(10, 60), random.randint(1, 50)
                if b > a: a, b = b, a
            elif op == '*':
                a, b = random.randint(2, 12), random.randint(2, 12)
            else:
                a, b = self._int_pair_divisible(10, 120, 2, 12)

        else:  # Hard
            if op == '+':
                a, b = random.randint(50, 200), random.randint(20, 150)
            elif op == '-':
                a, b = random.randint(50, 200), random.randint(0, 150)
                if b > a: a, b = b, a
            elif op == '*':
                a, b = random.randint(6, 20), random.randint(6, 20)
            else:
                a, b = self._int_pair_divisible(50, 400, 2, 20)

        question = f"{a} {op} {b}"
        if op == '+':
            answer = a + b
        elif op == '-':
            answer = a - b
        elif op == '*':
            answer = a * b
        else:
            answer = a // b

        return question, float(answer)
