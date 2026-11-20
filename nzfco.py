from itertools import combinations
from math import factorial

"""
Implementing Wiens & Servedio (2000) method
Non-zero frequency cut-off to identify diagnostic character states
"""

def major_f(n_ind: int, p_min: float):
	f = (1.0 - p_min) ** n_ind
	return f


def major_p(n: int, p: float, k: int, c: int):
	"""
	n : Number of individuals sampled.

	p : Probability of I don't know.

	k : Number of characters seemed to be diagnostic.

	c : Total number of characters explored.

	"""

	pfinal = 0
	f = major_f(n, p)

	#for r in range(1, k+1):
	for r in range(k, c+1):
		th = factorial(c) / (factorial(r) * factorial(c - r))
		th *= f ** r * (1 - f) ** (c - r)
		pfinal += th

	return pfinal


tab = [
	(6, 0.01, 50, 50),
	(7, 0.01, 45, 45),
	(7, 0.01, 50, 50),
	(8, 0.01, 40, 40),
	(8, 0.01, 45, 45),
	(8, 0.01, 50, 50),
	(35, 0.01, 9, 9),
]

for th in tab:
	print(th)
	print(major_p(*th))