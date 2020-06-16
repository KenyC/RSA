# %%

%run rsa.py

# %%

from exh import *
from exh.prop import true

# %%

alts = [true, a | b, a, b, a & b]
universe = Universe(fs = alts)

rsa = RSA(universe, alts)

# %%
rsa.compute(1)
rsa.heatmap_listener()

# %%
rsa.compute(3, reset = False)

print("LISTENER")
rsa.heatmap_listener()
print("SPEAKER")
rsa.heatmap_speaker()

# %%
d = Pred(3, name = "d", depends = "x")
alts = [a | ~ a, Ex > d, Ax > d]
universe = Universe(fs = alts)

rsa = RSA(universe, alts)

# %%
rsa.compute(1)
rsa.heatmap_listener()

# %%

rsa.compute_speaker()
rsa.heatmap_speaker()


# %%

class A:
	a = 1

	def f():
		return B()

class B(A):
	a = 2
	def g():
		pass

class C(A):
	pass

# %%

print(A.a)
print(B.a)
print(C.a)
# %%