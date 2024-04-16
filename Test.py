from Structures import BalchStruct, ClopperPearson, KNBalchStruct, NormTPivot, ECDFStruct
import numpy as np
import matplotlib.pyplot as plt

xs = np.linspace(0,1,501)
k, n = 4, 10
S = BalchStruct(k, n)
print(S.cut(0.4))
print(S.possibility(0.7))
plt.plot(xs, [S.possibility(x) for x in xs])
plt.show()

S = ClopperPearson(k, n)
print(S.cut(0.4))
print(S.possibility(0.7))
plt.plot(xs, [S.possibility(x) for x in xs])
plt.show()

# Seems a little squiffy on the left
S = KNBalchStruct(k, n, maxn = 100)
print(S.cut(0.4))
print(S.possibility(0.7))
plt.plot(xs, [S.possibility(x) for x in xs])
plt.show()

xs = np.linspace(-1,1,501)
d = np.random.randn(20)
S = NormTPivot(d)
print(S.cut(0.4))
print(S.possibility(0.7))
plt.plot(xs, [S.possibility(x) for x in xs])
plt.show()

S = ECDFStruct(d, origin = 0)
print(S.cut(0.4))
print(S.possibility(0.7))
plt.plot(xs, [S.possibility(x) for x in xs])
plt.show()
