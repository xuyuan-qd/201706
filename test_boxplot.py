import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

x = [[1,2,3],[2,3,4]]
y = [[1,2],[3,4]]
plt.figure()
plt.boxplot(x)
plt.savefig("test.png")
plt.figure()
plt.boxplot(y)
plt.savefig("testy.png")

