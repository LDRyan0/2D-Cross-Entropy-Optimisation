import scipy.stats
import matplotlib.pyplot as plt
import numpy as np

x = np.array([0.15, 0.17, 0.20, 0.21, 0.33, 0.36, 0.36, 0.39, 0.40, 0.49, 0.53])
a, b, loc, scale = scipy.stats.beta.fit(x, floc=0, fscale=1)

y = scipy.stats.beta.pdf(np.arange(np.size(x)), a, b, loc=0, scale=np.size(x))
plt.plot(x,y)
plt.show()
print(a, b)

# n_iter = 1000
# params = np.empty((n_iter, 2))
# for i in range(n_iter):    
#     n_samples = 1000
#     hist_dist = scipy.stats.rv_histogram((y2, np.append(np.arange(k), k)))
#     data = hist_dist.rvs(size=n_samples)
#     a, b, c, d = scipy.stats.beta.fit(data, floc=0, fscale=k)

#     params[i, 0] = a
#     params[i, 1] = b
