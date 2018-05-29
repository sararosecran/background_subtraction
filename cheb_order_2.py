#finds best fit of 2d chebyshev polynomial (2nd order)
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import emcee
import corner
from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


#wc model
def model(init_guess, x, y):
	a0, a1, a2, a3, a4, a5, a6, a7, a8 = init_guess
	mod = a0 + a1*y + a2*x + a3*x*y + a4*y**2 + a5*x**2 + a6*x*y**2 + a7*y*x**2 + a8*(x**2)*(y**2)
	return mod

#same function used for fitting to find initial guess of values.
def curve(x, y, a0, a1, a2, a3, a4, a5, a6, a7, a8):
	mod = a0 + a1*y + a2*x + a3*x*y + a4*y**2 + a5*x**2 + a6*x*y**2 + a7*y*x**2 + a8*(x**2)*(y**2)
	return mod

#find chi^2
def chi2_func(init_guess, x, y, pix_val, var):
	mod = model(init_guess, x, y)
	N = len(x)
	chi2 = -((1.0 / (N-1.0)) * np.sum((pix_val - mod)**2 / (var**2)))
	return chi2





#read in data
f = open('fit_data.dat', "r")
line = f.readlines()[2:]
f.close()
x = np.array([])
y = np.array([])
pix_val = np.array([])
var = np.array([])
for i in range(len(line)):
	x = np.append(x, float(line[i].split()[0]))
	y = np.append(y, float(line[i].split()[1]))
	pix_val = np.append(pix_val, float(line[i].split()[2]))
	var = np.append(var, float(line[i].split()[3]))

#Get initial fit
init_guess0 = [0.1, 0.1, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
#X, Y = np.meshgrid(x, y, copy=False)
#Z = X**2 + Y**2 + np.random.rand(*X.shape)*0.01
#X = X.flatten()
#Y = Y.flatten()
#A = np.array([X*0+1, Y, X, X*Y]).T
#B = Z.flatten()

#coeff, r, rank, s = np.linalg.lstsq(A, B)
#print coeff
#popt = [coeff[0], coeff[1], coeff[2], coeff[3], coeff[4]]
#x = np.linspace(R.min(),R.max(),len(R))
#popt, pcov = curve_fit(curve, x, wc, p0=init_guess0)
#perr = np.sqrt(np.diag(pcov))



#chi2 minimization, small number of steps
ndim = len(init_guess0)
nwalkers = 100
pos = [init_guess0 + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] #initial position, near initial guess
sampler = emcee.EnsembleSampler(nwalkers, ndim, chi2_func, args=(x, y, pix_val, var)) #create ensembles of walkers
sampler.run_mcmc(pos, 500) #run the mcmc for 500 steps
#plt.plot(sampler.chain[:,:,1].T, '-', color='k', alpha=0.3)
#plt.show()
samples = sampler.chain[:, 499:, :].reshape((-1, ndim)) #flattens the chain
norm_a0, norm_a1, norm_a2, norm_a3, norm_a4, norm_a5, norm_a6, norm_a7, norm_a8 = map(lambda v: (v[1], v[2]- v[1], v[1]-v[0]), zip(*np.percentile(samples, [16,50,84], axis=0))) # gives best fit parameters




#chi2 minimization, large number of steps
popt = [norm_a0[0], norm_a1[0], norm_a2[0], norm_a3[0], norm_a4[0], norm_a5[0], norm_a6[0], norm_a7[0], norm_a8[0]]
pos = [popt + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] #initial position, near initial guess
sampler = emcee.EnsembleSampler(nwalkers, ndim, chi2_func, args=(x, y, pix_val, var)) #create ensembles of walkers
sampler.run_mcmc(pos, 50000) #run the mcmc for 500 steps
#plt.plot(sampler.chain[:,:,1].T, '-', color='k', alpha=0.3)
#plt.show()
samples = sampler.chain[:, 49000:, :].reshape((-1, ndim)) #flattens the chain
norm_a0, norm_a1, norm_a2, norm_a3, norm_a4, norm_a5, norm_a6, norm_a7, norm_a8 = map(lambda v: (v[1], v[2]- v[1], v[1]-v[0]), zip(*np.percentile(samples, [16,50,84], axis=0))) # gives best fit parameters

print 'med, 1 & 2 sig (a0): ', norm_a0
print 'med, 1 & 2 sig (a1): ', norm_a1
print 'med, 1 & 2 sig (a2): ', norm_a2
print 'med, 1 & 2 sig (a3): ', norm_a3
print 'med, 1 & 2 sig (a4): ', norm_a4
print 'med, 1 & 2 sig (a5): ', norm_a5
print 'med, 1 & 2 sig (a6): ', norm_a6
print 'med, 1 & 2 sig (a7): ', norm_a7
print 'med, 1 & 2 sig (a8): ', norm_a8


#contours
fig = corner.corner(samples, labels=["a0","a1", "a2", "a3", "a4", "a5",], truths=[norm_a0[0],norm_a1[0],norm_a2[0],norm_a3[0]])
plt.title('2nd Order')

plt.show()
