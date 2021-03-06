from graph import Graph
from scipy import stats
from scipy.stats import multivariate_normal
import numpy as np
import itertools
import math
import sys
import os as os
import cPickle as pickle



def conditional_prob_on(PAstate, Ematrix, Wbind, omega):
	
	"""
	The conditional prob is dependent on the biophysical model

	"""
	if isinstance(omega, float):
		omega = np.array([omega])

	omegainv= omega**-1
	num = 0.0
	den = 0.0

	for j in range(0,len(Ematrix)):
		num = num + omegainv[j]*np.exp(-Ematrix[j])*PAstate[j]
		den = den + omegainv[j]*(np.exp(-Ematrix[j])*PAstate[j] + np.exp(-Wbind)*(1-PAstate[j]))
	
	return num / (1 + den)

def conditional_prob_on2coop(PAstate, Ematrix, Wbind,J, omega):

	'''
	Note here we address cooperativity between two proteins. Therefore Ematrix has size 2. Here assuming 
	PAstat[1] (i.e. x4)  and PAstate[0] (i.e. x3) are both activators
	'''
	omegainv = omega**-1
	num = 0.0
	den = 0.0

	cooperation = PAstate[0]*(PAstate[1])*(omegainv[0]*omegainv[1])* np.exp(-Ematrix[0]-Ematrix[1]-J)
	cooperation_un = PAstate[0]*(1-PAstate[1])*omegainv[0]*omegainv[1]*np.exp(-Ematrix[0]-Wbind)+ PAstate[1]*(1-PAstate[0])*omegainv[1]*omegainv[0]*np.exp(-Ematrix[1]-Wbind)
	activator = PAstate[0]*(omegainv[0])* np.exp(-Ematrix[0])+  PAstate[1]*(omegainv[1])* np.exp(-Ematrix[1])
	den = 1.0 + cooperation + cooperation_un + activator
	
	return (cooperation ) / den









def GD(p1on, p2on, P0c1, P0c2, Pnow1, Pnow2, eta, bindingE, bindingW, J, regC, oall, GDsteps):
	""" bindingE is labelled as follows E[0]=E_13, E[1]=E_23, E[2]=E_34, E[3]=E_35
		training_data[i,j]: [i=0] for x1, [i=1] for x2, [i=2] for x3, [i=4] for x3
							j=0,1,....  sample label

		p1on is the concatenation of p1on[csweep1, csweep2]
		p2on is the concatenation of p2on[csweep1, csweep2]
		P0c1 is the concatenation of z1[csweep1, csweep2]
		P0c2 is the concatenation of z2[csweep1, csweep2]
		Pnow1
		Pnow1

 	"""
	n = p1on.shape[0]
	nabla_E = np.zeros(len(bindingE))
	underflowcutoff = 0.001 
	delP3_12_1 = np.zeros((2,2,2), dtype = float)
	delP3_12_2 = np.zeros((2,2,2), dtype = float)
	delP4_3 = np.zeros((2,2), dtype = float)
	delP5_3 = np.zeros((2,2), dtype = float)
	

	f3_13_4 = np.zeros((2,2,2,2)) 
	f3_23_4 = np.zeros((2,2,2,2))
	f3_13_5 = np.zeros((2,2,2,2))
	f3_23_5 = np.zeros((2,2,2,2))
	f4_3 = np.zeros((2,2,2,2))
	f5_3 = np.zeros((2,2,2,2))


	freqall = oall**-1
	

	for tstep in xrange(GDsteps):

		for sampleindex in xrange(n):

			delP3_12_1 = np.zeros((2,2,2), dtype = float)
			delP3_12_2 = np.zeros((2,2,2), dtype = float)
			delP4_3 = np.zeros((2,2), dtype = float)
			delP5_3 = np.zeros((2,2), dtype = float)
			

			f3_13_4 = np.zeros((2,2,2,2)) 
			f3_23_4 = np.zeros((2,2,2,2))
			f3_13_5 = np.zeros((2,2,2,2))
			f3_23_5 = np.zeros((2,2,2,2))
			f4_3 = np.zeros((2,2,2,2))
			f5_3 = np.zeros((2,2,2,2))

			tmp = np.array(list(itertools.product([[0],[1]], repeat = 2)))

			for j in xrange(tmp.shape[0]):
				i1 = tmp[j][0][0] # index for x1
				i2 = tmp[j][1][0] # index for x2

				
				estart = 0
				delP3_12_1[1,i1, i2]=  (-freqall[0]*np.exp(-bindingE[estart])*float(i1)*(1
				+(freqall[0]+freqall[1]-freqall[0]*float(i1)-freqall[1]*float(i2))*np.exp(-bindingW)))/((1
				+freqall[0]*np.exp(-bindingE[estart])*float(i1)
				+freqall[1]*np.exp(-bindingE[estart+1])*float(i2)
				+np.exp(-bindingW)*(freqall[0]+freqall[1]-freqall[0]*float(i1)-freqall[1]*float(i2)))**2)

				delP3_12_2[1,i1, i2]=  (-freqall[1]*np.exp(-bindingE[estart+1])*float(i2)*(1
				+(freqall[0]+freqall[1]-freqall[0]*float(i1)-freqall[1]*float(i2))*np.exp(-bindingW)))/((1
				+freqall[0]*np.exp(-bindingE[estart])*float(i1)
				+freqall[1]*np.exp(-bindingE[estart+1])*float(i2)
				+np.exp(-bindingW)*(freqall[0]+freqall[1]-freqall[0]*float(i1)-freqall[1]*float(i2)))**2)


			for j in xrange(2):

				estart = 2
				delP4_3[1, j] = (-freqall[2]*np.exp(-bindingE[estart])*float(j)*(1
				+(freqall[2]-freqall[2]*float(j))*np.exp(-bindingW)))/((1
				+freqall[2]*np.exp(-bindingE[estart])*float(j)
				+np.exp(-bindingW)*(freqall[2]-freqall[2]*float(j))**2))

				delP5_3[1, j] = (-freqall[2]*np.exp(-bindingE[estart+1])*float(j)*(1
				+(freqall[2]-freqall[2]*float(j))*np.exp(-bindingW)))/((1
				+freqall[2]*np.exp(-bindingE[estart+1])*float(j)
				+np.exp(-bindingW)*(freqall[2]-freqall[2]*float(j))**2))



			delP3_12_1[0,:,:] = 1.0-delP3_12_1[1,:,:]
			delP3_12_2[0,:,:] = 1.0-delP3_12_2[1,:,:]
			delP4_3[0,:] = 1.0 - delP4_3[1,:]
			delP5_3[0,:] = 1.0 - delP5_3[1,:]

			
		

			tmp = np.array(list(itertools.product([[0],[1]], repeat = 4)))

			for i in xrange(tmp.shape[0]):
				i4 = tmp[i][0][0] # index for x4 or x5
				i1 = tmp[i][1][0] # index for x1
				i2 = tmp[i][2][0] # index for x2
				i3 = tmp[i][3][0] # index for x3


				# Note the special structure of P34 and P35
				f3_13_4[i4,i1,i2,i3] = P34[i3,i4]*delP3_12_1[i3,i1,i2]*p1on[sampleindex]*p2on[sampleindex]
				f3_23_4[i4,i1,i2,i3] = P34[i3,i4]*delP3_12_2[i3,i1,i2]*p1on[sampleindex]*p2on[sampleindex]
				f3_13_5[i4,i1,i2,i3] = P35[i3,i4]*delP3_12_1[i3,i1,i2]*p1on[sampleindex]*p2on[sampleindex]
				f3_23_5[i4,i1,i2,i3] = P35[i3,i4]*delP3_12_2[i3,i1,i2]*p1on[sampleindex]*p2on[sampleindex]
				f4_3[i4,i1,i2,i3] = delP4_3[i4,i3]*P123[i3,i1,i2]*p1on[sampleindex]*p2on[sampleindex]
				f5_3[i4,i1,i2,i3] = delP5_3[i4,i3]*P123[i3,i1,i2]*p1on[sampleindex]*p2on[sampleindex]


			for i in xrange(3):
				f3_13_4 = np.sum(f3_13_4, axis = 1)
				f3_23_4 = np.sum(f3_23_4, axis = 1)
				f3_13_5 = np.sum(f3_13_5, axis = 1)
				f3_23_5 = np.sum(f3_23_5, axis = 1)
				f4_3 = np.sum(f4_3, axis = 1)
				f5_3 = np.sum(f5_3, axis = 1)


			
				

			
			nabla_E[0] = nabla_E[0] + 2*(Pnow1[sampleindex]-P0c1[sampleindex])*f3_13_4[1]+  2*(Pnow2[sampleindex]-P0c2[sampleindex])*f3_13_5[1] -2*regC*nabla_E[0]

			nabla_E[1] = nabla_E[1] + 2*(Pnow1[sampleindex]-P0c1[sampleindex])*f3_23_4[1]+  2*(Pnow2[sampleindex]-P0c2[sampleindex])*f3_23_5[1] -2*regC*nabla_E[1]
				

			nabla_E[2] = nabla_E[2] + 2*(Pnow1[sampleindex]-P0c1[sampleindex])*f4_3[1]-2*regC*nabla_E[2]
			nabla_E[3] = nabla_E[3] + 2*(Pnow2[sampleindex]-P0c2[sampleindex])*f5_3[1]-2*regC*nabla_E[3]
			
			


		nabla_E = nabla_E / float(n)
		#nabla_E = nabla_E/np.linalg.norm(nabla_E)
		bindingE = bindingE - eta*nabla_E

	return bindingE


def SGD(self, training_data, epochs, mini_batch_size, eta,
			test_data=None):
	"""Train the neural network using mini-batch stochastic
	gradient descent.  The ``training_data`` is a list of tuples
	``(x, y)`` representing the training inputs and the desired
	outputs.  The other non-optional parameters are
	self-explanatory.  If ``test_data`` is provided then the
	network will be evaluated against the test data after each
	epoch, and partial progress printed out.  This is useful for
	tracking progress, but slows things down substantially."""
	if test_data: n_test = len(test_data)
	n = len(training_data)
	for j in xrange(epochs):
		random.shuffle(training_data)
		mini_batches = [
			training_data[k:k+mini_batch_size]
			for k in xrange(0, n, mini_batch_size)]
		for mini_batch in mini_batches:
			[W, allbias] = self.update_mini_batch(mini_batch, eta)
		if test_data:
			print "Epoch {0}: {1} / {2}".format(
				j, self.evaluate(test_data), n_test)
		else:
			print "Epoch {0} complete".format(j)
	return [W, allbias]



def update_mini_batch(self, mini_batch, eta):

	"""Update the network's weights and biases by applying
	gradient descent using backpropagation to a single mini batch.
	The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
	is the learning rate."""
	nabla_b = [np.zeros(b.shape) for b in self.biases]
	nabla_w = [np.zeros(w.shape) for w in self.weights]
	for x, y in mini_batch:
		delta_nabla_b, delta_nabla_w = self.backprop(x, y)
		nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
		nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
	self.weights = [w-(eta/len(mini_batch))*nw 
					for w, nw in zip(self.weights, nabla_w)]
	self.biases = [b-(eta/len(mini_batch))*nb 
					for b, nb in zip(self.biases, nabla_b)]

	return [self.weights, self.biases]


def create_name(learning_rate, typestr1, typestr2, reg_level, runs, tgtype, gdstep, cnum):
	
	str_r = 'DBL2_212' + typestr1 + '_' + typestr2 + '_ETA' + str(learning_rate) +'_reg_levelC' + str(reg_level) + '_GDsteps'+ repr(gdstep) + '_Nruns' +str(runs) + tgtype + '_'+ repr(cnum) +'_NEWICv3reversed'

	return str_r



# here specify c1, c2 sweep and the sweepwindow size
#sweep1start = int(sys.argv[1])  # argument i in the batch file
#sweep2start = int(sys.argv[2])  # argument i in the batch file
sweep1start = 0
sweep2start = 0



# Here are some parameters for learning

#learning_eta = float(sys.argv[1])  # argument i in the batch file
#regularization = float(sys.argv[2]) # argument j in the batch file
learning_eta = 5.0 
regularization = 1.0  #float(sys.argv[2]) # argument j in the batch file
Nruns = 50 # num of runs (BP+GD)
NGD = 50 # Gradient descent steps
c1 = 0.5  # threshold for input 1
c2 = 0.5  # threshold for input 2
targetc1 = 0.5 # threshold for target in dim 1
targetc2 = 0.5 # threshold for target in dim 2
beta1 = 10
beta2 = 10
Samplesize = 1 # don't need any samples, set 1 for concatenate purpose
targetype = 'discrete' # 'BP' or 'discrete' or 'normal'
typestr1 = 'IM' # RM, AD, BL 
typestr2 = 'BL'




# File I/O
dir_name='./'
case_num = 2 # for data manipulation purpose


# First specify the initial binding affinities

Nproteins = 5
Ebind = np.zeros([Nruns, 4])
Ebind[0,:] = np.array([-2.0,   -2.0,   1.0, -2.0])  #  for RM
wrong_binding = 2.0  # This can be incorporated into learning
cooperativityJ = -5.0  

# here for chemical potential mu_i = - log omega_i
# for uniform case, set omega345= Nproteins

omega345 =  Nproteins # 4
omegarest = ((1-3*omega345**-1)/(Nproteins-3))**-1

omegavec = np.zeros(Nproteins) + float(Nproteins)
mu = - np.log(omegavec) # chemical potential


# input space spec
csweep = np.linspace(0, 1, num = 11)

# marginals useful later

P4onall = np.zeros([len(Ebind)-1,len(csweep),len(csweep)])
P5onall = np.zeros([len(Ebind)-1,len(csweep),len(csweep)])


marginal4 = np.zeros([2,len(csweep),len(csweep)]) # for 'BP', [0] for off and [1] for on 
marginal5 = np.zeros([2,len(csweep),len(csweep)]) # for 'BP'


# p(x1=1| {c} ) and p(x2=1| {c})
input1 = (1+np.exp(-beta1*(csweep-c1)))**-1 # activation prob
input2 = (1+np.exp(-beta2*(csweep-c2)))**-1





## below specify target ##

clen = len(csweep)
cmin = np.min(csweep)
cmax = np.max(csweep)
cstep = csweep[1] - csweep[0]
dx, dy = cstep, cstep

yd, xd = np.mgrid[slice(cmin, cmax + dy, dy),
				slice(cmin, cmax + dx, dx)]


zd1= np.zeros([len(xd), len(yd)], dtype= float)
zd2= np.zeros([len(xd), len(yd)], dtype= float)
mid = int(np.floor(clen/2))


if typestr1 == 'RM':
	xgrid= np.linspace(0,0.5, num=len(xd))

	for i in xrange(len(xd)):
		zd1[:,i] = np.linspace(xgrid[i], 1.0, num = len(xd))
	zd1[0:mid+1, 0:mid+1] = 0.001
elif typestr1 == 'AD':
	xgrid= np.linspace(0,1.0, num=len(xd))

	for i in xrange(len(xd)):
		zd1[:,i] = np.linspace(xgrid[i], 1.0, num = len(xd))
	zd1[0:mid+1, 0:mid+1] = 0.001

elif typestr1 == 'IM':
	xbegin= np.linspace(0,1.0, num=len(xd))
	xend = np.linspace(1.0,0.5, num = len(xd))

	for i in xrange(len(xd)):
		zd1[:,i] = np.linspace(xbegin[i], xend[i], num = len(xd))
	zd1[0:mid+1, 0:mid+1] = 0.001
	zd1 = zd1*0.9



elif typestr1 == 'BL':
	xbegin= np.linspace(0,0.5, num=len(xd))
	xend = np.linspace(0.5,1.0, num = len(xd))

	for i in xrange(len(xd)):
		zd1[:,i] = np.linspace(xbegin[i], xend[i], num = len(xd))
	zd1[0:mid+1, 0:mid+1] = 0.001
	zd1 = zd1*0.9


# Below for target 2

if typestr2 == 'RM':
	xgrid= np.linspace(0,0.5, num=len(xd))

	for i in xrange(len(xd)):
		zd2[:,i] = np.linspace(xgrid[i], 1.0, num = len(xd))
	zd2[0:mid+1, 0:mid+1] = 0.001
elif typestr2 == 'AD':
	xgrid= np.linspace(0,1.0, num=len(xd))

	for i in xrange(len(xd)):
		zd2[:,i] = np.linspace(xgrid[i], 1.0, num = len(xd))
	zd2[0:mid+1, 0:mid+1] = 0.001

elif typestr2 == 'IM':
	xbegin= np.linspace(0,1.0, num=len(xd))
	xend = np.linspace(1.0,0.5, num = len(xd))

	for i in xrange(len(xd)):
		zd2[:,i] = np.linspace(xbegin[i], xend[i], num = len(xd))
	zd2[0:mid+1, 0:mid+1] = 0.001
	zd2 = zd2*0.9
elif typestr2 == 'BL':
	xbegin= np.linspace(0,0.5, num=len(xd))
	xend = np.linspace(0.5,1.0, num = len(xd))

	for i in xrange(len(xd)):
		zd2[:,i] = np.linspace(xbegin[i], xend[i], num = len(xd))
	zd2[0:mid+1, 0:mid+1] = 0.001
	zd2 = zd2*0.9





# Start main loop

for progindex in xrange(len(Ebind)-1):

	targetP0all1=0.0
	targetP0all2=0.0
	p1onall = 0.0
	p2onall = 0.0
	pnow4all = 0.0
	pnow5all=0.0


	for sweep1 in range(sweep1start, len(csweep)): # fix p(x1(c1)), loop over in the batch file

		for sweep2 in range(sweep2start,len(csweep)):  # fix p(x2(c2)) or range(sweep2start, len(csweep))

			G = Graph()
			x1 = G.addVarNode('x1',2)
			x2 = G.addVarNode('x2',2)
			x3 = G.addVarNode('x3',2)
			x4 = G.addVarNode('x4',2)
			x5 = G.addVarNode('x5',2)
			


			# Pa3 encodes (x1,x2)=(0,0), (0,1), (1,0), (1,1)
			Pa3 = np.array(list(itertools.product([[0],[1]], repeat = 2))) # repeat= |pa(x3)| 

			# Pa4, Pa5 
			Pa4 = np.array([[0],[1]])
			Pa5 = np.array([[0],[1]])


			# Below start to specify factor nodes
			# p(x1)=[p(x1=0), p(x1=1)]
			P1 = np.array([[1-input1[sweep1]],[input1[sweep1]]])
			G.addFacNode(P1, x1)

			# p(x2)=[p(x2=0), p(x2=1)]
			P2 = np.array([[1-input2[sweep2]],[input2[sweep2]]])
			G.addFacNode(P2, x2)


			# Set up energy 
			Ebindinit = Ebind[progindex,:]
			E3 = Ebindinit[0:2] # i.e. E12 E23, ngb of x3
			E4 = np.array([Ebindinit[2]]) # i.e. E34, ngb of x4
			E5 = np.array([Ebindinit[3]]) # i.e. E35, ngb of x5

		

			# note the syntax: conditional_prob_on(PAstate, Ematrix, Wbind, omega):

			# below for factor node P123
			P123 = np.zeros((2,2,2)) # [ p(x3=0|x1 x2), p(x3=1|x1 x2)]
			P123[:,0,0]=[1.0-conditional_prob_on(Pa3[0], E3, wrong_binding, omegavec[0:2]), conditional_prob_on(Pa3[0], E3, wrong_binding, omegavec[0:2])]
			P123[:,0,1]=[1.0-conditional_prob_on(Pa3[1], E3, wrong_binding, omegavec[0:2]), conditional_prob_on(Pa3[1], E3, wrong_binding, omegavec[0:2])]
			P123[:,1,0]=[1.0-conditional_prob_on(Pa3[2], E3, wrong_binding, omegavec[0:2]), conditional_prob_on(Pa3[2], E3, wrong_binding, omegavec[0:2])]
			P123[:,1,1]=[1.0-conditional_prob_on(Pa3[3], E3, wrong_binding, omegavec[0:2]), conditional_prob_on(Pa3[3], E3, wrong_binding, omegavec[0:2])]


			G.addFacNode(P123,x3,x1,x2)

			#P34=p(x4|x3) [[x4 OFF when x3=0, x4 ON when x3=0], [x4 OFF when x3=1, x4 ON when x3=1]]
			P34 = np.array([[1-conditional_prob_on(Pa4[0], E4, wrong_binding, omegavec[2]), conditional_prob_on(Pa4[0], E4, wrong_binding, omegavec[2])]\
			,[1-conditional_prob_on(Pa4[1], E4, wrong_binding, omegavec[2]), conditional_prob_on(Pa4[1], E4, wrong_binding, omegavec[2])]])

			G.addFacNode(P34, x3, x4) 


			#P35=p(x5|x3) [[x5 OFF when x3=0, x5 ON when x3=0], [x5 OFF when x3=1, x5 ON when x3=1]]
			P35 = np.array([[1-conditional_prob_on(Pa5[0], E5, wrong_binding, omegavec[2]), conditional_prob_on(Pa5[0], E5, wrong_binding, omegavec[2])]\
			,[1-conditional_prob_on(Pa5[1], E5, wrong_binding, omegavec[2]), conditional_prob_on(Pa5[1], E5, wrong_binding, omegavec[2])]])

			G.addFacNode(P35, x3, x5)




			# This network is small enough
			brute = G.bruteForce()
			distx4 = G.marginalizeBrute(brute, 'x4')
			distx5 = G.marginalizeBrute(brute, 'x5')

			''' below us BP
			marg = G.marginals()
			distx4 = marg['x4']
			distx5 = marg['x5']

			'''


			
			

			marginal4[0,sweep1,sweep2]= float(distx4[0]) # x4 off
			marginal4[1,sweep1,sweep2]= float(distx4[1]) # x4 on

			marginal5[0,sweep1,sweep2]= float(distx5[0]) # x5 off
			marginal5[1,sweep1,sweep2]= float(distx5[1]) # x5 on

			P4onall[progindex,sweep1,sweep2]= float(distx4[1])  # x7 on for all \vec{c}
			P5onall[progindex,sweep1,sweep2]= float(distx5[1])  # x7 on for all \vec{c}

			

			# Collect/concatenate the inputs and outputs	
			targetP0all1 = np.append(targetP0all1, zd1[sweep1,sweep2])
			targetP0all2 = np.append(targetP0all2, zd2[sweep1,sweep2])
			p1onall = np.append(p1onall, input1[sweep1])
			p2onall = np.append(p2onall, input2[sweep2])
			pnow4all = np.append(pnow4all, float(distx4[1]))
			pnow5all = np.append(pnow5all, float(distx5[1]))


			

			#print 'Complete  the first ' +repr(progindex+1)+ ' run(s) out of ' + repr(len(Ebind)-1)+ ' runs' 


			G.reset()

	

	targetP0all1 = np.delete(targetP0all1, 0)
	targetP0all2 = np.delete(targetP0all2, 0)
	p1onall = np.delete(p1onall, 0)
	p2onall = np.delete(p2onall, 0)
	pnow4all = np.delete(pnow4all, 0)
	pnow5all = np.delete(pnow5all, 0)

	'''
	print targetP0all1.shape
	print targetP0all2.shape
	print p1onall.shape
	print p2onall.shape
	print pnow4all.shape
	print pnow5all.shape
	'''

	
	#p1on, p2on, P0c1, P0c2, Pnow1, Pnow2, eta, bindingE, bindingW, J, regC, oall, GDsteps
	gdtmp = GD(p1onall, p2onall, targetP0all1, targetP0all2, pnow4all, pnow5all, learning_eta, Ebindinit, wrong_binding, cooperativityJ, regularization, omegavec, NGD)
	Ebind[progindex+1,:]  = gdtmp
	#print gdtmp
	


# def create_name(learning_rate, reg_level, runs, tgtype, gdstep, cnum):	
print file_name
file_name = create_name(learning_eta, typestr1, typestr2, regularization, Nruns, targetype, NGD, case_num) 
name = os.path.join(dir_name, file_name)
pickle.dump( {'BindingEnergy':Ebind, 'x4marginal':P4onall, 'x5marginal':P5onall, 'x4marginalBP': marginal4,'x5marginalBP': marginal5, 'zd1': zd1, 'zd2': zd2, 'chemicalpotential': mu}, open(name, 'wb'))


