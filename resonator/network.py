# -*- coding: utf-8 -*-
"""
/***********************************************************************************************************\

 This NEURON + Python script associated with paper:                                                        
 Ruben A. Tikidji-Hamburyan, Joan José Martínez, John A. White, Carmen C. Canavier                         
    Resonant interneurons can increase robustness of gamma oscillations                                    
                                                                                                           
 Network of 300 Izhikevich's resonators connected by double-exponential synapses is modeled                
 All parameters may be set up by command line arguments listed AFTER script name. The command should be:   
  nrngui -nogui -python network.py [PARAMETERS]                                                            
 A list of avalible parameters can be printed out by command:                                              
  nrngui -nogui -python network.py --help
                                                                                                           
 To replicate Figure 4 run:                                                                                
  nrngui -nogui -python network.py -Iapp=0.15e-5 -gsyn=3e-7 -nstart='(900.,0.21e-5,1000)' -Istd=0 -gui -view 
                                                                                                           
 To replicate points from Figure 8A run:                                                                   
  nrngui -nogui -python network.py -Iapp=0.15e-5 -gsyn=3e-7 -Istd=Xe-5 -gui=off
    X is std dev in nA                                                                                     
                                                                                                           
 To replicate points from Figure 8B run:                                                                   
  nrngui -nogui -python network.py -Iapp=0.2e-5,X -gsyn=3e-7 -Istd=0.0 -gui=off
    X is CV multiple on 0.2e-5, mean value                                                                 
                                                                                                           
 To replicate points from Figure 8C run:                                                                   
  nrngui -nogui -python network.py -Iapp=0.15e-5 -gsyn=3e-7 -Istd=0.0 -gsynscale=1,X -gui=off
    X is CV of total synaptic conductance distributed within population.                                   
                                                                                                           
 To replicate points from Figure 8D run:                                                                   
  nrngui -nogui -python network.py -Iapp=0.15e-5 -gsyn=3e-7 -Istd=0.0 -delay=3,X -gui=off
    X is delay standard deviation in ms                                                                    
                                                                                                           
 === NOTES FOR ALL Figures 8 simulations ===                                                               
 ===  1. the results of simulations is saved into network.csv file.                                        
 ===  2. to see traces and rasterplot remove -gui=off and add -gui -view for any command above              
 ===  3. when SPC felt below 0.15 the network is very sensitive to initial conditions and result may be    
         slightly different from the paper, specifically for large Iapp CVs.                                   
                                                                                                           
 To replicate points on Figure 7:
  nrngui -nogui -python network.py -Iapp=0.15e-5 -gsyn=XXe-5 -tau2=YY -Istd=0 -gui -view 
    XX and YY is gsyn in nS and tau_fall in ms                                                             
 
 To replicate Figure 9C3:                                                                                                          
  nrngui -nogui -python network.py -Istd=0 -Iapp=0.2e-5 -gsyn=0.005e-5 -delay=1 -view -gui -tstop=5000 -F=1,0.1125 -c=299 -Vinit=-65 -sort=F

 To replicate Figure 10C1:
  nrngui -nogui -python network.py -Iapp=0.15e-5 -gsyn=0.05e-5 -Istd=26.1e-7 -Vinit=-65 -c=299 -gui -view -tstop=5000
 
 To replicate Figure 10C2:
  nrngui -nogui -python network.py -Iapp=0.15e-5 -gsyn=0.03e-5 -Istd=3.915e-5 -Vinit=-65 -c=299 -gui -view -tstop=5000

 To replicate Figure 10C3:
  nrngui -nogui -python network.py -Iapp=0.15e-5 -gsyn=0.03e-5 -Istd=10.455e-5 -Vinit=-65 -c=299 -gui -view -tstop=5000

 Copyright: Ruben Tikidji-Hamburyan <rth@nisms.krinc.ru> Feb.2013 - Aug.2015                               

\************************************************************************************************************/
"""

import numpy as np
import sys,os,csv,threading
import random as rnd
from neuron import h

###### Paramters:
ncell		= 300			# number of neurons in population
ncon		= 40#(20,60)#40	# number of input connections per neuron
#ntype		= "TypeI"		# use: 'TypeI' for Class 1 neuron
#ntype		= "RS-moca"			# 	'RS' - resonator
ntype		= "RS"			# 	'RS' - resonator
#ntype		= "FS"			# 	'FS' - fast spiking
methods		= {
	'R2'		: True,
	'maxFreq'	: 200.0,		# max frequency
	'peakDetec' : True,			# Turn on/off peak detector
	'gkernel'	: (10.0,50.0),	# Kernel and size (5,25),#
	"netFFT"	: False,		# Turn on/off network FFT
	"nrnFFT"	: False,		# Turn on/off neuron FFT
	'netISI'	: 30001,		# max net ISI
	'nrnISI'	: 30001,		# max neuron ISI
	'cliptrn'	: False,#1000,#False alse,#500,#False,	# Clip transience for first n ms or False
	'traceView'	: False,		# Trace and nulcline view
	'traceWidth': 55.0,			#
	'tracetail'	: 'conductance',#'total current',#'firing rate',#'conductance', #'current' 'total current',
	'save'		: False,		# To save all data from model. Don't turn on!
	'patview'	: True,			# Turn on/off Pattern vew
	'gui'		: True,
	'git'		: False,		# Turn on/off git core (Never turn-on at the head node!!!!!!!)
	'gif'		: False,		# Generate gif instead pop up on a screan.
	'fftkernel'	: 5.0,
	'isikernel'	: 5.0,
	'corefunc'	: (4,8,64),
	'coreindex'	: False,		# Turn on/off Core indexing
	'corelog'	: 'network',
	'noexit'	: False,
	'FPcurve'	: False,
	'GPcurve'	: False,
	'IGcurve'	: False,
	'Connectom'	: False,
	'G-Spikerate'	: False,
	'F-Spikerate'	: False,
	'Gtot-dist'	: False,
	'Gtot-rec'	: False,#True,	#record all gtotal in neurons
	'Gtot-stat'	: True, #False, #record gtot statistic
	'sycleprop'	: False,#True,
	'external'	: False,
	## Min
	'extprop'	: 0.5,				# Calculate probability to fire after external input

	'timestep'	: 0.01,#0.005,
	'sortbysk'	: False, #'ST',#'F',#'GT',#'NC',#'GT','G',#'I', #'F',#'I',#'F',#False,			#Do not use
	'taunorm'	: False,#True,#False,#True,
	'nstart'	: False, #
	'cliprst'	: False,#10,#False,#20,
	'TaS'		: True,
	'lastspktrg': True,
	'fullrast'	: True,
	'bias'		: False, #0.0000015
	'gtot-dist'	: 'NORM',#'LOGN', #LOGN - lognormal, 'NORM' - normal
	'gsyn-dist' : 'LOGN',#'NORM',#'LOGN', #same
	'initset'	: None,
	'cycling'	: False, #4,False
	'popfr'		: False,		#calculate population firing rate
}

#Neuron paramters:
if ntype == "RS":
	#RS neuron:
	#a,b,c,d,U,V	= 0.1,0.26,-65.0,-1.0,-16.083619212816,-61.86007190636312
#	a,b,c,d,U,V	= 0.1,0.26,-65.0,-1.0,-13.96214168703548,-65.81070843517298# Seady Satate at Iapp=0.15
#	a,b,c,d,U,V	= 0.1,0.26,-65.0,-1.0,(-13.96214168703548,2),(-65.81070843517298,5)
#	a,b,c,d,U,V	= 0.1,0.26,-65.0,-1.0,(-16.083619212816,5),(-61.86007190636312,20)
	a,b,c,d,U,V	= 0.1,0.26,-65.0,-1.0,(-15.083619212816,5),(-51.86007190636312,20)
	F 			= 1.#1.25 #0.25
	Iapp		= 0.0000015	# Iapp. it's equal to 0.1955 in XPP or Matlab. 0.000001955 in the rheobase for RS
#	Iapp		= (0.000001955,0.0000008)	#Don't use it. It's fake!
	Istdev		= 0.000001	# SD for 0.01ms step. it is equal to 0.09 in XPP
#	Istdev		= 0.0000008
#	Istdev		= 0.00000125
#	Istdev		= 0.000008
##Exp2Syn
	weight		= (3e-7, 0.0)	# Synaptic conductance. It is equal to 0.05 in XPP and Matlab
##e2ssn
#	weight		= (1e-7*7.62224658625,0.0) #e2ssn
#	weight		= (3e-7*7.62224658625,0.0) #e2ssn
#	weight		= (5e-7*7.62224658625,0.0) #e2ssn
	delay		= (0.1,0)			# delay in a system
#	Iapp		= 0.00002	# Iapp. it's equal to 0.1955 in XPP or Matlab. 0.000001955 in the rheobase for RS
#	Istdev		= 0.000005		# SD for 0.01ms step. it is equal to 0.2 in XPP
#	delay		= (5.0,0)
	gsynscale	= 1.0

elif ntype == "RS-moca":
	#RS neuron:
	a,b,c,d,U,V	= 0.1,0.26,-65.0,2.0,-16.083619212816,-61.86007190636312
	#a,b,c,d,U,V	= 0.1,0.26,-65.0,-1.0,(-16.083619212816,5),(-61.86007190636312,20)
	F 			= 1.#1.25 #0.25
	Iapp		= 0.0000015	# Iapp. it's equal to 0.1955 in XPP or Matlab. 0.000001955 in the rheobase for RS
#	Iapp		= (0.000001955,0.0000008)	#Don't use it. It's fake!
	Istdev		= 0.00000#1	# SD for 0.01ms step. it is equal to 0.09 in XPP
#	Istdev		= 0.0000008
#	Istdev		= 0.00000125
#	Istdev		= 0.000008
##Exp2Syn
	weight		= (3e-7, 0.0)	# Synaptic conductance. It is equal to 0.05 in XPP and Matlab
##e2ssn
#	weight		= (1e-7*7.62224658625,0.0) #e2ssn
#	weight		= (3e-7*7.62224658625,0.0) #e2ssn
#	weight		= (5e-7*7.62224658625,0.0) #e2ssn
	delay		= (0.1,0)			# delay in a system
#	Iapp		= 0.00002	# Iapp. it's equal to 0.1955 in XPP or Matlab. 0.000001955 in the rheobase for RS
#	Istdev		= 0.000005		# SD for 0.01ms step. it is equal to 0.2 in XPP
#	delay		= (5.0,0)
	gsynscale	= 1.0

elif ntype == "FS":
	#FS neuron:
	a,b,c,d,U,V	= 0.02,0.2,-65.0,2.0,(-12.836,2),(-64.183,10)
#	a,b,c,d,U,V	= 0.02,0.2,-65.0,2.0,-12.836,-64.183
#	a,b,c,d,U,V	= 0.02,0.2,-65.0,2.0,-13.836,-12.183
	F 			= 1.#1.25 #0.25
	Iapp		= 0.0000015	# Iapp. it's equal to 0.1955 in XPP or Matlab. 0.000001955 in the rheobase for RS
#	Iapp		= (0.000001955,0.0000008)	#Don't use it. It's fake!
	Istdev		= 0.00000#1	# SD for 0.01ms step. it is equal to 0.09 in XPP
#	Istdev		= 0.0000008
#	Istdev		= 0.00000125
#	Istdev		= 0.000008
##Exp2Syn
	weight		= (3e-7, 0.0)	# Synaptic conductance. It is equal to 0.05 in XPP and Matlab
##e2ssn
#	weight		= (1e-7*7.62224658625,0.0) #e2ssn
#	weight		= (3e-7*7.62224658625,0.0) #e2ssn
#	weight		= (5e-7*7.62224658625,0.0) #e2ssn
	delay		= (0.1,0)			# delay in a system
#	Iapp		= 0.00002	# Iapp. it's equal to 0.1955 in XPP or Matlab. 0.000001955 in the rheobase for RS
#	Istdev		= 0.000005		# SD for 0.01ms step. it is equal to 0.2 in XPP
#	delay		= (5.0,0)
	gsynscale	= 1.0

elif ntype == "TypeI":
	#Type I neuron:
	a,b,c,d,U,V	= 0.02,-0.1,-55,6,6.3908,-63.9
	F = 1.0
	Iapp		= 0.00010		# Iapp. it's equal to 10 in XPP or Matlab# 0.00022562 in the rheobase for TypeI
	Istdev		= 0.00019		# SD for 0.01ms step. it is equal to 19 in XPP
##Exp2Syn
#	weight		= (2.5e-5,0.0)	# Synaptic conductance. It is equal to 2.5 in XPP and Matlab
##e3ssn
	weight		= (2.5e-5*7.62224658625,0.0)	# Synaptic conductance. It is equal to 2.5 in XPP and Matlab
#	delay		= (7.5,0)		# delay in a system
	delay		= (0.1,0)		# delay in a system
#	Iapp		= 0.00032		# Iapp. it's equal to 10 in XPP or Matlab# 0.00022562 in the rheobase for TypeI
#	Istdev		= 0.00005		# SD for 0.01ms step. it is equal to 19 in XPP

#Synaptic paramters:
ST1,ST2,SE	= 2.0,5.0,-70.0
#ST1,ST2,SE	= 2.0,10.0,-70.0
synscaler	= None

#Simulation paramters:
tstop		= 10001			#10000
tvl,tvr		= 0, 1500 #1000 #1000 # 2000
trustbrak	= 5
nsig		= 3
unlock		= 12 #24, 64, 78, 9959 12(?) ###12 - doesn't work!!! try somwthing different.......
#####

class neuron:
	def __init__(self):
		self.soma = h.Section()
		self.soma.L = 1.
		self.soma.diam=1/np.pi
		self.soma.nseg=1
		self.soma.cm=1
		self.izh	= h.izhcur(0.5, sec=self.soma)
		self.innp	= h.InNp(0.5, sec=self.soma)
		self.rnd	= h.Random(np.random.randint(0,32562))
		self.innp.noiseFromRandom(self.rnd)
		self.innp.dur	= h.tstop
		self.innp.delay	= 0
		self.innp.per	= 0.1
		self.innp.mean	= 0.0
		self.innp.stdev	= 0.0
		self.syn	= h.Exp2Syn(0.5, sec=self.soma)
		#self.syn	= h.e2ssn(0.5, sec=self.soma)
		self.syn.e		= -75.0
		self.syn.tau1	= 2.0
		self.syn.tau2	= 10.0
		######## Recorders ##########
		self.spks	= h.Vector()
		self.sptr	= h.APCount(.5, sec=self.soma)
		self.sptr.thresh = 25
		self.sptr.record(self.spks)
		if methods['gui']:
			self.inoise	= h.Vector()
			self.inoise.record(self.innp._ref_i)
			self.volt	= h.Vector()
			self.volt.record(self.soma(0.5)._ref_v)
			self.uolt	= h.Vector()
			self.uolt.record(self.izh._ref_u)
			self.isyn	= h.Vector()
			self.isyn.record(self.syn._ref_i)
			self.gsyn	= h.Vector()
			self.gsyn.record(self.syn._ref_g)
		######## Registrations ###### 
		self.gsynscale	= 0.0
		self.concnt		= 0.0
		self.gtotal		= 0.0
		self.tsynscale	= 1.0
	def setparams(self,a = None, b = None, c = None, d = None, e = None, f = None, g = None, V=None, U = None, F = None, Iapp = None, Insd = None, SynE = None, SynT1 = None, SynT2 = None):
		if a != None : self.izh.a = a
		if b != None : self.izh.b = b
		if c != None : self.izh.c = c
		if d != None : self.izh.d = d
		if e != None : self.izh.e = e
		if f != None : self.izh.f = f
		if g != None : self.izh.g = g
		if V != None : self.soma(0.5).v = V
		if U != None : self.izh.uinit = U
		if F != None : self.izh.F = F
		if methods['bias']: self.izh.I = methods['bias']
		########
		if Iapp != None : self.innp.mean  = Iapp
		if Insd != None : self.innp.stdev = Insd
		self.innp.dur = h.tstop
		self.innp.delay = 0
		self.innp.per = 0.1
		########
		if SynE != None:  self.syn.e	= SynE
		if SynT1 != None: self.syn.tau1	= SynT1
		if SynT2 != None: self.syn.tau2	= SynT2
	def addnoise(self,Iapp=0.,Insd=0.,delay=0.,dur=0.,per=0.1):
		self.andnoise = h.InNp(0.5, sec=self.soma)
		self.andrnd	= h.Random(np.random.randint(0,32562))
		self.andnoise.noiseFromRandom(self.andrnd)
		self.andnoise.dur	= h.tstop
		self.andnoise.mean  = Iapp
		self.andnoise.stdev = Insd
		self.andnoise.delay	= delay
		self.andnoise.per	= per
		self.andnoise.dur	= dur

#class symulation:
	#def __init___(self,params):
		#if params.get("a",False):

def onclick1(event):
	if not hasattr(onclick1,"aix"):
		aix=zooly.add_subplot(111)
	onclick1.et = event.xdata
	
	### BUG
	onclick1.tl, onclick1.tr = onclick1.et-methods['traceWidth'], onclick1.et+methods['traceWidth']
	onclick1.idx, = np.where( (t > onclick1.tl) * (t < onclick1.tr))
	
	if not hasattr(onclick1,"marks"):
		onclick1.marks = []
		onclick1.marks.append( p.plot([onclick1.tl,onclick1.tl],[-80,30],"r--",lw=2)[0] )
		onclick1.marks.append( p.plot([onclick1.tr,onclick1.tr],[-80,30],"r--",lw=2)[0] )
	else:
		onclick1.marks[0].set_xdata([onclick1.tl,onclick1.tl])
		onclick1.marks[1].set_xdata([onclick1.tr,onclick1.tr])
	
	if not hasattr(onclick1,"lines"):
		onclick1.lines = []
		for n in neurons:
			volt = np.array(n.volt)
			onclick1.lines.append(aix.plot(t[onclick1.idx],volt[onclick1.idx])[0])
	else:
		vmin,vmax = 1000,-1000
		for ind,n in map(None,xrange(ncell),neurons):
			volt = np.array(n.volt)
			if vmin > volt[onclick1.idx].min():vmin = volt[onclick1.idx].min()
			if vmax < volt[onclick1.idx].max():vmax = volt[onclick1.idx].max()
			onclick1.lines[ind].set_xdata(t[onclick1.idx])
			onclick1.lines[ind].set_ydata(volt[onclick1.idx])
			onclick1.lines[ind].set_linewidth(1)
			onclick1.lines[ind].set_ls("-")

		aix.set_xlim(onclick1.tl,onclick1.tr)
		#print vmin,"---",vmax
		aix.set_ylim(vmin,vmax)
	if hasattr(zoolyclickevent,"lines"):
		del zoolyclickevent.lines
	moddy.canvas.draw()
	zooly.canvas.draw()
	mainfig.canvas.draw()

def getnulls(vmin,vmax,gsyn,inoise,ibias):
	vx=np.linspace(vmin,vmax,200)
	u0=b*vx
#	v0 =0.04*vx*vx+5.*vx+140.-(ibias+gsyn*(vx-SE))*100000.
	v0 =0.04*vx*vx+5.*vx+140.-ibias*100000.
	v0n=0.04*vx*vx+5.*vx+140.-(inoise+gsyn*(vx-SE))*100000.
	#print "DB>\n vmin=%g, vmax=%g\n b=%g\n u0min=%g, u0max=%g\n<DB"%(vmin,vmax,b,u0[0],u0[-1])
	#print "gsyn=%g, biase Iapp=%g, total Iapp = %g"%(gsyn*100000.,ibias*100000.,inoise*100000.)
	return vx,u0,v0,v0n
	
def numsptk(postidx,idxrange):
	prespikes = np.array([])
	trange=t[idxrange]
	sptk = np.zeros(trange.size)
	for nidx in OUTList[postidx]:
		sptime = np.array(neurons[nidx].spks)
		sptime = sptime[ np.where( (sptime > trange[0]) * (sptime < trange[-1]) ) ]
		prespikes = np.append(prespikes,sptime)
	
	prespikes = np.sort(prespikes)
	#print prespikes
	accumulator = 0
	for tm in trange:
		mp = np.where(prespikes < tm)[0]
		sptk[np.where( trange == tm )] = mp.size
	return sptk
	
def getprespikes(postidx,tl,tr):
	postspk = []
	for nidx in OUTList[postidx]:
		#DB>>
		#print nidx,":",
		#<<DB
		for nspk in neurons[nidx].spks[ np.where( (neurons[nidx].spks >= tl)*(neurons[nidx].spks < tr) ) ]:
			#DB>>
			#print nspk,
			#<<DB
			postspk.append([nspk,nidx] )
		#DB>>
		#print 
		#<<DB
		
	return np.array( postspk )
	
def zoolyclickevent(event):
	zoolyclickevent.spikesymbol = "."
	if not hasattr(onclick1,"lines"): return
	et = event.xdata
	ev = event.ydata
	idx = np.where( np.abs(t-et)<h.dt)[0][0]
	#DB>>
	#print idx, et,ev
	#<<DB
	vmax = abs(neurons[0].volt.x[idx] - ev)
	zoolyclickevent.imax = 0
	for ind,n in map(None,xrange(ncell),neurons):
		onclick1.lines[ind].set_linewidth(1)
		onclick1.lines[ind].set_ls("-")
		if vmax > abs(n.volt.x[idx] - ev) :
			vmax = abs(n.volt.x[idx] - ev)
			zoolyclickevent.imax = ind
		#print vmax,n.volt.x[idx],ev
	onclick1.lines[zoolyclickevent.imax].set_linewidth(4)
	onclick1.lines[zoolyclickevent.imax].set_ls("--")
	zooly.canvas.draw()
	#print "#",zoolyclickevent.imax,"\t v:",neurons[zoolyclickevent.imax].volt.x[idx],"\t g:",neurons[zoolyclickevent.imax].gsyn.x[idx]
	
	zoolyclickevent.v,zoolyclickevent.u = np.array(neurons[zoolyclickevent.imax].volt),np.array(neurons[zoolyclickevent.imax].uolt)
	zoolyclickevent.g,zoolyclickevent.i = np.array(neurons[zoolyclickevent.imax].gsyn),np.array(neurons[zoolyclickevent.imax].inoise)
	vmin,vmax =zoolyclickevent.v[onclick1.idx].min(),zoolyclickevent.v[onclick1.idx].max()
	vx,u0,v0,v0n = getnulls(vmin,vmax,zoolyclickevent.g[idx], zoolyclickevent.i[idx],neurons[zoolyclickevent.imax].innp.mean)
	zoolyclickevent.sptk = numsptk(zoolyclickevent.imax,onclick1.idx)
	zoolyclickevent.rst = getprespikes(zoolyclickevent.imax,onclick1.tl, onclick1.tr)
	#print "size sptk %d, idx %d time %d"%(sptk.size,onclick1.idx.size,tprin[onclick1.idx].size)
	if not hasattr(zoolyclickevent,"lines"):
		zoolyclickevent.lines = []
		#zoolyclickevent.lines.append(faxi.plot(zoolyclickevent.rst[:,0],zoolyclickevent.rst[:,1],"k|",ms=9,lw=5)[0])
		zoolyclickevent.lines.append(faxi.plot(zoolyclickevent.rst[:,0],zoolyclickevent.rst[:,1],"k"+zoolyclickevent.spikesymbol,ms=9,lw=5)[0])
		zoolyclickevent.lines.append(vaxi.plot(tprin[onclick1.idx],zoolyclickevent.v[onclick1.idx],"k-")[0])
		zoolyclickevent.lines.append(uaxi.plot(tprin[onclick1.idx],zoolyclickevent.u[onclick1.idx],"k-")[0])
		zoolyclickevent.lines.append(gaxi.plot(tprin[onclick1.idx],zoolyclickevent.g[onclick1.idx],"k-")[0])
		zoolyclickevent.lines.append(naxi.plot(zoolyclickevent.v[onclick1.idx],zoolyclickevent.u[onclick1.idx],"k-")[0])
		zoolyclickevent.lines.append(naxi.plot(vx,u0,"k--")[0])
		zoolyclickevent.lines.append(naxi.plot(vx,v0,"k-.")[0])
		zoolyclickevent.lines.append(naxi.plot(vx,v0n,"k.")[0])
		zoolyclickevent.lines.append(iaxi.plot(tprin[onclick1.idx],zoolyclickevent.i[onclick1.idx],"k-")[0])
		#zoolyclickevent.lines.append(saxi.plot(tprin[onclick1.idx],zoolyclickevent.sptk,"k-")[0])
	else:
		zoolyclickevent.lines[0].set_xdata(zoolyclickevent.rst[:,0])
		zoolyclickevent.lines[0].set_ydata(zoolyclickevent.rst[:,1])
		zoolyclickevent.lines[1].set_xdata(tprin[onclick1.idx])
		zoolyclickevent.lines[1].set_ydata(zoolyclickevent.v[onclick1.idx])
		zoolyclickevent.lines[2].set_xdata(tprin[onclick1.idx])
		zoolyclickevent.lines[2].set_ydata(zoolyclickevent.u[onclick1.idx])
		zoolyclickevent.lines[3].set_xdata(tprin[onclick1.idx])
		zoolyclickevent.lines[3].set_ydata(zoolyclickevent.g[onclick1.idx])
		zoolyclickevent.lines[4].set_xdata(zoolyclickevent.v[onclick1.idx])
		zoolyclickevent.lines[4].set_ydata(zoolyclickevent.u[onclick1.idx])
		zoolyclickevent.lines[5].set_xdata(vx)
		zoolyclickevent.lines[5].set_ydata(u0)
		zoolyclickevent.lines[6].set_xdata(vx)
		zoolyclickevent.lines[6].set_ydata(v0)
		zoolyclickevent.lines[7].set_xdata(vx)
		zoolyclickevent.lines[7].set_ydata(v0n)
		zoolyclickevent.lines[8].set_xdata(tprin[onclick1.idx])
		zoolyclickevent.lines[8].set_ydata(zoolyclickevent.i[onclick1.idx])
		#zoolyclickevent.lines[9].set_xdata(tprin[onclick1.idx])
		#zoolyclickevent.lines[9].set_ydata(zoolyclickevent.sptk)
	faxi.set_ylim(0,ncell)
	vaxi.set_ylim(zoolyclickevent.v[onclick1.idx].min(),zoolyclickevent.v[onclick1.idx].max())
	uaxi.set_ylim(zoolyclickevent.u[onclick1.idx].min(),zoolyclickevent.u[onclick1.idx].max())
	gaxi.set_ylim(zoolyclickevent.g[onclick1.idx].min(),zoolyclickevent.g[onclick1.idx].max())
	iaxi.set_ylim(zoolyclickevent.i[onclick1.idx].min(),zoolyclickevent.i[onclick1.idx].max())
	#saxi.set_ylim(0,zoolyclickevent.sptk[-1]+2)
	faxi.set_xlim(onclick1.tl, onclick1.tr)
	naxi.set_xlim(vmin,vmax)
	naxi.set_ylim(zoolyclickevent.u[onclick1.idx].min(),zoolyclickevent.u[onclick1.idx].max())
	moddy.canvas.draw()
	
def zoolykeyevent(event):
	if not hasattr(zoolyclickevent,"lines"): return
	if event.key == "K":
		v,u,g,i = np.array(neurons[zoolyclickevent.imax].volt),np.array(neurons[zoolyclickevent.imax].uolt),np.array(neurons[zoolyclickevent.imax].gsyn),np.array(neurons[zoolyclickevent.imax].inoise)
		sptk = numsptk(zoolyclickevent.imax,onclick1.idx)
		rst = getprespikes(zoolyclickevent.imax,onclick1.tl, onclick1.tr)
		zoolyclickevent.lines.append(faxi.plot(zoolyclickevent.rst[:,0],zoolyclickevent.rst[:,1],zoolyclickevent.spikesymbol,ms=9,lw=5)[0])
		zoolyclickevent.lines.append(vaxi.plot(tprin[onclick1.idx],v[onclick1.idx])[0])
		zoolyclickevent.lines.append(uaxi.plot(tprin[onclick1.idx],u[onclick1.idx])[0])
		zoolyclickevent.lines.append(gaxi.plot(tprin[onclick1.idx],g[onclick1.idx])[0])
		zoolyclickevent.lines.append(naxi.plot(v[onclick1.idx],u[onclick1.idx])[0])
		zoolyclickevent.lines.append(iaxi.plot(tprin[onclick1.idx],i[onclick1.idx])[0])
		#zoolyclickevent.lines.append(saxi.plot(tprin[onclick1.idx],sptk)[0])
	elif event.key == "X":
		for lin in zoolyclickevent.lines:
			lin.remove()
		del zoolyclickevent.lines
	moddy.canvas.draw()	

def moddykeyevent(event):
	zoolykeyevent(event)


def neuronsoverview(event):
	global vindex
	if event.key == "up":
		vindex += 1
		if vindex >= ncell : vindex = ncell -1
	elif event.key == "down":
		vindex -= 1
		if vindex < 0 : vindex = 0
	elif event.key == "home": vindex = 0
	elif event.key == "end" : vindex = ncell -1
	if event.key == "pageup":
		vindex += 10
		if vindex >= ncell : vindex = ncell -1
	elif event.key == "pagedown":
		vindex -= 10
		if vindex < 0 : vindex = 0
	vtrace.set_ydata( np.array(neurons[vindex].volt)[:tprin.size])
	mainfig.canvas.draw()
		
#===============================================#
#               MAIN PROGRAMM                   #
#===============================================#
if __name__ == "__main__":
	if len(sys.argv) > 1:
		def patternmatch(pattern,arg):
			if arg[:len(pattern)] != pattern: return None
			return arg[len(pattern):]
		for arg in sys.argv:
			redarg = patternmatch("-corelog=",arg)
			if redarg != None: methods["corelog"] = redarg
			redarg = patternmatch("-gsyn=",arg)
			if redarg != None:
				redarg = redarg.split(",")
				if type(redarg) is list and  len(redarg) >= 2:
					weight = (float(redarg[0]),float(redarg[1]))
				else:
					weight = float(redarg[0])
			redarg = patternmatch("-gif=",arg)
			if redarg != None: methods['gif'] = redarg
			redarg = patternmatch("-Iapp=",arg)
			if redarg != None:
				redarg = redarg.split(",")
				if type(redarg) is list and  len(redarg) >= 2:
					Iapp = (float(redarg[0]),float(redarg[1]))
				else:
					Iapp = float(redarg[0])
			
			redarg = patternmatch("-Istd=",arg)
			if redarg != None: Istdev = float(redarg)
			redarg = patternmatch("-gui=",arg)
			if redarg != None:
				if redarg == "False" or redarg =="off" or redarg =="OFF" or redarg == '0':
					methods["gui"] = False
			redarg = patternmatch("-git",arg)
			if redarg != None: methods["git"] = True
			redarg = patternmatch("-core=",arg)
			if redarg != None: methods['corefunc'] = int(redarg)
			redarg = patternmatch("-F=",arg)
			if redarg != None:
				redarg = redarg.split(",")
				if type(redarg) is list and len(redarg) >= 2:
					F = (float(redarg[0]),float(redarg[1]))
				else:
					F = float(redarg[0])
			redarg = patternmatch("-noexit",arg)
			if redarg != None: methods["noexit"] = True
			redarg = patternmatch("-gsynscale=",arg)
			if redarg != None:
				redarg = redarg.split(",")
				if type(redarg) is list and  len(redarg) >= 2:
					gsynscale = (float(redarg[0]),float(redarg[1]))
				else:
					gsynscale = float(redarg[0])
			redarg = patternmatch("-FPcurve",arg)
			if redarg != None: methods["FPcurve"] = True
			redarg = patternmatch("-GPcurve",arg)
			if redarg != None: methods["GPcurve"] = True
			redarg = patternmatch("-dt=",arg)
			if redarg != None: methods["timestep"] = float(redarg)
#			redarg = patternmatch("-tstop=",arg)
#			if redarg != None: tstop = float(redarg)
			redarg = patternmatch("-n=",arg)
			if redarg != None: ncell = int(redarg)
			redarg = patternmatch("-c=",arg)
			if redarg != None:
				redarg = redarg.split(",")
				if len(redarg) == 1: ncon = int(redarg[0])
				if len(redarg) == 2: ncon = ( int(redarg[0]), int(redarg[1]) )
				if len(redarg) == 3: ncon = ( int(redarg[0]), int(redarg[1]) , int(redarg[2]))
			### Synapses
			redarg = patternmatch("-tau1=",arg);
			if redarg != None: ST1					= float(redarg)
			redarg = patternmatch("-tau2=",arg);	
			if redarg != None: ST2					= float(redarg)
			redarg = patternmatch("-Esyn=",arg);	
			if redarg != None: SE					= float(redarg)
			redarg = patternmatch("-taunorm=",arg);
			if redarg != None: methods['taunorm']	= bool(int(redarg))
			redarg = patternmatch("-tsynscaler=",arg);	
			if redarg != None: exec "synscaler = "+redarg

			redarg = patternmatch("-external=",arg);
			if redarg != None:
				exec( "methods['external'] = "+redarg )

			redarg = patternmatch("-delay=",arg)
			if redarg != None:
				redarg = redarg.split(",")
				if type(redarg) is list  and len(redarg) >= 2:
					delay = (float(redarg[0]),float(redarg[1]))
				else:
					delay = (float(redarg[0]), 0.0)
			
			redarg = patternmatch("-view",arg)
			if redarg != None: tstop			= tvr + 1
			redarg = patternmatch("-tstop=",arg)
			if redarg != None:
				tvr			= float(redarg)
				tstop		= tvr + 1
			redarg = patternmatch("-bias=",arg)
			if redarg != None: methods['bias'] = float(redarg)
			redarg = patternmatch("-sort=",arg)
			if redarg != None: methods['sortbysk'] = redarg
			redarg = patternmatch("-gtot-dist=",arg)
			if redarg != None: methods['gtot-dist'] = redarg
			redarg = patternmatch("-gsyn-dist=",arg)
			if redarg != None: methods['gsyn-dist'] = redarg
			redarg = patternmatch("-initset=",arg)
			if redarg != None: exec "methods['initset'] ="+redarg
			
			redarg = patternmatch("-Vinit=",arg)
			if redarg != None:
				methods['initset'] = None
				exec( "V = "+redarg)
			redarg = patternmatch("-Uinit=",arg)
			if redarg != None:
				methods['initset'] = None
				exec( "U = "+redarg)
			redarg = patternmatch("-nstart=",arg)
			if redarg != None:
				exec( "methods['nstart'] = "+redarg)
			if arg == '-h' or arg == '-help' or arg == '--h' or arg == '--help':
				print __doc__
				print 
				print "USAGE: nrngui -nogui -python network.py [parameters]"
				print "\nPARAMETERS:"
				print "-n=          number of neurons in population"
				print "-c=          number of connections per neuron"
				print "-Iapp=       apply current. Use scaling factor 1e-5 to get nA"
				print "             Iapp may be a constant or mean,standard deviation across population."
				print "-Istd=       amplitude of noise. Should be scaled by 1e-5 to get nA"
				print "-gui=ON|OFF  Turn on/off gui and graphs"
				print "-F=          Set up neuron dynamics scale factor"
				print "             F may be a constant or mean,standard deviation across population."
				print "-gsyn=       conductance of single synapse. Use scaling factor 1e-5 to get nS"
				print "             gsyn may be a constant or mean,standard deviation for all synapses in model"
				print "-gsynscale=  total synaptic conductance.  Use scaling factor 1e-5 to get nS"
				print "             gsynscale may be a constant or mean,standard deviation for all neurons within population"
				print "-tau1=       rising time constant in ms"
				print "-tau2=       falling time constant in ms"
				print "-Esyn=       synaptic reversal potential in mV"
				print "-taunorm=0|1 On or Off normalization by space under the curve"
				print "-tsynscaler= scaling coefficient for synaptic time constants"
				print "-delay=      axonal delay in ms"
				print "             delay may be a constant or mean,standard deviation for all synapses in model"
				print "-view        limits simulation and save memory"
				
				exit(0)
			
			

	
	with open("network.start","w") as fd:
		for arg in sys.argv: fd.write("%s "%arg)
	if (ST1 != 2.0 or ST2 != 5.0) and methods['taunorm']:
		from norm_translation import getscale
		nFactor = getscale(2.0,5.0,ST1,ST2)
		if type(weight) == tuple or type(weight) == list:
			weight		= (weight[0]*nFactor, weight[1]*nFactor)
		else:
			weight		= (weight*nFactor, 0.)

###DB>
	print "=================================="
	print "===         :PARAMETERS:       ==="
	print "=================================="
	print "  > a=",a,"b=",b,"c=",c,"d=",d
	print "  > U0=",U,"V0=",V
	print "  > F           = ",F
	print "  > Bias        = ",methods['bias']
	print "  > Iapp        = ",Iapp
	print "  > Istdev      =  %g"%Istdev
	print "  > gsyn        = ",weight
	print "  > gsyn dist.  = ", methods['gsyn-dist'] 
	print "  > delay       = ",delay
	print "  > gsynscale   = ", gsynscale
	print "  > gtotal dis. = ", methods['gtot-dist'] 
	print "  > dt          = ", methods["timestep"]
	print "  > ncell       = ", ncell
	print "  > ncon        = ", ncon
	print "  > Syn tau1/2  = ", ST1,"/",ST2
	print "  > Syn E       = ", SE
	print "  > Syn taunorm = ", methods['taunorm']
	print "  > Syn t scaler= ", synscaler
	print "  > External    = ", methods['external']
	print "  > tstop       = ", tstop
	print "  > Sorted by   : ", methods['sortbysk']
	print "  > Init setup  : ", methods['initset']
	print "  > Noise Start : ", methods['nstart']
	print "==================================\n"
###<DB

	
	if methods["gui"]:
		import matplotlib.pyplot as plt
		print "=================================="
		print "===        GUI turned ON       ==="
		print "==================================\n"
	
	h.init()
	h.tstop 	= tstop
	#h.v_init 	= V
	h.dt		= methods["timestep"]
	

	#### Create Neurons and setup noise and Iapp
	print "=================================="
	print "===        Create Neurons      ==="
	print "==================================\n"
	neurons = [ neuron() for x in xrange(ncell) ]
	
	if type(V) is float or type(V) is int:
		xV = V*np.ones(ncell)
	elif type(V) is tuple:
		xV = V[0]+np.random.randn(ncell)*V[1]
	elif type(V) is str:
		xV = np.genfromtxt(V)
	if type(U) is float or type(U) is int:
		xU = U*np.ones(ncell)
	elif type(U) is tuple:
		xU = U[0]+np.random.randn(ncell)*U[1]
	elif type(U) is str:
		xU = np.genfromtxt(U)
			
	for n,i in zip(neurons,xrange(ncell)):
		if type(F) is float or type(F) is int:
			Fx = F
		elif type(F) is tuple:
			Fx = F[0] + np.random.randn()*F[1]
			while Fx <= 0.0: Fx = F[0] + np.random.randn()*F[1]
		if type(Iapp) is float or type(Iapp) is int:
			xIapp = Fx*Iapp
		elif type(Iapp) is tuple:
			xIapp = (Fx*Iapp[0]+Fx*np.random.randn()*Iapp[1])
		if synscaler != None:
			if type(synscaler) is float or type(synscaler) is int:
				n.tsynscale = float(synscaler)
				xST1,xST2 = ST1 * n.tsynscale, ST2 * n.tsynscale
			elif type(synscaler) is list or type(synscaler) is tuple:
				if len(synscaler) >= 2:
					n.tsynscale = -1.0
					while( n.tsynscale < 0.0 ):
						n.tsynscale = synscaler[0] + np.random.randn()*synscaler[1]
					xST1,xST2 = ST1 * n.tsynscale, ST2 * n.tsynscale
				else :
					n.tsynscale = float(synscaler[0])
					xST1,xST2 = ST1 * n.tsynscale, ST2 * n.tsynscale
			else:
				xST1,xST2 = ST1,ST2
		else:
			xST1,xST2 = ST1,ST2


		n.setparams(a=a, b=b, c=c, d=d, V=xV[i], U=xU[i], F=Fx, SynT1=xST1, SynT2=xST2, SynE=SE, Iapp=-xIapp, Insd=Istdev)
	
	if methods['initset'] != None:
		for setting in methods['initset']:
			if type(setting[0]) is int:
				itr=[setting[0]]
			elif type(setting[0]) is tuple or type(setting[0]) is list:
				if len(setting[0]) == 1: itr=setting[0][0]
				if len(setting[0]) == 2: itr=range(setting[0][0],setting[0][1])
				if len(setting[0]) == 3: itr=range(setting[0][0],setting[0][1],setting[0][2])
			for n in itr:
				if 'V' in setting[1]:
					if type(setting[1]['V']) is float or type(setting[1]['V']) is int:
						neurons[n].soma(0.5).v = setting[1]['V']
					elif type(setting[1]['V']) is tuple or type(setting[1]['V']) is list:
						neurons[n].soma(0.5).v = setting[1]['V'][0]+setting[1]['V'][1]*np.random.randn()
				if 'U' in setting[1]:
					if type(setting[1]['U']) is float or type(setting[1]['U']) is int:
						neurons[n].izh.uinit  = setting[1]['U']
					elif type(setting[1]['U']) is tuple or type(setting[1]['U']) is list:
						neurons[n].izh.uinit  = setting[1]['U'][0]+setting[1]['U'][1]*np.random.randn()
						
	#DB>>
	#for n in neurons:
	#	print n.soma.v, n.izh.uinit
	#exit(0)
	#<<DB
		

	if methods['nstart']:
		for n in neurons:
			n.addnoise(Iapp=0.,Insd=methods['nstart'][1],delay=methods['nstart'][0],dur=methods['nstart'][2],per=0.1)

	h.finitialize()
	h.fcurrent()
	h.frecord_init()

	t = h.Vector()
	t.record(h._ref_t)


	#### Create Connection List:
	OUTList = [ [] for x in xrange(ncell)]
	if ncon > 0:
		print "=================================="
		print "===        Map Connections     ==="
		print "==================================\n"
		for toid in xrange(ncell):
			from_excaption = [ 0 for x in xrange(ncell) ]
			from_excaption[toid] = 1
			upcnt = 0
			total = 0
			if type(ncon) is tuple or type(ncon) is list:
				if len(ncon) > 1:
					neurons[toid].concnt = int(np.random.random()*(ncon[1]-ncon[0]) + ncon[0])
				else:
					neurons[toid].concnt = ncon[0]
			else:
				neurons[toid].concnt = ncon
			while  upcnt < 10000*ncell:
				upcnt += 1
				fromid = rnd.randint(0, ncell-1)
				if from_excaption[fromid] == 1 : continue
				upcnt  = 0
				total += 1
				from_excaption[fromid] = 1
				OUTList[toid].append(fromid)
				if total >= neurons[toid].concnt :break
			else:
				sys.stderr.write("Couldn't obey connections conditions\nNeuron:%d TOTLA:%d CURRENT:%d\n"%(toid,ncon,total))
				for x in OUTList:
					sys.stderr.write("ID:%d #%d\n"%(x[0],x[1]))
				sys.exit(1)

	#DB>
		#for i in OUTList:
			#print len(i)
			#for j in i:	print "%03d"%(j),
			#print
		#sys.exit(0)
	#<DB
	if methods['cycling']:
		print "=================================="
		print "===      Cycles counting       ==="
		print "=================================="
		mat=np.matrix( np.zeros((ncell,ncell)) )
		for i,vec in map(None,xrange(ncell),OUTList):
			mat[i,vec]=1
		kx = []
		for cnt in xrange(methods['cycling']):
			kx.append(np.trace(mat)/ncell)
			mat = mat.dot(mat)
		print "  > Cyclopedic numbers : ",kx
		print "==================================\n"
		#methods['cycling'] = kx
		del mat
		
		
	

	#### Create Conneactions:
	print "=================================="
	print "===    Make the Connections    ==="
	print "==================================\n"
	connections = []
	for x in map(None,xrange(ncell),OUTList):
		if type(gsynscale) is tuple :
			if gsynscale[1] <= 0: gx = gsynscale[0]
			elif methods["gtot-dist"] == "NORM":
				### Trancated normal
				gx = gsynscale[1]*np.random.randn()+gsynscale[0]
				while gx < 0.0 : gx = gsynscale[1]*np.random.randn()+gsynscale[0]
			elif methods["gtot-dist"] == "LOGN":
				### Lognormal
				gx = np.random.lognormal(mean=mp.log(gsynscale[0])-gsynscale[1]**2/2., sigma=gsynscale[1])
		else:
			gx = float(gsynscale)
		neurons[x[0]].gsynscale = gx
		for pre in x[1]:
			if type(delay) is tuple :
				if delay[1] > 0:
					dx = -1
					while dx < 0.1: dx = delay[1]*np.random.randn()+delay[0]
				else:
					dx = float(delay[0])
			else:
				dx = float(delay)
			if type(weight) is tuple :
				if weight[1] <= 0: wx = weight[0]
				elif methods["gsyn-dist"] == "NORM":
					#### Trancated normal
					wx = weight[1]*np.random.randn()+weight[0]
					while wx < 0.0 : wx = weight[1]*np.random.randn()+weight[0]
				elif methods["gsyn-dist"] == "LOGN":
					### Lognormal
					wx = np.random.lognormal(mean=np.log(weight[0])-weight[1]**2/2., sigma=weight[1])
			else:
				wx = float(weight)
			if type(ncon) is tuple or type(ncon) is list:
				if len(ncon) > 2:
					wx *= float(ncon[2])/float(neurons[x[0]].concnt)
			if methods['taunorm'] and synscaler != None:
				#DB print "norm by factor",1./neurons[x[0]].tsynscale
				wx /= neurons[x[0]].tsynscale
			#####DB>>
			#print "DB:gx=",gx,"dx=",dx,"wx=",wx
			#####<<DB
			#connections.append( h.NetCon(neurons[pre].soma(0.5)._ref_v,neurons[x[0]].syn,
					#25, dx, gx*wx,
					#sec=neurons[pre].soma) )
			connections.append( (h.NetCon(neurons[pre].soma(0.5)._ref_v,neurons[x[0]].syn,
					25, dx, gx*wx,
					sec=neurons[pre].soma),pre,x[0]) )
			neurons[x[0]].gtotal += gx*wx
	#DB>>
	#plt.figure(0)
	#w=np.array([c[0].weight[0] for c in connections])
	#print np.mean(w), np.std(w)
	#plt.hist(w,bins=50,range=(0,1e-6))
	#plt.show()
	#exit(0)
	#<<DB

	
	if methods['external']:
		ex_netstim	= h.NetStim(.5, sec = neurons[0].soma)
		ex_netstim.start	= methods['external'][0]
		ex_netstim.noise	= 0
		ex_netstim.interval	= methods['external'][1]
		ex_netstim.number	= methods['external'][2]
		ex_syn,ex_netcon = [],[]
		for n in neurons:
			if len(methods['external']) >=9:
				if rnd.random() > methods['external'][8]: continue
			ex_syn_new = h.Exp2Syn(0.5, sec=n.soma)
			ex_syn_new.e	= methods['external'][4]
			ex_syn_new.tau1	= methods['external'][5]
			ex_syn_new.tau2	= methods['external'][6]
			ex_syn.append(ex_syn_new)
			exdelay = -1.0
			if len(methods['external']) < 8:
				exdelay = 0.1
			elif type(methods['external'][7]) is tuple :
				while exdelay <= 0.0 : exdelay = methods['external'][7][0]+np.random.randn()*methods['external'][7][1]
			elif type(methods['external'][7]) is float or type(methods['external'][7]) is int :
				exdelay = methods['external'][7]
			
			ex_netcon_new	= h.NetCon(ex_netstim, ex_syn_new, 1,exdelay ,methods['external'][3], sec = n.soma)
			ex_netcon.append(ex_netcon_new)

	print "=================================="
	print "===           RUN              ==="
	print "==================================\n"
	npc = h.ParallelContext()
	if type(methods['corefunc']) is int:
		npc.nthread(methods['corefunc'])
		sys.stderr.write( "Setup %g core\n"%(methods['corefunc']) )
	elif delay[0] > h.dt*2 or delay[1] > h.dt*2 :
		#### Setup parallel context if there are delays
		if not os.path.exists("/etc/beowulf") and os.path.exists("/sysini/bin/busybox"):
			#I'm not on head node. I can use all cores
			methods['corefunc'] = methods['corefunc'][2]
			npc.nthread(methods['corefunc'])
			sys.stderr.write( "Setup %g core\n"%(methods['corefunc']) )
		elif os.path.exists("/etc/beowulf"):
			#I'm on head node. I grub only half :)
			methods['corefunc'] = methods['corefunc'][1]
			npc.nthread(methods['corefunc'])
			sys.stderr.write( "Setup %g core\n"%(methods['corefunc']) )
		else:
			#I'm on Desktop :(
			methods['corefunc'] = methods['corefunc'][0]
			npc.nthread(methods['corefunc'])
			sys.stderr.write( "Setup %g core\n"%(methods['corefunc']) )
	h.run()

	if methods["save"]:
		print "=================================="
		print "===     Saving the Results     ==="
		print "==================================\n"
		fd = open("network-param.txt","w")
		fd.write("ncell=%d\nncon=%d\nntype=%s\n"%(ncell,ncon,ntype))
		fd.write("a=%g\nb=%g\nc=%g\nd=%g\nU=%g\nV=%g\n"%(a,b,c,d,U,V))
		fd.write("Iapp=%g\nIstdev=%g\nweight=(%g,%g)\ndelay=(%g,%g)\n"%(Iapp,Istdev,weight[0],weight[1],delay[0],delay[1]))
		fd.write("ST1=%g\nST2=%g\nSE=%g\n"%(ST1,ST2,SE))
		fd.write("tstop=%g\ndt=%g\nnsampls=%d"%(tstop,h.dt,t.size()))
		fd.close()
		fdv = open("network-volt.csv","w")
		fdn = open("network-noise.csv","w")
		fds = open("network-syn.csv","w")
		for idx in xrange(int(np.round(t.size()))):
			vstr = "%g"%t.x[idx]
			nstr = "%g"%t.x[idx]
			sstr = "%g"%t.x[idx]
			for n in neurons:
				vstr +=",%g"%n.volt.x[idx]
				nstr +=",%g"%n.inoise.x[idx]
				sstr +=",%g"%n.isyn.x[idx]
			fdv.write(vstr+"\n")
			fdn.write(nstr+"\n")
			fds.write(sstr+"\n")
		fdv.close()
		fdn.close()
		fds.close()

	print "=================================="
	print "===          Analysis          ==="
	print "==================================\n"
	
	t = np.array(t)
	if methods['gui']:
		plt.rc('text', usetex = True )
		plt.rc('font', family = 'serif')
		plt.rc('svg', fonttype = 'none')
		mainfig = plt.figure(1)
		if methods['traceView']:
			cid = mainfig.canvas.mpl_connect('button_press_event', onclick1)
		p=plt.subplot(411)
		tprin=np.array(t)
		tprin = tprin[ np.where( tprin < tvr ) ]
		vindex = (ncell-1)/2
		vtrace, = plt.plot(tprin,np.array(neurons[vindex].volt)[:tprin.size],"k")
		plt.ylim(ymax=40.)
		mainfig.canvas.mpl_connect('key_press_event',neuronsoverview)
		plt.ylabel("Voltage (mV)", fontsize=16)
		if methods["external"]:
			ex0 = methods["external"][0]
			ex1 = methods["external"][1]
			for ex2 in xrange(methods["external"][2]):
				plt.plot([ex0+ex1*ex2,ex0+ex1*ex2],[0,30],"r--")
		plt.subplot(412,sharex=p)
		nurch = np.arange(1,ncell+1,1)
		if methods['sortbysk']:
			if methods['sortbysk'] == 'F':
				nindex = [ (neurons[i].izh.F,i) for i in xrange(ncell)]
				nindex.sort()
				#print nindex
				for i in xrange(ncell):
					nurch[nindex[i][1]]=i
			if methods['sortbysk'] == 'I':
				nindex = [ (-neurons[i].innp.mean,i) for i in xrange(ncell)]
				nindex.sort()
				#print nindex
				for i in xrange(ncell):
					nurch[nindex[i][1]]=i
			if methods['sortbysk'] == 'G':
				nindex = [ (-neurons[i].gsynscale,i) for i in xrange(ncell)]
				nindex.sort()
				#print nindex
				for i in xrange(ncell):
					nurch[nindex[i][1]]=i
			if methods['sortbysk'] == 'NC':
				nindex = [ (-neurons[i].concnt,i) for i in xrange(ncell)]
				nindex.sort()
				#print nindex
				for i in xrange(ncell):
					nurch[nindex[i][1]]=i
			if methods['sortbysk'] == 'GT':
				nindex = [ (-neurons[i].gtotal,i) for i in xrange(ncell)]
				nindex.sort()
				#print nindex
				for i in xrange(ncell):
					nurch[nindex[i][1]]=i
			if methods['sortbysk'] == 'ST':
				nindex = [ (neurons[i].tsynscale,i) for i in xrange(ncell)]
				nindex.sort()
				#print nindex
				for i in xrange(ncell):
					nurch[nindex[i][1]]=i
				
			#print nurch
	pmean = 0.
	pcnt  = 0


	meancur = np.zeros(t.size)
	spbins  = np.zeros( int(np.floor(tstop))+1 )
	specX	= np.arange(spbins.size, dtype=float)
	specX	*= 1000.0/tstop
	#pnum	= specX.size/2
	pnum 	= 200.*tstop/1000.0
	specX	= specX[:pnum]
	if methods["nrnFFT"]:
		specN	= np.zeros(pnum)
#	specV	= np.zeros(t.size())
	if 10 < methods["nrnISI"] <= 3000:
		isi		= np.zeros(methods["nrnISI"])
	if methods['coreindex']:
		coreindex = [0.0, 0.0]

	if methods['gui']:
		xrast = np.array([])
		yrast = np.array([])
	for (idx,n) in map(None,xrange(ncell),neurons):
		n.spks = np.array(n.spks)
		if methods['gui']:
			spk = n.spks[ np.where (n.spks < tvr) ]
			if not methods['cliprst']:
				xrast = np.append(xrast,spk)
				yrast = np.append(yrast,np.repeat(nurch[idx],spk.size))
			elif idx%methods['cliprst'] == 0:
				xrast = np.append(xrast,spk)
				yrast = np.append(yrast,np.repeat(nurch[idx],spk.size))
		
		if methods['cliptrn']:
			fstidx = np.where(n.spks > methods['cliptrn'] )[0][0]
			aisi = n.spks[fstidx+1:] - n.spks[fstidx:-1]
		else:
			aisi = n.spks[1:] - n.spks[:-1]
		if methods['coreindex']:
			coreindex[0] += np.sum((aisi[1:] - aisi[:-1])/aisi[:-1])
			coreindex[1] += aisi.size - 1
		if 10 < methods["nrnISI"] <= 3000:
			for i in aisi[ np.where(aisi < methods["nrnISI"]) ]:
				isi[ int(np.floor(i)) ] += 1.0
		pmean += np.sum(aisi)
		pcnt  += n.spks.size - 1
		if methods['gui']:
			if methods['tracetail'] == 'total current':
				meancur += np.array(n.isyn.x) + np.array(n.inoise.x)
			elif methods['tracetail'] == 'current':
				meancur += np.array(n.isyn.x)
			elif methods['tracetail'] == 'conductance':
				meancur += np.array(n.gsyn.x)
		spn	= np.zeros(spbins.size)
		for sp in n.spks:
			spbins[ int( np.floor(sp) ) ] +=1
			spn[ int( np.floor(sp) ) ] +=1
		if methods['cliptrn']:
			spn = spn[methods['cliptrn']:]
		if methods["nrnFFT"]:
			fft = np.abs( np.fft.fft(spn)*1.0/tstop )**2
			specN += fft[:pnum]
	if methods['gui']:
		plt.plot(xrast,yrast,"k|",mew=0.75,ms=1)#,ms=10)
		
		if methods['fullrast']	: plt.ylim(ymin=0,ymax=ncell)
		else			: plt.ylim(ymin=0)
	if methods["nrnFFT"]:
		specN /= float(ncell)#	specV /= float(ncell)
	
	if methods['cliptrn']:
		spbins = spbins[methods['cliptrn']:]
	
	if methods['popfr']:
		popfr = np.mean(spbins)
		print "=================================="
		print "===       MEAN FIRING RATE     ==="
		print "  > MFR =           ",popfr
		print "==================================\n"

	if (methods["netFFT"] or methods["nrnFFT"]):
		print "=================================="
		print "===            FFT             ==="
		print "==================================\n"
		fft = np.abs( np.fft.fft(spbins)*1.0/tstop )**2

	##EN
	#probscale = np.zeros(ncell + 1)
	#probscale[0] = 1./float(ncell + 1)
	#for x in range(1,ncell + 1):
		#probscale[x] = probscale[0]*probscale[x-1]
	#pspbin = np.array([ probscale[int(x)] for x in  spbins] )
	#en = np.sum( (-1)*pspbin*np.log(pspbin) )
	
	if methods['coreindex']:
		coreindex = coreindex[0]/coreindex[1]
		print coreindex, 1./(1.+ abs(coreindex))
		sys.exit(0)
		#with open("coreindex.csv","w") as fd:
			#for i in coreindex: fd.write("%g\n"%i)
		#coreindex = np.corrcoef(coreindex[:-1],y=coreindex[1:])[0][1]
	
	#external stimulation index
	if methods['external'] and methods['extprop']:
		print "=================================="
		print "===      Spike Probability     ==="
		print "==================================\n"
		spprop = 0
		for etx in xrange(methods['external'][2]):
			lidx = int( np.floor(methods['external'][0]+methods['external'][1]*etx) )
			ridx = int( np.floor(lidx + methods['external'][1]*methods['extprop']) )
			spprop += float( np.sum(spbins[lidx:ridx]) )
		spprop /= methods['external'][2]*ncell
			

	if methods["peakDetec"] or methods["R2"]:
		print "=================================="
		print "===         Peak Detector      ==="
		print "==================================\n"
		kernel = np.arange(-methods["gkernel"][1],methods["gkernel"][1],1.)
		kernel = np.exp(kernel**2/methods["gkernel"][0]/(-methods["gkernel"][0]))
		module = np.convolve(spbins,kernel)
		module = module[kernel.size/2:1-kernel.size/2]
		#spbinmax = (np.diff(np.sign(np.diff(module))) < 0).nonzero()[0] + 1
		#spbinmin = (np.diff(np.sign(np.diff(module))) > 0).nonzero()[0] + 1
		spbinmark = []
		for idx in (np.diff(np.sign(np.diff(module))) < 0).nonzero()[0] + 1:
			spbinmark.append([idx,1])
		for idx in (np.diff(np.sign(np.diff(module))) > 0).nonzero()[0] + 1:
			spbinmark.append([idx,-1])
		spbinmark.sort()
		spbinmark = np.array(spbinmark)
		peakmark  = []
		spc,ccnt = 0.,0.
		for mx in np.where( spbinmark > 0 )[0]:
			if mx <= 2 or mx >= (spbinmark.size/2 -2):continue
			if spbinmark[mx-1][1] > 0 or spbinmark[mx+1][1] > 0 or spbinmark[mx][1] < 0:continue
			peakmark.append(spbinmark[mx])
			ccnt += 1
			spc += np.sum(spbins[spbinmark[mx-1][0]:spbinmark[mx+1][0]])
		if ccnt > 0:
			spc /= ccnt
		
			

	##R2
	##Per
	if methods["R2"]:
		print "=================================="
		print "===             R2             ==="
		print "==================================\n"
		X,Y,Rcnt,netpermean,netpercnt=0.,0.,0.,0.0,0.0
		phydist  = []
		for mx in np.where( spbinmark > 0 )[0]:
			if mx >= (spbinmark.size/2 - 3):continue
			if spbinmark[mx+1][1] > 0 or spbinmark[mx+2][1] < 0 or spbinmark[mx][1] < 0:continue
			Pnet = float(spbinmark[mx+2][0] - spbinmark[mx][0])
			netpermean += Pnet
			netpercnt  += 1.
			for n,i in map(None,spbins[spbinmark[mx][0]:spbinmark[mx+2][0]],xrange(spbinmark[mx+2][0] - spbinmark[mx][0])):
				phyX = np.cos(np.pi*2.*float(i)/Pnet)
				phyY = np.sin(np.pi*2.*float(i)/Pnet)
				X += n*phyX
				Y += n*phyY
				Rcnt += n
				if methods['sycleprop']:
					#phydist.append( (360.*np.arctan2(phyY,phyX)/2/np.pi,n) )
					phydist.append( (np.arctan2(phyY,phyX),n) )
		if Rcnt > 0.:
			R2 = np.sqrt((X/Rcnt)**2+(Y/Rcnt)**2)
		if netpercnt > 0.:
			netpermean /= ( netpercnt - 1)
		if methods['sycleprop']:
			phydist = np.array(phydist)
			phydist[:,1] /= np.sum(phydist[:,1])
			phyhist,phyhistbins = np.histogram(phydist[:,0], bins=37, weights=phydist[:,1],range=(-np.pi-np.pi/36,np.pi+np.pi/36))
			
		
	if 10 < methods["netISI"] < 3000:
		print "=================================="
		print "===          NET ISI           ==="
		print "==================================\n"
		netisi	= np.zeros(methods["netISI"])
		lock = threading.RLock()
		
		def calcnetisi(ns):
			global netisi, lock
			scans	= np.zeros(ncell,dtype=int)
			localnetisi = np.zeros(methods["netISI"])
			for n in ns:
				for sp in n.spks:
					for (idx,m) in map(None,xrange(ncell),neurons):
						if m.spks.size < 2 : continue
						while m.spks[scans[idx]] <= sp and scans[idx] < m.spks.size - 1 : scans[idx] += 1
						if m.spks[scans[idx]] <= sp : continue
						if m == n and m.spks[scans[idx]] - sp < 1e-6 : continue
						aisi = m.spks[scans[idx]] - sp
						if int(aisi) >= methods["netISI"] : continue
						localnetisi[ int(aisi) ] += 1
			with lock:
				netisi += localnetisi
		pids = [ threading.Thread(target=calcnetisi, args=(neurons[x::methods['corefunc']],)) for x in xrange(methods['corefunc']) ]
		for pidx in pids:
			pidx.start()
			#print pidx, "starts"
		for pidx in pids:
			pidx.join()
			#print pidx,"finishs"
			
			
	if methods["TaS"] or methods['lastspktrg']:
		print "=================================="
		print "===           T & S            ==="
		print "==================================\n"
		allspikes,activeneurons = [],0.
		for n in neurons:
			allspikes += list(n.spks)
			if n.spks.size !=0 :activeneurons += 1.
		allspikes.sort()
		allspikes = np.array(allspikes)
		TaSisi = allspikes[1:]-allspikes[:-1]
		if methods['lastspktrg']:
			lastspktrg = int( np.mean(allspikes) > tstop/4. )
			
		
		del allspikes
		mean1TaSisi = np.mean(TaSisi)
		TaSindex	= (np.sqrt(np.mean(TaSisi**2) - mean1TaSisi**2)/mean1TaSisi - 1.)/np.sqrt(activeneurons) 
		
		
	if methods['Gtot-dist'] :
		print "=================================="
		print "===   G-scaler distribution    ==="
		print "==================================\n"
		gsk = [ n.gsynscale for n in neurons ]
		gskhist,gskbins = np.histogram(gsk, bins=ncell/25, normed=True)#/10)#, normed=True)
		gskhist /= np.sum(gskhist)
		del gsk
		#DB>>
		#print gskhist
		#print gskbins
		#print np.sum(gskhist)
		#<<DB
		
	if methods['Gtot-stat']:
		print "=================================="
		print "===     G-total Statistics     ==="
		agtot = np.array([ n.gtotal/n.concnt for n in neurons ])
		mgtot = np.mean(agtot)
		sgtot = np.std(agtot)
		print "  > mean   gtotal =           ",mgtot
		print "  > stderr gtotal =           ",sgtot
		print "  > CV     gtotal =           ",sgtot/mgtot
		print "==================================\n"

			
	#EN
	#p.set_title("Mean individual Period = %g, Sychrony(Entropy) = %g(%g)"%(pmean/pcnt,1./(1.+en),en))
	
	##R2
	if methods['gui']:
		title = "Mean individual Period = %g"%(pmean/pcnt)
		if methods['popfr']:
			title += 'Mean FR =%g'%popfr
		if methods["R2"]:
			if Rcnt > 0 :
				title += r", $R^2$ = %g, Mean network Period = %g, Spike per cycle = %g"%(R2**2,netpermean,spc)
			else:
				title += ", *Fail to estimate network period*"
		elif methods["peakDetec"]:
			title += ", Spike per cycle = %g. "%(spc)
		if methods['TaS']:
			title += ", TaS = %g"%TaSindex
		if methods['lastspktrg']:
			title += ", LST = %g"%lastspktrg
		p.set_title(title)

		plt.subplot(413,sharex=p)
		if methods['cliptrn']:
			nppoints = np.arange(tvl+methods['cliptrn'],tvr,1.0)
			plt.bar(nppoints,spbins[:tvr-methods['cliptrn']],0.5,color="k")
			hight = spbins[:tvr-methods['cliptrn']].max()
			if methods["peakDetec"] or methods["R2"] :
				for mark in spbinmark:
					if mark[0]+methods['cliptrn'] < tvl or mark[0]+methods['cliptrn'] > tvr: continue
					if mark[1] > 0:
						plt.plot([mark[0]+methods['cliptrn'],mark[0]+methods['cliptrn']],[0,hight],"r--")
					else:
						plt.plot([mark[0]+methods['cliptrn'],mark[0]+methods['cliptrn']],[0,hight],"b--")
		else:
			nppoints = np.arange(tvl,tvr,1.0)
			plt.bar(nppoints,spbins[tvl:tvr],0.5,color="k")
			hight = spbins[tvl:tvr].max()
			if methods["peakDetec"] or methods["R2"] :
				for mark in spbinmark:
					if mark[0] < tvl or mark[0] > tvr: continue
					if mark[1] > 0:
						plt.plot([mark[0],mark[0]],[0,hight],"r--")
					else:
						plt.plot([mark[0],mark[0]],[0,hight],"b--")
#			plt.plot(nppoints,module[tvl:tvr]/np.sum(kernel),"k--")
#			plt.plot(nppoints,module[tvl:tvr],"k--")
		plt.ylabel("Rate (ms$^{-1}$)", fontsize=16)


	meancur = meancur / float(-ncell)
#	meancur = meancur / float(ncell)
	if methods['gui']:
		plt.subplot(414,sharex=p)
		if methods['tracetail'] == 'total current' or methods['tracetail'] == 'current':
			if ntype == "RS":
				#Hysteresis of type II
				plt.ylabel("Current (nA)", fontsize=16)
				plt.plot(tprin,meancur[:tprin.size]*10e6)
				plt.plot([tvl,tvr],[0.000001821*10e6,0.000001821*10e6],"r--")
				plt.plot([tvl,tvr],[0.000002625*10e6,0.000002625*10e6],"r--")
				#plt.plot([0,2000],[0.00000,0.0],"r--")
			elif ntype == "TypeI":
				# Sidle-node on type I
				plt.ylabel(r"Current ($\mu$A)", fontsize=16)
				plt.plot(tprin,meancur[:tprin.size]*10e3)
				plt.plot([0,2000],[0.00022562*10e3,0.00022562*10e3],"r--")
			else:
				plt.ylabel("Current (nA)", fontsize=16)
				plt.plot(tprin,meancur[:tprin.size]*10e6)
		elif methods['tracetail'] == 'conductance':
			plt.ylabel("Conductance (nS)", fontsize=16)
			plt.plot(tprin,meancur[:tprin.size]*(-10e6))
		elif methods['tracetail'] == 'firing rate' and ( methods["peakDetec"] or methods["R2"] ):
			plt.ylabel("Firing Rate (ms$^{-1}$)", fontsize=16)
			plt.plot(nppoints,module[tvl:tvr]/np.sum(kernel),"k--")
			hight = np.max(module[tvl:tvr]/np.sum(kernel))
			for mark in spbinmark:
				if mark[0] < tvl or mark[0] > tvr: continue
				if mark[1] > 0:
					plt.plot([mark[0],mark[0]],[0,hight],"k--")
		plt.xlabel("time (ms)", fontsize=16)


	
	if (methods["netFFT"] or methods["nrnFFT"]) and methods['gui']:
		plt.figure(2)
		if methods["netFFT"] and methods["nrnFFT"]:
			pl=plt.subplot(211)
		elif methods["netFFT"]:
			pl=plt.subplot(111)
		if methods["netFFT"]:
			plt.title("Network spectrum")
			plt.bar(specX[1:],fft[1:pnum],0.75)
		if methods["netFFT"] and methods["nrnFFT"]:
			plt.subplot(212,sharex=pl)
		elif methods["nrnFFT"]:
			plt.subplot(111)
		if methods["nrnFFT"]:
			plt.title("Neuronal spectrum")
			plt.bar(specX[1:],specN[1:],0.75)

	#plt.subplot(313,sharex=p)
	#specX =np.arange(0.0,tstop+h.dt,h.dt)
	#specX *= 1000.0/tstop/h.dt
	#pnum = specX.size/2
	#plt.title("Voltage spectrum")
	#plt.plot(specX[1:pnum],specV[1:pnum])
	#plt.xlim(0,200)
	
	if 10 < methods["netISI"] <= 3000 and sum(netisi) > 0: netisi /= sum(netisi)
	if 10 < methods["nrnISI"] <= 3000 and sum(isi) > 0: isi /= sum(isi)
	if (10 < methods["netISI"] <= 3000 or 10 < methods["nrnISI"] <= 3000) and methods['gui']:
		plt.figure(3)
		if 10 < methods["netISI"] <= 3000 and 10 < methods["nrnISI"] <= 3000:
			pl=plt.subplot(211)
		elif 10 < methods["netISI"] <= 3000 :
			plt.subplot(111)
		if 10 < methods["netISI"] <= 3000: 
			plt.title("Network ISI")
			plt.ylabel("Normalized number of interspike intervals", fontsize=16)
			plt.bar(np.arange(methods["netISI"]),netisi,0.75)
		if 10 < methods["netISI"] <= 3000 and 10 < methods["nrnISI"] <= 3000:
			plt.subplot(212)#,sharex=pl)
		elif 10 < methods["nrnISI"] <= 3000:
			plt.subplot(111)
		if 10 < methods["nrnISI"] <= 3000:
			plt.ylabel("Normalized number of interspike intervals", fontsize=16)
			plt.title("Neuronal ISI")
			plt.bar(np.arange(methods["nrnISI"]),isi,0.75,color='k')
			plt.xlim(0,methods["nrnISI"])
			plt.xlabel("Interspike intervals (ms)", fontsize=16)
	
	if methods['traceView'] and methods['gui']:
		zooly = plt.figure(4)
		zooly.canvas.mpl_connect('button_press_event', zoolyclickevent)
		zooly.canvas.mpl_connect('key_press_event', zoolykeyevent)
		moddy = plt.figure(5)
		faxi = plt.subplot2grid((6,10),(0,0),colspan=4,rowspan=2)
		vaxi = plt.subplot2grid((6,10),(2,0),colspan=4,sharex=faxi)
		uaxi = plt.subplot2grid((6,10),(3,0),colspan=4,sharex=faxi)
		gaxi = plt.subplot2grid((6,10),(4,0),colspan=4,sharex=faxi)
		iaxi = plt.subplot2grid((6,10),(5,0),colspan=4,sharex=faxi)
		#saxi = plt.subplot2grid((6,10),(5,0),colspan=4,sharex=faxi)
		naxi = plt.subplot2grid((6,10),(0,5),colspan=6,rowspan=6)
		moddy.canvas.mpl_connect('key_press_event', moddykeyevent)

	if methods['FPcurve'] and methods['gui']:
		plt.figure(6)
		f  = np.array([ [n.izh.F,n.spks.size]       for n in neurons])/tstop*1000. #because ms
		#f = np.sort(f, axis=0)
		plt.plot(f[:,0] , f[:,1],"k^",ms=9)
		xrng=plt.xlim()
		plt.plot(np.array(xrng),[1000./netpermean,1000./netpermean],"k--")

	if methods['GPcurve'] and methods['gui']:
		plt.figure(7)
		f  = np.array([ [n.gsynscale,n.spks.size]       for n in neurons])
		#f = np.sort(f, axis=0)
		plt.plot(f[:,0] ,f[:,1],"k+")
			
	if methods['sycleprop'] and methods['gui']:
		plt.figure(8)
		polarax = plt.subplot(111, polar=True)
		#bars = polarax.bar(phydist[:,1], phydist[:,0], width=0.25, bottom=0.0)
		#np.histogram(phydist[:,0], bins=180, weights=phydist[:,1])
		#polarax.hist(phydist[:,0], bins=36, weights=phydist[:,1])
		polarax.bar(phyhistbins[:-1],phyhist,width=phyhistbins[1]-phyhistbins[0],bottom=0)
		#DB>>
		plt.figure(9)
		plt.bar(phyhistbins[:-1],phyhist,width=phyhistbins[1]-phyhistbins[0],bottom=0)
		#<<DB
	if methods['Gtot-dist'] and methods['gui']:
		plt.figure(10)
		plt.bar(gskbins[:-1],gskhist,width=gskbins[1]-gskbins[0],color="b")
		#plt.hist(gsk,bins=ncell/50)

	if methods['corelog']:
		with open(methods['corelog']+".csv","a") as fd:
			
			#0:Type,1:a,2:b,3:c,4:d,5:(U),6:(V),7:(F),8:(Gscale),9:(Iapp),
			#10:(weight),11:(delay),12:Istdev,13:mean np,14:spc,15:R2,16:mean netp
			fd.write("%s,%g,%g,%g,%g,"%(ntype,a,b,c,d))
			if type(U) is float or type(U) is int:
				fd.write("%g,"%(U))
			elif type(U) is tuple:
				fd.write("%g:%g,"%U)
			elif type(U) is str:
				fd.write("%s,"%U)
			if type(V) is float or type(V) is int:
				fd.write("%g,"%(V))
			elif type(V) is tuple:
				fd.write("%g:%g,"%V)
			elif type(V) is str:
				fd.write("%s,"%V)

			if type(F) is float or type(F) is int:
				fd.write("%g,"%(F))
			else:
				fd.write("%g:%g,"%F)
			if type(gsynscale) is float or type(gsynscale) is int:
				fd.write("%g,"%(gsynscale))
			else:
				fd.write("%g:%g,"%gsynscale)
			if type(Iapp) is float:
				fd.write("%g,"%(Iapp))
			else:
				fd.write("%g:%g,"%Iapp)
			if type(weight) is float:
				fd.write("%g,"%(weight))
			else:
				fd.write("%g:%g,"%weight)
			if type(delay) is float:
				fd.write("%g,"%(delay))
			else:
				fd.write("%g:%g,"%delay)
			if pcnt > 1:
				fd.write("%g,%g,"%(Istdev,pmean/pcnt))
			else:
				fd.write("%g,x,"%(Istdev))
			if methods["R2"]:
				if Rcnt > 0 :
					fd.write("%g,%g,%g,"%(spc,R2,netpermean))
				else:
					fd.write("%g,x,%g,"%(spc,netpermean))
			elif methods["peakDetec"]:
				fd.write("%g,x,x,"%(spc))
			else:
				fd.write("x,x,x,")
			if methods['IGcurve']:
				fd.write("IGF,%d,"%(ncell))
				for n in neurons:
					fd.write("%g:%g:%g,"%(n.innp.mean,n.gsynscale,n.izh.F))
			if methods['Connectom']:
				fd.write("CONNECTOM,%d,"%(len(connections)))
				for n in connections:
					fd.write("%d:%d:%g:%g,"%(n[1],n[2],n[0].weight[0],n[0].delay))
			if methods["netFFT"] or methods["nrnFFT"]:
				fftkernel = np.arange(-methods['fftkernel']*3,methods['fftkernel']*3,1.)
				fftkernel = np.exp(fftkernel**2/(-methods['fftkernel']**2))
			if methods["netFFT"]:
				fftmodule = np.convolve(fft[1:pnum],fftkernel)
				fftmodule = fftmodule[fftkernel.size/2:1-fftkernel.size/2]
				fftmax = (np.diff(np.sign(np.diff(fftmodule))) < 0).nonzero()[0] + 1
				fftmin = (np.diff(np.sign(np.diff(fftmodule))) > 0).nonzero()[0] + 1
				fd.write("netFFTmax,%d,"%fftmax.size)
				for fftm in fftmax:
					fd.write("%g:%g,"%(specX[fftm+1],fft[fftm+1]))
				fd.write("netFFTmin,%d,"%fftmin.size)
				for fftm in fftmin:
					fd.write("%g:%g,"%(specX[fftm+1],fft[fftm+1]))
			if methods["nrnFFT"]:
				fftmodule = np.convolve(specN[1:],fftkernel)
				fftmodule = fftmodule[fftkernel.size/2:1-fftkernel.size/2]
				fftmax = (np.diff(np.sign(np.diff(fftmodule))) < 0).nonzero()[0] + 1
				fftmin = (np.diff(np.sign(np.diff(fftmodule))) > 0).nonzero()[0] + 1
				fd.write("nrnFFTmax,%d,"%fftmax.size)
				for fftm in fftmax:
					fd.write("%g:%g,"%(specX[fftm+1],specN[fftm+1]))
				fd.write("nrnFFTmin,%d,"%fftmin.size)
				for fftm in fftmin:
					fd.write("%g:%g,"%(specX[fftm+1],specN[fftm+1]))
			if 10 < methods["netISI"] <= 3000 or 10 < methods["nrnISI"] <= 3000:
				isikernel = np.arange(-methods['isikernel']*3,methods['isikernel']*3,1.)
				isikernel = np.exp(isikernel**2/(-methods['isikernel']**2))
			if 10 < methods["netISI"] <= 3000 :
				if sum(netisi) > 0:
					isimodule = np.convolve(netisi,isikernel)
					isimodule = isimodule[isikernel.size/2:1-isikernel.size/2]
					isimax = (np.diff(np.sign(np.diff(isimodule))) < 0).nonzero()[0] + 1
					isimin = (np.diff(np.sign(np.diff(isimodule))) > 0).nonzero()[0] + 1
					fd.write("netISImax,%d,"%isimax.size)
					for isim in isimax:
						fd.write("%g:%g,"%(isim-2,netisi[isim-2]))
					fd.write("netISImin,%d,"%isimin.size)
					for isim in isimin:
						fd.write("%g:%g,"%(isim-2,netisi[isim-2]))
				else:
					fd.write("netISImax,0,")
					fd.write("netISImin,0,")
			if 10 < methods["nrnISI"] <= 3000 :
				if sum(isi) > 0:
					isimodule = np.convolve(isi,isikernel)
					isimodule = isimodule[isikernel.size/2:1-isikernel.size/2]
					isimax = (np.diff(np.sign(np.diff(isimodule))) < 0).nonzero()[0] + 1
					isimin = (np.diff(np.sign(np.diff(isimodule))) > 0).nonzero()[0] + 1
					fd.write("nrnISImax,%d,"%isimax.size)
					for isim in isimax:
						fd.write("%g:%g,"%(isim-2,isi[isim-2]))
					fd.write("nrnISImin,%d,"%isimin.size)
					for isim in isimin:
						fd.write("%g:%g,"%(isim-2,isi[isim-2]))
				else:
					fd.write("nrnISImax,0,")
					fd.write("nrnISImin,0,")
			if methods['coreindex']:
				fd.write("%g,"%coreindex)
			if methods['F-Spikerate']:
				fd.write("FNC,%d,%g"%(ncell,tstop))
				for n in neurons:
					fd.write(",%f:%d"%(n.izh.F,n.spks.size))
			if methods['G-Spikerate']:
				fd.write(",GSNC,%d"%(ncell))
				for n in neurons:
					fd.write(",%f:%d"%(n.gsynscale,n.spks.size))
			if methods['TaS']:
				fd.write(",TaS,%g"%TaSindex)
			if methods['lastspktrg']:
				fd.write(",LST,%d"%lastspktrg)
			fd.write(",SYN:%g:%g:%g:%d"%(ST1,ST2,SE,int(methods['taunorm'])) )
			if methods['external'] and methods['extprop']:
				fd.write(",EXTPROP,%g,%g,%g,%d"%(spprop,methods['external'][0],methods['external'][1],methods['external'][2]))
			#Syn stat
			allconv = np.array( [ c[0].weight[0] for c in connections ] )
			fd.write(",STAT:G,%g,%g,%g,%g"%(np.mean(allconv),np.std(allconv),np.min(allconv),np.max(allconv)) )
			allconv = np.array( [ c[0].delay for c in connections ] )
			fd.write(",STAT:D,%g,%g,%g,%g"%(np.mean(allconv),np.std(allconv),np.min(allconv),np.max(allconv)) )
			# TSscaler
			if synscaler != None:
				if type(synscaler) is float or type(synscaler) is int:
					fd.write(",GScaler,{}".format(synscaler))
				elif type(synscaler) is list or type(synscaler) is tuple:
					if len(synscaler) >= 2:
						fd.write(",GScaler,{}:{}".format(synscaler[0],synscaler[1]))
					else:
						fd.write(",GScaler,{}".format(synscaler[0]))
				else:
					fd.write(",GScaler,None")
			else: fd.write(",GScaler,None")
			if methods['popfr']:fd.write(",MFR,%g"%popfr)
			if methods['sycleprop']:
				fd.write(",Cyledist,%d,%g"%(phyhist.shape[0],phyhistbins[1]-phyhistbins[0]))
				for p,n in zip(phyhistbins[:-1],phyhist):
					fd.write(",%g:%g"%(p,n))
			if methods['Gtot-rec']:
				fd.write(",Gtot-rec,%d"%(ncell))
				for n in neurons:
					fd.write(",%g:%g"%(n.gsynscale,n.gtotal))
			if methods['Gtot-stat']:
				agtot = np.array([ n.gtotal/n.concnt for n in neurons ])
				mgtot = np.mean(agtot)
				sgtot = np.std(agtot)
				fd.write(",Gtot-stat,%g,%g,%g"%(mgtot,sgtot,sgtot/mgtot))
			fd.write("\n")

	if methods['gui']:
		if methods['gif']:
			plt.savefig(methods['gif'])
		else:
			plt.show()
	if not methods['noexit']:
		sys.exit(0)
