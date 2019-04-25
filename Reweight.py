#! /usr/bin/env python3

### Create a reweighted FES ###

import numpy as np
import pandas as pd
import linecache
import sys
import subprocess
import argparse

### most togled parameters ###
print_stride=10000
prefix=''
#prefix='bck.0.'
bias_column=3
rct_column=4
jarz_column=5

### parser ###
parser = argparse.ArgumentParser(description='reweight')
parser.add_argument('-c',dest='cv_column',type=int,default=2,required=False,help='cv column to be used from colvar file')
parser.add_argument('-t',dest='transient',type=int,default=0,required=False,help='transient time to be skipped')
parser.add_argument('-j',dest='jarz',default=False,required=False,action='store_true',help='use Jarzynski-like weights')
args = parser.parse_args()
cv_column=args.cv_column
transient=args.transient
jarz_on=args.jarz

### print some info ###
print(' Some common parameters:')
print('  - reweghting on CV: %d'%cv_column)
print('  - print_stride = %d'%print_stride)
if jarz_on:
  print('  - using jarzinski-like weights')
if transient:
  print('  - transient = %d'%transient)
if prefix:
  print('  - prefix: '+prefix)

### input stuff ###
Kb=0.0083144621 #kj/mol
temp=300
#- grid -
grid_min=-np.pi
grid_max=np.pi
grid_bin=100
periodic=True
#- files in -
cv_header=5
file_ext='.data'
colvar_file=prefix+'Colvar'
#- files out -
FES_file='FES_rew-'+str(cv_column)
if jarz_on:
  FES_file='jarz-'+FES_file
FES_head='cv_bin  prat_unif_FES'
if transient>0:
  tran_dir='tran'+str(transient)

### setup ###
beta=1/(Kb*temp)
#- fix grid -
edge_shift=0
if periodic:
  edge_shift=(grid_max-grid_min)/grid_bin
cv_grid=np.linspace(grid_min,grid_max-edge_shift,grid_bin)
#- get walkers -
cmd=subprocess.Popen('ls '+colvar_file+'* |wc -l',shell=True,stdout=subprocess.PIPE)
output=cmd.communicate()
n_walkers=int(output[0])
print('  - n_walkers found: %d'%n_walkers)
if n_walkers==1:
  colvar_files=[colvar_file+file_ext]
else:
  colvar_files=[colvar_file+'.'+str(n)+file_ext for n in range(n_walkers)]
#- get cv data -
if cv_header<0:
  cv_header=1
  while linecache.getline(colvar_files[0],cv_header).split()[0]=='#!':
    cv_header+=1
  cv_header-=1
columns=[0,cv_column,bias_column,rct_column]
if jarz_on:
  columns=[0,cv_column,bias_column,rct_column,jarz_column]
data=pd.read_table(colvar_files[0],dtype=float,sep='\s+',skiprows=cv_header,header=None,usecols=columns)
time=np.array(data.ix[:,0])
cv_=np.array(data.ix[:,cv_column])
V_=np.array(data.ix[:,bias_column])
rc_=np.array(data.ix[:,rct_column])
if jarz_on:
  jarz_=np.array(data.ix[:,jarz_column])
del columns[0]
for n in range(1,n_walkers):
  data=pd.read_table(colvar_files[n],dtype=float,sep='\s+',skiprows=cv_header,header=None,usecols=columns)
  cv_=np.column_stack((cv_,data.ix[:,cv_column]))
  V_=np.column_stack((V_,data.ix[:,bias_column]))
  rc_=np.column_stack((rc_,data.ix[:,rct_column]))
  if jarz_on:
    jarz_=np.column_stack((jarz_,data.ix[:,jarz_column]))
del data
print(' --> all data loaded <-- ')

# - prepare output files -
create_dir='bck.meup.sh {0}; mkdir -p {0}'
current_FES=FES_file+'/'+FES_file+'.t-%d'+file_ext
if transient>0:
  FES_file=tran_dir+'/'+FES_file
  current_FES=tran_dir+'/'+current_FES
cmd=subprocess.Popen(create_dir.format(FES_file),shell=True)
cmd.wait()

#- initialize some variables -
full_hist=np.zeros(grid_bin)
jarz_t=0

# - set transient -
skip=0
for t in range(len(time)):
  if time[t]>=transient:
    break
  skip+=1

# - run the thing -
for t in range(skip,len(time)):
  if jarz_on:
    jarz_t=jarz_[t]
  hist_t,e=np.histogram(cv_[t],bins=grid_bin,range=(grid_min,grid_max),weights=np.exp(beta*(V_[t]-rc_[t]-jarz_t)),density=False)
  full_hist+=hist_t
  if time[t]%print_stride==0:
    prat_unif_fes=-1/beta*np.log(full_hist)
    prat_unif_fes-=min(prat_unif_fes)
    np.savetxt(current_FES%time[t],np.c_[cv_grid,prat_unif_fes],header=FES_head,fmt='%.6f')

# - be sure to print also last fes -
if time[-1]%print_stride!=0:
    print('  >> adding last FES... ')
    prat_unif_fes=-1/beta*np.log(full_hist)
    prat_unif_fes-=min(prat_unif_fes)
    np.savetxt(current_FES%time[t],np.c_[cv_grid,prat_unif_fes],header=FES_head,fmt='%.6f')

# - get final FES -
backup='bck.meup.sh '+FES_file+file_ext
copy='cp '+current_FES%time[-1]+' '+FES_file+file_ext
cmd=subprocess.Popen(backup+';'+copy,shell=True)
cmd.wait()

