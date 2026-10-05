#!/usr/bin/env python3
"""Waypoint approach: dynamic block-to-history occupation matching.

A* chooses a sequence of adjacent rectangular cells. The single integrator
x_dot=u, ||u||<=u_max follows projected entry waypoints. Physical residence
Times update eta and the objective exactly. Holding is realized as a uniform
within-cell waypoint patrol instead of stationary u=0 whenever possible.
"""
from __future__ import annotations
import argparse, csv, heapq, json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

rows=16; cols=16; n=rows*cols
rho=1-1e-7; blocks_default=30_000; horizon_default=500; random_seed=13
rho_endpoint=0.7; distance_weight=6.; endpoint_weight=2.
travel_edge_weight=0.; path_error_weight=4.
astar_heuristic_weight=.10
objective_tolerance=1e-15; tolerance=1e-12
# Dynamics


u_max=.35
self_loop_time= 0.01
entry_fraction=.02
maximum_block_time_default=0.0  # 0 disables physical-time cap
patrol_margin_fraction=.08
cw=1/cols; ch=1/rows

def idx(r,c): return r*cols+c
def rc(i): return divmod(i,cols)
centers=np.array([((c+.5)/cols,(r+.5)/rows) for r in range(rows) for c in range(cols)])
neighbors=[]
for r in range(rows):
 for c in range(cols):
  q=[]
  if r>0:q.append(idx(r-1,c))
  if r<rows-1:q.append(idx(r+1,c))
  if c>0:q.append(idx(r,c-1))
  if c<cols-1:q.append(idx(r,c+1))
  neighbors.append(q)

def gaussian(p,m,C):
 d=p-m; return np.exp(-.5*np.einsum('ni,ij,nj->n',d,np.linalg.inv(C),d))
target=(.5*gaussian(centers,np.array([.25,.75]),np.array([[.012,0],[0,.018]]))
       +.35*gaussian(centers,np.array([.75,.30]),np.array([[.020,.008],[.008,.015]]))
       +.15*gaussian(centers,np.array([.70,.80]),np.array([[.008,0],[0,.008]])))
target=.98*target+.02; target/=target.sum()
def objective(eta): return float(np.sum((eta-target)**2))
def first_variation(eta): return 2*(eta-target)
def manhattan(a,b):
 ar,ac=rc(a);br,bc=rc(b);return abs(ar-br)+abs(ac-bc)
def bounds(i):
 r,c=rc(i);return c*cw,(c+1)*cw,r*ch,(r+1)*ch
def cell_of(x):
 xx=min(max(float(x[0]),0.),np.nextafter(1.,0.));yy=min(max(float(x[1]),0.),np.nextafter(1.,0.))
 return idx(min(rows-1,int(yy*rows)),min(cols-1,int(xx*cols)))

@dataclass
class Choice: node:int; endpoint_error:float; score:float
@dataclass
class Record:
 block:int;mode:str;total_time:float;block_time:float;travel_time:float;patrol_time:float
 route_cells:int;patrol_waypoints:int;destination:int;objective_before:float;objective_after:float
 history_gap:float;block_error:float;ratio:float;contracted:bool;astar_cost:float

def choose_destination(current,g,H):
 s=float(g.min());e=g-s;cand=np.flatnonzero(e<=rho_endpoint*H+tolerance)
 if not len(cand):cand=np.flatnonzero(np.isclose(g,s,atol=1e-15,rtol=0))
 r0,c0=rc(current);rr=cand//cols;cc=cand%cols;d=(np.abs(rr-r0)+np.abs(cc-c0)).astype(float)
 score=distance_weight*d/max(1.,rows+cols-2)+endpoint_weight*e[cand]/max(H,tolerance)
 j=int(np.argmin(score));return Choice(int(cand[j]),float(e[cand[j]]),float(score[j]))

def astar(start,goal,g,x,speed):
 """Plan a cell route using first-order physical travel time.

 Each state is (cell, incoming_cell). Keeping the incoming cell distinguishes
 different entry sides. The stored continuous waypoint is propagated along the
 best label for that state.

 The edge cost uses the exact residence-time split returned by transition():
   t_u * max(0, g[u]-g[goal]) + t_v * max(0, g[v]-g[goal]).

 Manhattan distance is retained as a lightly weighted, dimensionally scaled
 search heuristic. It guides the search but does not dominate the physical cost.
 """
 if start==goal:return [start],0.
 if speed<=0:raise ValueError('speed must be positive')
 gv=float(g[goal]);gscale=max(float(g.max()-g.min()),1e-15)
 time_per_cell=min(cw,ch)/speed
 def heuristic(cell):
  return astar_heuristic_weight*manhattan(cell,goal)*time_per_cell*gscale
 
 start_state=(int(start),-1)
 best={start_state:0.};pred={};position={start_state:np.asarray(x,float).copy()}
 q=[(heuristic(start),0.,start_state)]
 goal_state=None
 while q:
  _,cost,state=heapq.heappop(q);u,_=state
  if cost>best.get(state,np.inf)+tolerance:continue
  if u==goal:goal_state=state;break
  for v in neighbors[u]:
   v=int(v);next_state=(v,u)
   x_next,t_u,t_v=transition(position[state],u,v,speed)
   edge_cost=(t_u*max(0.,float(g[u])-gv)
              +t_v*max(0.,float(g[v])-gv))
   z=cost+edge_cost
   if z<best.get(next_state,np.inf)-tolerance:
    best[next_state]=z;pred[next_state]=state;position[next_state]=x_next
    heapq.heappush(q,(z+heuristic(v),z,next_state))
 if goal_state is None:raise RuntimeError('No path')
 states=[goal_state]
 while states[-1]!=start_state:states.append(pred[states[-1]])
 return [state[0] for state in states[::-1]],float(best[goal_state])

def entry_waypoint(x,current,nxt):
 r0,c0=rc(current);r1,c1=rc(nxt);e=entry_fraction*min(cw,ch);y=np.array(x,float)
 if c1==c0+1:y[0]=(c0+1)*cw+e;y[1]=np.clip(y[1],r1*ch+e,(r1+1)*ch-e)
 elif c1==c0-1:y[0]=c0*cw-e;y[1]=np.clip(y[1],r1*ch+e,(r1+1)*ch-e)
 elif r1==r0+1:y[1]=(r0+1)*ch+e;y[0]=np.clip(y[0],c1*cw+e,(c1+1)*cw-e)
 elif r1==r0-1:y[1]=r0*ch-e;y[0]=np.clip(y[0],c1*cw+e,(c1+1)*cw-e)
 else:raise ValueError('Nonadjacent cells')
 return y

def transition(x,current,nxt,speed):
 y=entry_waypoint(x,current,nxt);d=y-x;L=float(np.linalg.norm(d))
 if L<=1e-15:return y,0.,0.
 r0,c0=rc(current);r1,c1=rc(nxt)
 alpha=((max(c0,c1)*cw-x[0])/d[0]) if c1!=c0 else ((max(r0,r1)*ch-x[1])/d[1])
 alpha=float(np.clip(alpha,0.,1.));tau=L/speed
 return y,alpha*tau,(1-alpha)*tau

def realize_route(x,path,speed):
 times=np.zeros(n);points=[];travel=0.;y=np.array(x,float);u=int(path[0])
 for v in path[1:]:
  z,t0,t1=transition(y,u,int(v),speed);times[u]+=t0;times[int(v)]+=t1
  travel+=t0+t1;y=z;u=int(v);points.append(y.copy())
 return y,times,travel,points

def required_patrol_time(transit,g,s,destination,H):
 travel=float(transit.sum());threshold=rho*H;E=float(np.dot(g-s,transit));e=float(g[destination]-s)
 if travel>0 and E/travel<=threshold+tolerance:return 0.
 den=threshold-e
 if den<=0:return np.inf
 return max(0.,(E-threshold*travel)/den)

def uniform_patrol(x,cell,duration,speed,rng):
 """Use uniform interior waypoints until exactly duration is consumed.

 Every segment lies in the same rectangular cell because rectangles are convex.
 The final segment is truncated so total patrol time equals `duration` exactly.
 """
 if duration<=0:return np.array(x,float),[],0
 xmin,xmax,ymin,ymax=bounds(cell);m=patrol_margin_fraction*min(cw,ch)
 lo=np.array([xmin+m,ymin+m]);hi=np.array([xmax-m,ymax-m]);p=np.array(x,float)
 # Ensure the starting point is safely interior before random patrol.
 p=np.minimum(np.maximum(p,lo),hi);points=[];remaining=float(duration);count=0
 while remaining>tolerance:
  q=rng.uniform(lo,hi);d=q-p;L=float(np.linalg.norm(d))
  if L<=1e-15:continue
  tau=L/speed
  if tau<=remaining+tolerance:
   p=q;remaining-=tau;points.append(p.copy())
  else:
   p=p+(remaining/tau)*d;remaining=0.;points.append(p.copy())
  count+=1
 return p,points,count

def run(blocks,max_horizon,max_block_time,seed,speed,loop_time,verbose):
 rng=np.random.default_rng(seed);current=int(rng.integers(n));x=centers[current].copy()
 occupation=np.zeros(n);occupation[current]=loop_time;T=loop_time
 trace=[x.copy()];records=[];times=[T];objectives=[objective(occupation/T)]
 for k in range(blocks):
  eta=occupation/T;before=objective(eta)
  if before<=objective_tolerance:break
  g=first_variation(eta);s=float(g.min());H=float(g@eta-s)
  if H<=tolerance:break
  choice=choose_destination(current,g,H);path,cost=astar(current,choice.node,g,x,speed)
  feasible=max_horizon is None or len(path)-1<=max_horizon;mode='contract'
  if feasible:
   endpoint,transit,travel,route_points=realize_route(x,path,speed)
   patrol=required_patrol_time(transit,g,s,choice.node,H)
   if len(path)==1:patrol=max(patrol,loop_time)
   feasible=np.isfinite(patrol) and (max_block_time is None or travel+patrol<=max_block_time+tolerance)
  if not feasible:
   mode='self_loop_uniform_patrol';choice=Choice(current,float(g[current]-s),0.);path=[current];cost=0.
   endpoint=x.copy();transit=np.zeros(n);travel=0.;route_points=[]
   patrol=loop_time if max_block_time is None else min(loop_time,max_block_time)
  # Replace all stationary holding/self-loop time by within-cell uniform waypoint motion.
  endpoint,patrol_points,patrol_count=uniform_patrol(endpoint,choice.node,patrol,speed,rng)
  block_times=transit.copy();block_times[choice.node]+=patrol;block_time=float(block_times.sum())
  p=block_times/block_time;err=float(g@p-s);contracted=err<=rho*H+tolerance
  occupation+=block_times;T+=block_time;x=endpoint;current=choice.node
  trace.extend(route_points);trace.extend(patrol_points);after=objective(occupation/T)
  records.append(Record(k,mode,T,block_time,travel,patrol,len(path),patrol_count,current,before,after,H,err,err/H,contracted,cost))
  times.append(T);objectives.append(after)
  if verbose>0 and (k+1)%verbose==0:
   print(f'block={k+1:7d} T={T:12.5f} G={after:.6e} route={len(path):4d} patrol={patrol:.5f} waypoints={patrol_count:4d}',flush=True)
 return occupation/T,np.asarray(trace),records,np.asarray(times),np.asarray(objectives)

def save(out,eta,trace,records,times,objectives):
 out.mkdir(parents=True,exist_ok=True)
 np.save(out/'trajectory_positions.npy',trace);np.save(out/'final_empirical_measure.npy',eta);np.save(out/'target.npy',target);np.save(out/'time_history.npy',times);np.save(out/'objective_history.npy',objectives)
 if records:
  with (out/'block_history.csv').open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=asdict(records[0]).keys());w.writeheader()
   for z in records:w.writerow(asdict(z))
 fig,ax=plt.subplots(figsize=(8,5));ax.loglog(times,objectives);ax.set(xlabel='Physical time T',ylabel=r'$G(\eta_T)$');ax.grid(True,which='both',alpha=.3);fig.tight_layout();fig.savefig(out/'objective_history.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,8));im=ax.imshow(target.reshape(rows,cols),origin='lower',extent=(0,1,0,1),cmap='viridis')
 if len(trace)>1:ax.plot(trace[:,0],trace[:,1],color='white',lw=.35,alpha=.72)
 ax.scatter(trace[0,0],trace[0,1],c='lime',edgecolors='black',s=35,label='start');ax.scatter(trace[-1,0],trace[-1,1],c='red',edgecolors='black',s=35,label='finish')
 ax.set(xlim=(0,1),ylim=(0,1),aspect='equal',title='Waypoint approach with uniform within-cell patrol');ax.legend();fig.colorbar(im,ax=ax,label='Target cell mass');fig.tight_layout();fig.savefig(out/'trajectory_over_target.png',dpi=200);plt.close(fig)

def main():
 p=argparse.ArgumentParser();p.add_argument('--blocks',type=int,default=blocks_default);p.add_argument('--max-horizon',type=int,default=horizon_default);p.add_argument('--max-block-time',type=float,default=0.)
 p.add_argument('--speed',type=float,default=u_max);p.add_argument('--self-loop-time',type=float,default=self_loop_time);p.add_argument('--seed',type=int,default=13);p.add_argument('--verbose-every',type=int,default=500);p.add_argument('--output-dir',type=Path,default=Path('block_to_history_A_star'));a=p.parse_args()
 cap=None if a.max_horizon==0 else a.max_horizon;time_cap=None if a.max_block_time==0 else a.max_block_time
 eta,trace,recs,times,objs=run(a.blocks,cap,time_cap,a.seed,a.speed,a.self_loop_time,a.verbose_every);save(a.output_dir,eta,trace,recs,times,objs)
 summary={'name':'waypoint approach','occupation':'exact physical residence time','holding':'uniform within-cell waypoint patrol','blocks':len(recs),'physical_time':float(times[-1]),'final_objective':objective(eta),'contracted_blocks':sum(r.contracted for r in recs),'fallback_patrol_blocks':sum(r.mode.startswith('self_loop') for r in recs)}
 (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
