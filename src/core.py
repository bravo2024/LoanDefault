"""core.py — Loan default with competing risks (prepayment vs default).
NOT generic binary classification. References: Fine & Gray (1999)."""
from __future__ import annotations
import numpy as np
def cumulative_incidence(times, events, event_type=1, max_time=365*5):
    """Cumulative incidence function for a specific event type (competing risks)."""
    t=np.asarray(times,float);e=np.asarray(events,int)
    out=[];n_at_risk=len(t)
    for day in range(1, max_time+1, 30):
        risk=(t>=day)&~((t<day+30)&(e==event_type))
        events_in_window=((t>=day)&(t<day+30)&(e==event_type)).sum()
        if n_at_risk>0:
            haz=events_in_window/n_at_risk;n_at_risk-=risk.sum()
        out.append(float(haz))
    return np.array(out)
def concordance_index(times, events, risk_scores):
    t,e,r=np.asarray(times,float),np.asarray(events,int),np.asarray(risk_scores,float);n=len(t);c,p=0,0
    for i in range(n):
        if e[i]!=1:continue
        for j in range(n):
            if t[j]>t[i]:
                p+=1
                if r[i]>r[j]:c+=1
                elif r[i]==r[j]:c+=0.5
    return c/p if p>0 else 0.5
