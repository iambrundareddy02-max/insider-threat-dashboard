import os, math, threading, time, webbrowser
import numpy as np
import pandas as pd
import joblib
from scipy.stats import chi2
from flask import Flask, jsonify, request, Response

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "data", "processed")
STATES = ["Low", "Medium", "High", "Critical"]

# If auto-detection picks a wrong column, set it here, e.g. {"login": "logon_count"}
MANUAL_MAP = {}

# ---------------- load data and model ----------------
pq = os.path.join(PROC, "features.parquet")
feat = pd.read_parquet(pq) if os.path.exists(pq) else pd.read_csv(os.path.join(PROC, "features.csv"))
risk = pd.read_csv(os.path.join(PROC, "risk_scores.csv"))
bundle = joblib.load(os.path.join(BASE, "models", "model.joblib"))
rf, iso, FEATS = bundle["rf"], bundle["iso"], bundle["features"]


def find(cands):
    for c in feat.columns:
        if c.lower() in cands:
            return c
    return None


ucol, dcol = find(["user", "user_id"]), find(["date", "day"])
if not (ucol and dcol):
    raise SystemExit("Cannot find user/date columns. Columns are: " + str(list(feat.columns)))

feat["_u"] = feat[ucol].astype(str)
feat["_d"] = pd.to_datetime(feat[dcol]).dt.strftime("%Y-%m-%d")
r2 = risk.rename(columns={"user": "_ru", "date": "_rd", "label": "_lbl",
                          "risk_score": "_risk", "severity": "_sev", "reasons": "_why"})
r2["_ru"] = r2["_ru"].astype(str)
r2["_rd"] = r2["_rd"].astype(str)
data = (feat.merge(r2, left_on=["_u", "_d"], right_on=["_ru", "_rd"], how="inner")
            .sort_values("_d").reset_index(drop=True))
if data.empty:
    raise SystemExit("features file and risk_scores.csv do not match. Re-run: py train_model.py")

X_all = data[FEATS].fillna(0)
rf_all = rf.predict_proba(X_all)[:, 1]
iso_all = -iso.score_samples(X_all)
RF_LO, RF_HI, IS_LO, IS_HI = rf_all.min(), rf_all.max(), iso_all.min(), iso_all.max()


def nz(v, lo, hi):
    return float(np.clip((v - lo) / (hi - lo + 1e-9), 0, 1))


def model_score(rp, ip):
    return 100 * (0.7 * nz(rp, RF_LO, RF_HI) + 0.3 * nz(ip, IS_LO, IS_HI))


def fl(x):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else round(x, 3)


def floor_std(s, m):
    s = float(s) if s == s else 0.0
    return max(s, 0.25 * abs(float(m)), 0.5)


def sev_of(score):
    return "Low" if score < 25 else "Medium" if score < 50 else "High" if score < 75 else "Critical"


# ---------------- map the 4 input fields to dataset columns ----------------
ACT = {"login": ("Login Count", ["logon", "login"]),
       "file": ("File Access", ["file"]),
       "usb": ("USB Activity", ["usb", "device"]),
       "email": ("Email Count", ["email", "mail"])}
BAD = ["zscore", "z_", "_z", "deviation", "_dev", "base", "peer", "ratio", "avg", "std",
       "after", "weekend", "external", "attach", "size", "hour", "mean"]


def pick(keys):
    c = [f for f in FEATS if any(k in f.lower() for k in keys)
         and not any(b in f.lower() for b in BAD)]
    return min(c, key=len) if c else None


MAP = {k: (MANUAL_MAP[k] if MANUAL_MAP.get(k) in FEATS else pick(v[1])) for k, v in ACT.items()}
print("Detected activity columns:", MAP)

G = {k: (float(data[c].mean()), floor_std(data[c].std(), data[c].mean()))
     for k, c in MAP.items() if c}
FMED = data[FEATS].median()
FM = data[FEATS].mean()
FS = data[FEATS].std().replace(0, 1).fillna(1)


# ---------------- Markov chain on severity states ----------------
def trans_counts(seq):
    C = np.zeros((4, 4))
    idx = [STATES.index(s) for s in seq if s in STATES]
    for a, b in zip(idx[:-1], idx[1:]):
        C[a, b] += 1
    return C


GC = sum(trans_counts(g["_sev"].tolist()) for _, g in data.groupby("_u"))
GP = (GC + 0.5) / (GC.sum(axis=1, keepdims=True) + 2.0)

# ---------------- correlation (global) ----------------
cc = [(ACT[k][0], MAP[k]) for k in ACT if MAP[k]]
cm = data[[c for _, c in cc] + ["_risk"]].corr().fillna(0).round(2)
CORR = {"labels": [n for n, _ in cc] + ["Risk Score"], "matrix": cm.values.tolist()}
lc = data[FEATS].corrwith(data["_lbl"]).fillna(0)
TOPL = [{"feature": i, "corr": round(float(lc[i]), 3)}
        for i in lc.abs().sort_values(ascending=False).head(8).index]

app = Flask(__name__)


@app.route("/")
def home():
    return Response(PAGE, mimetype="text/html")


@app.route("/api/users")
def users():
    g = data.groupby("_u").agg(m=("_risk", "max")).sort_values("m", ascending=False)
    return jsonify(g.index.tolist()[:1500])


@app.route("/api/prefill")
def prefill():
    uid = request.args.get("id", "").strip()
    ud = data[data["_u"] == uid]
    if ud.empty:
        return jsonify({})
    r = ud.sort_values("_risk", ascending=False).iloc[0]
    return jsonify({k: int(round(float(r[c]))) for k, c in MAP.items() if c})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    j = request.get_json(force=True) or {}
    uid = str(j.get("id", "")).strip()
    vals = {}
    for k in ACT:
        try:
            vals[k] = max(float(j.get(k) or 0), 0.0)
        except Exception:
            vals[k] = 0.0

    ud = data[data["_u"] == uid].sort_values("_d") if uid else data.iloc[0:0]
    known = len(ud) >= 5

    # ---- baselines, z-scores ----
    rows, zs, exps, obs, base = [], [], [], [], {}
    for k, (label, _) in ACT.items():
        col = MAP[k]
        if not col:
            rows.append({"label": label, "value": vals[k], "missing": True})
            continue
        gm, gs = G[k]
        if known:
            sm = float(ud[col].mean())
            ss = floor_std(ud[col].std(), sm)
        else:
            sm, ss = gm, gs
        base[k] = (sm, ss, gm, gs)
        zu = (vals[k] - sm) / ss
        rows.append({"label": label, "value": vals[k], "missing": False,
                     "self_mean": fl(sm) if known else None, "peer_mean": fl(gm),
                     "z_self": fl(zu) if known else None,
                     "z_peer": fl((vals[k] - gm) / gs), "z_used": fl(zu)})
        zs.append(max(zu, 0))
        exps.append(max(sm, 0.5))
        obs.append(vals[k])

    stat = float(100 * (1 - np.exp(-sum(zs) / 6)))

    # ---- chi-square goodness of fit ----
    if obs:
        cs = float(sum((o - e) ** 2 / e for o, e in zip(obs, exps)))
        chi = {"stat": fl(cs), "df": len(obs), "p": fl(chi2.sf(cs, len(obs)))}
    else:
        chi = {"stat": None, "df": 0, "p": None}

    # ---- model score on a row with the entered values ----
    row = (ud[FEATS].mean() if known else FMED).copy()
    for k, (sm, ss, gm, gs) in base.items():
        col = MAP[k]
        row[col] = vals[k]
        for d in FEATS:
            dl = d.lower()
            if d != col and col.lower() in dl and any(t in dl for t in ("zscore", "z_", "_z", "deviation", "_dev")):
                row[d] = (vals[k] - gm) / gs if "peer" in dl else (vals[k] - sm) / ss
    xr = pd.DataFrame([row[FEATS].astype(float).values], columns=FEATS).fillna(0)
    rp = float(rf.predict_proba(xr)[0, 1])
    ip = float(-iso.score_samples(xr)[0])
    model = model_score(rp, ip)

    final = round(0.5 * model + 0.5 * stat, 1)
    sev = sev_of(final)
    cur = STATES.index(sev)

    # ---- Markov prediction ----
    if known:
        UC = trans_counts(ud["_sev"].tolist())
        P = (UC + 5 * GP) / (UC.sum(axis=1, keepdims=True) + 5)
    else:
        P = GP
    probs = P[cur]
    pred = STATES[int(np.argmax(probs))]
    t_high = None
    if cur >= 2:
        t_high = 0
    else:
        try:
            sol = np.linalg.solve(np.eye(2) - P[:2, :2], np.ones(2))
            t = float(sol[cur])
            t_high = round(t, 1) if math.isfinite(t) and t >= 0 else None
        except Exception:
            t_high = None

    # ---- history ----
    hist = {"dates": [], "risk": []}
    if known:
        h = ud.tail(400)
        hist = {"dates": h["_d"].tolist(), "risk": [float(x) for x in h["_risk"]]}

    # ---- SHAP / contributions ----
    try:
        import shap
        sv = shap.TreeExplainer(rf).shap_values(xr)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.array(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        contrib = sv[0]
        method = "SHAP values (TreeExplainer on the Random Forest)"
    except Exception:
        z = ((row[FEATS].astype(float) - FM) / FS).values
        contrib = z * rf.feature_importances_
        method = "Feature contribution (z-score x model importance). Run 'pip install shap' for true SHAP values"
    order = np.argsort(-np.abs(contrib))[:8]
    shap_items = [{"feature": FEATS[i], "value": float(contrib[i])} for i in order]

    if not uid:
        note = "No Employee ID entered. Compared against the all-employee (peer) baseline."
    elif not known:
        note = f"Employee {uid} not found (or under 5 days of data). Using the peer baseline."
    else:
        note = f"Employee {uid}: {len(ud)} days of history used as the personal baseline."

    return jsonify(
        uid=uid, known=known, days=int(len(ud)), note=note,
        insider=int(ud["_lbl"].max()) if known else None,
        model_risk=fl(model), stat_risk=fl(stat), final_risk=final, severity=sev,
        rows=rows, chi=chi,
        markov={"matrix": np.round(P, 3).tolist(), "current": sev, "predicted": pred,
                "next_probs": {s: float(probs[i]) for i, s in enumerate(STATES)},
                "time_to_high": t_high, "transitions": int(GC.sum())},
        history=hist, corr=CORR, top_label=TOPL,
        shap={"items": shap_items, "method": method},
    )


PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Insider Threat Detection System</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box}
body{margin:0;font-family:Segoe UI,Arial,sans-serif;color:#e8edf8;background:linear-gradient(160deg,#08102e,#0d1b4d 60%,#08102e);min-height:100vh}
.wrap{max-width:1180px;margin:0 auto;padding:26px 20px 60px}
h1{text-align:center;margin:6px 0 6px;font-size:34px;letter-spacing:1px;color:#fff}
.sub{text-align:center;color:#22d3ee;font-size:14px;margin-bottom:34px;letter-spacing:.5px}
.top{display:flex;flex-wrap:wrap;gap:34px;align-items:flex-start}
.form{width:400px;max-width:100%}
.f{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
.f label{font-size:15px;width:110px}
.f input{flex:1;background:#1b2a56;border:1px solid #2c3d73;color:#fff;padding:10px 12px;font-size:15px;border-radius:2px;outline:none}
.f input:focus{border-color:#22d3ee}
.btn{background:#e8eaf0;color:#1a1f33;border:none;padding:16px 8px;font-size:12.5px;letter-spacing:.6px;text-transform:uppercase;cursor:pointer;font-weight:600}
.btn:hover{background:#fff}
#go{width:100%;font-size:17px;padding:18px;margin-top:6px}
.grid{flex:1;min-width:300px;display:grid;grid-template-columns:1fr 1fr;gap:12px;max-width:600px}
.blue{background:#1d4ed8;color:#fff;text-transform:none;font-size:14px}.blue:hover{background:#2563eb}
.purple{background:#7c3aed;color:#fff;text-transform:none;font-size:14px}.purple:hover{background:#8b5cf6}
.row2{display:flex;flex-wrap:wrap;gap:30px;justify-content:space-between;align-items:flex-start;margin-top:30px}
#msg{flex:1;min-width:260px;font-size:15px;padding-top:10px;color:#d6def2}
.panel{width:470px;max-width:100%;background:#f1f3f8;padding:8px;border-radius:2px}
.panel div{background:#0b1230;padding:14px 16px;font-weight:700;font-size:16px;margin-bottom:8px;color:#22d3ee}
#res{display:none;margin-top:40px}
.sec{background:rgba(255,255,255,.04);border:1px solid #27345f;border-radius:10px;padding:20px;margin-bottom:20px;scroll-margin-top:16px}
.sec h2{margin:0 0 14px;font-size:18px;border-left:4px solid #22d3ee;padding-left:12px;letter-spacing:.4px}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{padding:9px 8px;border-bottom:1px solid #27345f;text-align:left}
th{color:#9fb0d6;font-weight:600}
.heat td,.heat th{text-align:center}.heat tr.cur th{color:#22d3ee}
.mut{color:#9fb0d6;font-size:13px;line-height:1.5}
.big{font-size:30px;font-weight:700;margin:4px 0 10px}
.meter{height:12px;background:#1b2a56;border-radius:6px;overflow:hidden;margin-bottom:14px}
.meter div{height:100%}
.badge{padding:3px 12px;border-radius:20px;font-size:13px;font-weight:700;color:#08102e}
.cv{position:relative;height:300px}
.pb{display:flex;align-items:center;gap:12px;margin:8px 0}.pb span{width:70px}.pb b{width:60px;text-align:right}
.pb div{flex:1;height:14px;background:#1b2a56;border-radius:7px;overflow:hidden}.pb i{display:block;height:100%}
.warn{color:#fbbf24;font-size:13px}
pre{white-space:pre-wrap;background:#0b1230;padding:16px;border-radius:8px;font-size:13.5px;line-height:1.55;margin:0 0 12px}
.act .btn{margin-right:10px}
@media print{.noprint{display:none!important}body{background:#fff;color:#000}.sec{border:none}}
</style></head><body><div class="wrap">
<div class="noprint">
<h1>INSIDER THREAT DETECTION SYSTEM</h1>
<div class="sub">MATHEMATICAL RISK ANALYSIS &nbsp;|&nbsp; SECURITY MONITORING</div>
<div class="top">
 <div class="form">
  <div class="f"><label>Employee ID</label><input id="eid" list="ul" autocomplete="off"><datalist id="ul"></datalist></div>
  <div class="f"><label>Login Count</label><input id="login" type="number" min="0"></div>
  <div class="f"><label>File Access</label><input id="file" type="number" min="0"></div>
  <div class="f"><label>USB Activity</label><input id="usb" type="number" min="0"></div>
  <div class="f"><label>Email Count</label><input id="email" type="number" min="0"></div>
  <button class="btn" id="go">ANALYZE EMPLOYEE</button>
 </div>
 <div class="grid">
  <button class="btn" data-go="sec2">Show Baseline Chart</button>
  <button class="btn" data-go="sec3">Show Markov Transition Matrix</button>
  <button class="btn" data-go="sec4">Predict Next Risk State</button>
  <button class="btn" data-go="sec5">Estimate Time to High Risk</button>
  <button class="btn" data-go="sec6">Chi-Square Statistical Test</button>
  <button class="btn" data-go="sec7">Compare Self vs Peer Baseline</button>
  <button class="btn" data-go="sec8">Show Risk History</button>
  <button class="btn" data-go="sec11">Generate Risk Report</button>
  <button class="btn blue" data-go="sec9">Correlation Analysis</button>
  <button class="btn purple" data-go="sec10">SHAP Explanation</button>
 </div>
</div>
<div class="row2">
 <div id="msg">Enter employee activity and click ANALYZE EMPLOYEE. Tip: pick an Employee ID from the list and the fields fill with that employee's highest-risk day.</div>
 <div class="panel"><div id="st">RISK STATUS: WAITING FOR ANALYSIS</div><div id="mp">MARKOV PREDICTION: WAITING FOR ANALYSIS</div></div>
</div>
</div>

<div id="res">
 <div class="sec" id="sec1"><h2>1. Risk Status</h2><div id="s1"></div></div>
 <div class="sec" id="sec2"><h2>2. Baseline Chart</h2><div class="cv"><canvas id="c2"></canvas></div></div>
 <div class="sec" id="sec3"><h2>3. Markov Transition Matrix</h2><div id="s3"></div></div>
 <div class="sec" id="sec4"><h2>4. Predicted Next Risk State</h2><div id="s4"></div></div>
 <div class="sec" id="sec5"><h2>5. Estimated Time to High Risk</h2><div id="s5"></div></div>
 <div class="sec" id="sec6"><h2>6. Chi-Square Statistical Test</h2><div id="s6"></div></div>
 <div class="sec" id="sec7"><h2>7. Self vs Peer Baseline Comparison</h2><div id="s7"></div></div>
 <div class="sec" id="sec8"><h2>8. Risk History</h2><div class="cv" id="h8"><canvas id="c8"></canvas></div><p id="s8m" class="mut"></p></div>
 <div class="sec" id="sec9"><h2>9. Correlation Analysis</h2><div id="s9"></div></div>
 <div class="sec" id="sec10"><h2>10. SHAP Explanation</h2><div class="cv"><canvas id="c10"></canvas></div><p id="s10m" class="mut"></p></div>
 <div class="sec" id="sec11"><h2>11. Risk Report</h2><pre id="rep"></pre><div class="act noprint"><button class="btn" id="dl">Download report</button><button class="btn" id="pr">Print report</button></div></div>
</div>
</div>

<script>
const $=s=>document.querySelector(s);
const S=['Low','Medium','High','Critical'];
const SC={Low:'#22c55e',Medium:'#eab308',High:'#f97316',Critical:'#ef4444'};
Chart.defaults.color='#b8c4e0';Chart.defaults.borderColor='#27345f';
const ch={};let LAST=null;
function mk(id,cfg){if(ch[id])ch[id].destroy();ch[id]=new Chart(document.getElementById(id),cfg);}
const n=(v,d)=>(v===null||v===undefined)?'-':Number(v).toFixed(d===undefined?2:d);
const sev=s=>`<span class="badge" style="background:${SC[s]}">${s}</span>`;

fetch('/api/users').then(r=>r.json()).then(u=>{$('#ul').innerHTML=u.map(x=>`<option value="${x}">`).join('');});
$('#eid').addEventListener('change',async()=>{
 const id=$('#eid').value.trim();if(!id)return;
 const p=await (await fetch('/api/prefill?id='+encodeURIComponent(id))).json();
 ['login','file','usb','email'].forEach(k=>{if(p[k]!==undefined)$('#'+k).value=p[k];});
});

async function analyze(){
 const body={id:$('#eid').value.trim(),login:$('#login').value,file:$('#file').value,usb:$('#usb').value,email:$('#email').value};
 $('#msg').textContent='Analyzing...';
 let d;
 try{const r=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});d=await r.json();}
 catch(e){$('#msg').textContent='Error: could not reach the server.';return;}
 if(d.error){$('#msg').textContent='Error: '+d.error;return;}
 LAST=d;render(d);
 document.getElementById('sec1').scrollIntoView({behavior:'smooth'});
}

function render(d){
 $('#res').style.display='block';
 const col=SC[d.severity];
 $('#st').textContent='RISK STATUS: '+d.severity.toUpperCase()+' ('+d.final_risk+'/100)';$('#st').style.color=col;
 $('#mp').textContent='MARKOV PREDICTION: '+d.markov.predicted.toUpperCase();
 $('#msg').textContent=d.note;
 const miss=d.rows.filter(r=>r.missing).map(r=>r.label);

 $('#s1').innerHTML=`<div class="big" style="color:${col}">${d.severity} &nbsp; ${d.final_risk}/100</div>
 <div class="meter"><div style="width:${d.final_risk}%;background:${col}"></div></div>
 <table>
 <tr><td>Model risk (Random Forest + Isolation Forest)</td><td>${n(d.model_risk,1)}</td></tr>
 <tr><td>Statistical anomaly score (deviation from baseline)</td><td>${n(d.stat_risk,1)}</td></tr>
 <tr><td>Final risk (average of the two)</td><td>${d.final_risk}</td></tr>
 <tr><td>Baseline used</td><td>${d.known?"Personal history ("+d.days+" days)":"All-employee (peer) average"}</td></tr>
 <tr><td>Dataset ground-truth label</td><td>${d.insider===null?"n/a":(d.insider?"Labelled insider":"Not labelled as insider")}</td></tr>
 </table>
 <p class="mut">Severity bands: Low under 25, Medium 25-49, High 50-74, Critical 75 and above.</p>
 ${miss.length?`<p class="warn">No matching data column was found for: ${miss.join(", ")}. Those fields were ignored.</p>`:""}`;

 const lab=d.rows.filter(r=>!r.missing);
 mk('c2',{type:'bar',data:{labels:lab.map(r=>r.label),datasets:[
  {label:'Entered today',data:lab.map(r=>r.value),backgroundColor:'#22d3ee'},
  {label:'Own average',data:lab.map(r=>r.self_mean),backgroundColor:'#a78bfa'},
  {label:'Peer average',data:lab.map(r=>r.peer_mean),backgroundColor:'#64748b'}]},
  options:{maintainAspectRatio:false}});

 let h='<table class="heat"><tr><th>From / To</th>'+S.map(s=>'<th>'+s+'</th>').join('')+'</tr>';
 d.markov.matrix.forEach((r,i)=>{h+='<tr'+(S[i]===d.markov.current?' class="cur"':'')+'><th>'+S[i]+'</th>'+r.map(v=>`<td style="background:rgba(34,211,238,${Math.min(v*1.2,1)*0.55})">${n(v,3)}</td>`).join('')+'</tr>';});
 $('#s3').innerHTML=h+`</table><p class="mut">Rows are today's state, columns are the next state. Built from ${d.markov.transitions} observed day-to-day transitions${d.known?", blended with this employee's own history":""}. The current state is highlighted.</p>`;

 $('#s4').innerHTML=`<p>Current state: ${sev(d.markov.current)} &rarr; most likely next state: ${sev(d.markov.predicted)}</p>`+
  S.map(s=>`<div class="pb"><span>${s}</span><div><i style="width:${d.markov.next_probs[s]*100}%;background:${SC[s]}"></i></div><b>${n(d.markov.next_probs[s]*100,1)}%</b></div>`).join('');

 const t=d.markov.time_to_high;
 $('#s5').innerHTML=t===0?'<p class="big" style="color:#ef4444">Already at High or Critical risk</p>':
  t===null?'<p>High risk is not reachable from this state in the observed data.</p>':
  `<p class="big">${n(t,1)} days</p><p class="mut">Expected number of user-days until the employee first reaches High or Critical, starting from the ${d.markov.current} state (absorbing Markov chain).</p>`;

 const c=d.chi;
 $('#s6').innerHTML=c.p===null?'<p>Not enough data for the test.</p>':
  `<table><tr><td>Chi-square statistic</td><td>${n(c.stat,2)}</td></tr><tr><td>Degrees of freedom</td><td>${c.df}</td></tr>
  <tr><td>p-value</td><td>${c.p<0.0001?"&lt; 0.0001":n(c.p,4)}</td></tr>
  <tr><td>Verdict</td><td><b style="color:${c.p<0.05?'#ef4444':'#22c55e'}">${c.p<0.05?"Significant deviation from baseline (p &lt; 0.05)":"Consistent with baseline (p &ge; 0.05)"}</b></td></tr></table>
  <p class="mut">Tests whether the entered activity counts differ from the expected baseline counts. Approximate, because counts are treated as Poisson-like.</p>`;

 $('#s7').innerHTML='<table><tr><th>Activity</th><th>Entered</th><th>Own avg</th><th>Peer avg</th><th>Z vs own</th><th>Z vs peer</th></tr>'+
  d.rows.map(r=>r.missing?`<tr><td>${r.label}</td><td>${r.value}</td><td colspan="4">column not found</td></tr>`:
  `<tr><td>${r.label}</td><td>${r.value}</td><td>${n(r.self_mean,1)}</td><td>${n(r.peer_mean,1)}</td><td>${n(r.z_self,2)}</td><td>${n(r.z_peer,2)}</td></tr>`).join('')+
  '</table><p class="mut">A Z-score above 3 is a strong deviation. "Peer" means the average of all employees.</p>';

 if(d.history.dates.length){
  $('#h8').style.display='block';$('#s8m').textContent='';
  mk('c8',{type:'line',data:{labels:d.history.dates,datasets:[{label:'Risk score',data:d.history.risk,borderColor:'#22d3ee',backgroundColor:'#22d3ee22',fill:true,pointRadius:0,tension:.2}]},
   options:{maintainAspectRatio:false,scales:{y:{min:0,max:100},x:{ticks:{maxTicksLimit:8}}}}});
 }else{$('#h8').style.display='none';$('#s8m').textContent='No history available. Enter a valid Employee ID from the list.';}

 const L=d.corr.labels;
 let t9='<table class="heat"><tr><th></th>'+L.map(x=>'<th>'+x+'</th>').join('')+'</tr>';
 d.corr.matrix.forEach((r,i)=>{t9+='<tr><th>'+L[i]+'</th>'+r.map(v=>`<td style="background:rgba(${v>=0?"34,211,238":"239,68,68"},${Math.abs(v)*0.6})">${v.toFixed(2)}</td>`).join('')+'</tr>';});
 t9+='</table><h4>Features most correlated with the insider label</h4><table>'+d.top_label.map(x=>`<tr><td>${x.feature}</td><td>${x.corr}</td></tr>`).join('')+'</table>';
 $('#s9').innerHTML=t9;

 mk('c10',{type:'bar',data:{labels:d.shap.items.map(x=>x.feature),datasets:[{data:d.shap.items.map(x=>x.value),backgroundColor:d.shap.items.map(x=>x.value>=0?'#ef4444':'#22c55e')}]},
  options:{indexAxis:'y',maintainAspectRatio:false,plugins:{legend:{display:false}}}});
 $('#s10m').textContent=d.shap.method+'. Red bars push the risk up, green bars pull it down.';

 const rec={Low:'Continue routine monitoring.',Medium:'Increase monitoring and review recent activity.',High:'Manual review within 24 hours. Consider limiting USB and bulk file access.',Critical:'Escalate to the security team immediately and review access rights.'}[d.severity];
 const top3=d.shap.items.slice(0,3).map(x=>x.feature).join(', ');
 const lines=['INSIDER THREAT RISK REPORT','Generated: '+new Date().toLocaleString(),'',
  'Employee ID: '+(d.uid||'(not given)'),'Baseline: '+d.note,'',
  'ACTIVITY ENTERED',...d.rows.map(r=>'  '+r.label+': '+r.value),'',
  'RISK ASSESSMENT','  Final risk: '+d.final_risk+'/100 ('+d.severity+')','  Model risk: '+n(d.model_risk,1),'  Statistical score: '+n(d.stat_risk,1),'',
  'STATISTICAL TEST','  Chi-square: '+n(d.chi.stat,2)+' (df '+d.chi.df+'), p = '+n(d.chi.p,4),'',
  'MARKOV ANALYSIS','  Current state: '+d.markov.current,'  Most likely next state: '+d.markov.predicted,
  '  Expected time to High risk: '+(d.markov.time_to_high===null?'not reachable':d.markov.time_to_high+' days'),'',
  'TOP CONTRIBUTING FEATURES','  '+top3,'','RECOMMENDED ACTION','  '+rec,'',
  'Note: the CERT dataset is synthetic, so scores demonstrate the method and are not a deployment result.'];
 $('#rep').textContent=lines.join('\n');
}

$('#go').onclick=analyze;
['eid','login','file','usb','email'].forEach(i=>$('#'+i).addEventListener('keydown',e=>{if(e.key==='Enter')analyze();}));
document.querySelectorAll('[data-go]').forEach(b=>b.onclick=()=>{
 if(!LAST){$('#msg').textContent='Click ANALYZE EMPLOYEE first.';return;}
 document.getElementById(b.dataset.go).scrollIntoView({behavior:'smooth'});
});
$('#dl').onclick=()=>{if(!LAST)return;const a=document.createElement('a');
 a.href=URL.createObjectURL(new Blob([$('#rep').textContent],{type:'text/plain'}));
 a.download='risk_report_'+(LAST.uid||'employee')+'.txt';a.click();};
$('#pr').onclick=()=>window.print();
</script></body></html>"""


def _open(port):
    time.sleep(1.5)
    try:
        webbrowser.open(f"http://127.0.0.1:{port}/")
    except Exception:
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    host = "0.0.0.0"
    print(f"Dashboard running at http://{host}:{port}/  (Ctrl+C to stop)")
    if not os.environ.get("RENDER"):
        threading.Thread(target=_open, args=(port,), daemon=True).start()
    app.run(host=host, port=port, debug=False)
