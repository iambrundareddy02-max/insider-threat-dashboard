import os, json, threading, time, webbrowser
import pandas as pd
from flask import Flask, jsonify, request, Response

BASE = os.path.dirname(os.path.abspath(__file__))
RISK = os.path.join(BASE, "data", "processed", "risk_scores.csv")
METRICS = os.path.join(BASE, "models", "metrics.json")

if not os.path.exists(RISK):
    raise SystemExit("risk_scores.csv not found. Run:  py train_model.py  first.")

df = pd.read_csv(RISK)
df["reasons"] = df["reasons"].fillna("")
df["user"] = df["user"].astype(str)
metrics = json.load(open(METRICS)) if os.path.exists(METRICS) else {}

app = Flask(__name__)
ORDER = ["Low", "Medium", "High", "Critical"]


@app.route("/")
def home():
    return Response(PAGE, mimetype="text/html")


@app.route("/api/stats")
def stats():
    sev = df["severity"].value_counts()
    severity = {k: int(sev.get(k, 0)) for k in ORDER}
    daily = (df.assign(alert=df["severity"].isin(["High", "Critical"]))
               .groupby("date")["alert"].sum().reset_index().sort_values("date"))
    top = (df.groupby("user")
             .agg(max_risk=("risk_score", "max"), avg_risk=("risk_score", "mean"),
                  insider=("label", "max"))
             .sort_values("max_risk", ascending=False).head(10).reset_index())
    return jsonify(
        users=int(df["user"].nunique()),
        rows=int(len(df)),
        insiders=int(df.groupby("user")["label"].max().sum()),
        severity=severity,
        dates=daily["date"].tolist(),
        alerts=[int(x) for x in daily["alert"]],
        top=[{"user": r.user, "max_risk": round(float(r.max_risk), 1),
              "avg_risk": round(float(r.avg_risk), 1), "insider": int(r.insider)}
             for r in top.itertuples()],
    )


@app.route("/api/alerts")
def alerts():
    page = max(int(request.args.get("page", 1)), 1)
    q = request.args.get("q", "").strip().lower()
    sv = request.args.get("severity", "")
    d = df[df["severity"] == sv] if sv else df[df["severity"].isin(["Medium", "High", "Critical"])]
    if q:
        d = d[d["user"].str.lower().str.contains(q, regex=False)]
    d = d.sort_values("risk_score", ascending=False)
    total, per = len(d), 15
    pages = max((total + per - 1) // per, 1)
    page = min(page, pages)
    items = d.iloc[(page - 1) * per: page * per]
    return jsonify(items=json.loads(items.to_json(orient="records")),
                   page=page, pages=pages, total=total)


@app.route("/api/user/<uid>")
def user(uid):
    d = df[df["user"] == uid].sort_values("date")
    if d.empty:
        return jsonify(error="not found"), 404
    top = d.sort_values("risk_score", ascending=False).head(5)
    return jsonify(
        user=uid, dates=d["date"].tolist(),
        risk=[float(x) for x in d["risk_score"]],
        max_risk=float(d["risk_score"].max()),
        avg_risk=round(float(d["risk_score"].mean()), 1),
        malicious_days=int(d["label"].sum()),
        top=json.loads(top[["date", "risk_score", "severity", "reasons"]].to_json(orient="records")),
    )


@app.route("/api/metrics")
def get_metrics():
    m = {k: v for k, v in metrics.items() if k != "features"}
    m["n_features"] = len(metrics.get("features", []))
    return jsonify(m)


PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Sentinel - Insider Threat Detection</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0b1020;--card:#131a2e;--line:#243049;--txt:#e6ebf5;--mut:#8a97b4;--acc:#22d3ee}
*{box-sizing:border-box}
body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--txt)}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;padding:14px 28px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:5}
.logo{font-weight:700;font-size:20px;letter-spacing:1px}.logo span{color:var(--acc)}
nav button{background:none;border:1px solid transparent;color:var(--mut);padding:8px 14px;border-radius:8px;cursor:pointer;font-size:14px}
nav button.on,nav button:hover{color:var(--txt);border-color:var(--line);background:var(--card)}
main{max-width:1200px;margin:0 auto;padding:24px}
section{display:none}
h2{margin:0 0 16px}
.hero{background:linear-gradient(135deg,#12304a,#131a2e);border:1px solid var(--line);border-radius:14px;padding:28px;margin-bottom:20px}
.hero h1{margin:0 0 8px;font-size:28px}.hero p{margin:0;color:var(--mut);max-width:760px;line-height:1.5}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin-bottom:20px}
.kpi,.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.kpi .n{font-size:28px;font-weight:700;color:var(--acc)}.kpi .l{color:var(--mut);font-size:13px;margin-top:4px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px;margin-bottom:16px}
.card h3{margin:0 0 12px;font-size:15px;color:var(--mut);font-weight:600}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:10px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:600}
.mut{color:var(--mut);font-size:13px}
.badge{padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600;color:#0b1020}
.tag{background:#ef444433;color:#ef4444;padding:2px 8px;border-radius:10px;font-size:11px;margin-left:6px}
input,select,button.b{background:var(--card);color:var(--txt);border:1px solid var(--line);padding:9px 12px;border-radius:8px;font-size:14px}
button.b{cursor:pointer}button.b:hover{border-color:var(--acc)}
.bar{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}
a.u{color:var(--acc);text-decoration:none}a.u:hover{text-decoration:underline}
.cv{position:relative;height:280px}
.foot{display:flex;gap:12px;align-items:center;margin-top:14px}
</style></head><body>
<header><div class="logo">SENTINEL <span>//</span> Insider Threat Detection</div>
<nav>
<button data-tab="overview" class="on">Overview</button>
<button data-tab="alerts">Alerts</button>
<button data-tab="profile">User Profile</button>
<button data-tab="method">Methodology</button>
</nav></header>
<main>

<section id="overview" style="display:block">
<div class="hero"><h1>Insider Threat Detection &amp; Analysis</h1>
<p>Behavioural analytics on the CERT insider threat dataset. Each user-day is scored 0 to 100 using a supervised Random Forest blended with an Isolation Forest anomaly score, and flagged by severity.</p></div>
<div class="kpis" id="kpis"></div>
<div class="grid">
<div class="card"><h3>Severity distribution</h3><div class="cv"><canvas id="cSev"></canvas></div></div>
<div class="card"><h3>High and Critical alerts per day</h3><div class="cv"><canvas id="cTrend"></canvas></div></div>
</div>
<div class="grid">
<div class="card"><h3>Top 10 riskiest users (red = labelled insider)</h3><div class="cv"><canvas id="cTop"></canvas></div></div>
<div class="card"><h3>Top users</h3><table><thead><tr><th>User</th><th>Max risk</th><th>Avg risk</th></tr></thead><tbody id="tbody"></tbody></table></div>
</div>
</section>

<section id="alerts">
<h2>Alerts and Incidents</h2>
<div class="bar">
<input id="q" placeholder="Search user ID">
<select id="sv"><option value="">Medium and above</option><option>Critical</option><option>High</option><option>Medium</option><option>Low</option></select>
<button class="b" id="go">Search</button>
</div>
<div class="card"><table><thead><tr><th>User</th><th>Date</th><th>Risk</th><th>Severity</th><th>Why it was flagged</th></tr></thead><tbody id="abody"></tbody></table>
<div class="foot"><button class="b" id="prev">Prev</button><span id="pg" class="mut"></span><button class="b" id="next">Next</button></div></div>
</section>

<section id="profile">
<h2>User Risk Profile</h2>
<div class="bar"><input id="puser" placeholder="Enter user ID"><button class="b" id="pgo">Open</button></div>
<div class="card"><div id="pinfo" class="mut">Click a user in the Alerts page or enter an ID.</div>
<div class="cv" style="margin-top:12px"><canvas id="cUser"></canvas></div></div>
<div class="card" style="margin-top:16px"><h3>Highest-risk days</h3>
<table><thead><tr><th>Date</th><th>Risk</th><th>Severity</th><th>Reasons</th></tr></thead><tbody id="pbody"></tbody></table></div>
</section>

<section id="method">
<h2>Methodology</h2>
<div class="kpis" id="mk"></div>
<div class="card"><h3>How it works</h3>
<p class="mut" style="line-height:1.6">1) Raw logon, device, file, email and web logs from the CERT r4.2 dataset are aggregated into one row per user per day. 2) Behavioural features such as after-hours logons, USB use and file copies are built, along with deviation from each user's own baseline. 3) A Random Forest (trained on the labelled insiders with class weights) and an Isolation Forest (unsupervised anomaly detector) are combined into a 0 to 100 risk score. 4) Scores are split into Low, Medium, High and Critical by percentile. 5) The top contributing features are shown as the reason for each alert. The model is evaluated on a held-out later time period. Note: CERT data is synthetic, so results show method quality, not deployment performance.</p></div>
</section>
</main>

<script>
const $=s=>document.querySelector(s);
const COL={Low:'#22c55e',Medium:'#eab308',High:'#f97316',Critical:'#ef4444'};
Chart.defaults.color='#8a97b4';Chart.defaults.borderColor='#243049';
const charts={};
function mk(id,cfg){if(charts[id])charts[id].destroy();charts[id]=new Chart(document.getElementById(id),cfg);}
function show(t){
 document.querySelectorAll('section').forEach(s=>s.style.display=s.id===t?'block':'none');
 document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.dataset.tab===t));
}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>show(b.dataset.tab));
function bindUsers(){document.querySelectorAll('.u').forEach(a=>a.onclick=e=>{e.preventDefault();openUser(a.dataset.u);});}

async function loadStats(){
 const s=await (await fetch('/api/stats')).json();
 const k=[['Users monitored',s.users],['User-days analysed',s.rows.toLocaleString()],['Labelled insiders',s.insiders],['Critical alerts',s.severity.Critical],['High alerts',s.severity.High]];
 $('#kpis').innerHTML=k.map(x=>`<div class="kpi"><div class="n">${x[1]}</div><div class="l">${x[0]}</div></div>`).join('');
 const keys=Object.keys(s.severity);
 mk('cSev',{type:'doughnut',data:{labels:keys,datasets:[{data:keys.map(x=>s.severity[x]),backgroundColor:keys.map(x=>COL[x]),borderWidth:0}]},options:{maintainAspectRatio:false}});
 mk('cTrend',{type:'line',data:{labels:s.dates,datasets:[{label:'High+Critical alerts',data:s.alerts,borderColor:'#22d3ee',backgroundColor:'#22d3ee22',fill:true,pointRadius:0,tension:.25}]},options:{maintainAspectRatio:false,scales:{x:{ticks:{maxTicksLimit:8}}}}});
 mk('cTop',{type:'bar',data:{labels:s.top.map(t=>t.user),datasets:[{label:'Max risk',data:s.top.map(t=>t.max_risk),backgroundColor:s.top.map(t=>t.insider?'#ef4444':'#22d3ee')}]},options:{maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{min:0,max:100}},onClick:(e,el)=>{if(el.length)openUser(s.top[el[0].index].user);}}});
 $('#tbody').innerHTML=s.top.map(t=>`<tr><td><a href="#" class="u" data-u="${t.user}">${t.user}</a>${t.insider?'<span class="tag">insider</span>':''}</td><td>${t.max_risk}</td><td>${t.avg_risk}</td></tr>`).join('');
 bindUsers();
}

let page=1,pages=1;
async function loadAlerts(){
 const q=encodeURIComponent($('#q').value),sv=$('#sv').value;
 const r=await (await fetch(`/api/alerts?page=${page}&q=${q}&severity=${sv}`)).json();
 page=r.page;pages=r.pages;
 $('#abody').innerHTML=r.items.map(a=>`<tr><td><a href="#" class="u" data-u="${a.user}">${a.user}</a></td><td>${a.date}</td><td>${a.risk_score}</td><td><span class="badge" style="background:${COL[a.severity]}">${a.severity}</span></td><td class="mut">${a.reasons}</td></tr>`).join('')||'<tr><td colspan="5">No results</td></tr>';
 $('#pg').textContent=`Page ${r.page} of ${r.pages} (${r.total} alerts)`;
 bindUsers();
}
$('#go').onclick=()=>{page=1;loadAlerts();};
$('#q').addEventListener('keydown',e=>{if(e.key==='Enter'){page=1;loadAlerts();}});
$('#sv').onchange=()=>{page=1;loadAlerts();};
$('#prev').onclick=()=>{if(page>1){page--;loadAlerts();}};
$('#next').onclick=()=>{if(page<pages){page++;loadAlerts();}};

async function openUser(u){
 show('profile');$('#puser').value=u;
 const r=await fetch('/api/user/'+encodeURIComponent(u));
 if(!r.ok){$('#pinfo').textContent='User not found';return;}
 const d=await r.json();
 $('#pinfo').innerHTML=`<b style="color:#e6ebf5">${d.user}</b> &nbsp; max risk ${d.max_risk} &nbsp; avg risk ${d.avg_risk} &nbsp; labelled malicious days: ${d.malicious_days}`;
 mk('cUser',{type:'line',data:{labels:d.dates,datasets:[{label:'Risk score',data:d.risk,borderColor:'#22d3ee',pointRadius:0,tension:.2}]},options:{maintainAspectRatio:false,scales:{y:{min:0,max:100},x:{ticks:{maxTicksLimit:8}}}}});
 $('#pbody').innerHTML=d.top.map(a=>`<tr><td>${a.date}</td><td>${a.risk_score}</td><td><span class="badge" style="background:${COL[a.severity]}">${a.severity}</span></td><td class="mut">${a.reasons}</td></tr>`).join('');
}
$('#pgo').onclick=()=>{if($('#puser').value)openUser($('#puser').value.trim());};

async function loadMetrics(){
 const m=await (await fetch('/api/metrics')).json();
 const f=v=>typeof v==='number'?v.toFixed(3):'-';
 const items=[['Precision',f(m.precision)],['Recall',f(m.recall)],['F1 score',f(m.f1)],['ROC-AUC',f(m.roc_auc)],['Features',m.n_features||'-'],['Test rows',m.test_rows||'-']];
 $('#mk').innerHTML=items.map(x=>`<div class="kpi"><div class="n">${x[1]}</div><div class="l">${x[0]}</div></div>`).join('');
}
loadStats();loadAlerts();loadMetrics();
</script></body></html>"""


def _open(port):
    time.sleep(1.2)
    try:
        webbrowser.open(f"http://127.0.0.1:{port}/")
    except Exception:
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    host = "0.0.0.0"
    print(f"Sentinel running at http://{host}:{port}/  (Ctrl+C to stop)")
    if not os.environ.get("RENDER"):
        threading.Thread(target=_open, args=(port,), daemon=True).start()
    app.run(host=host, port=port, debug=False)