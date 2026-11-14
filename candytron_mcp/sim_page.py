"""The web page of the simulated table (served by candytron_mcp at /sim, and
through the control panel at http://127.0.0.1:8090/sim)."""

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Simulated Candy Table</title>
<style>
:root{--bg:#f4f5f7;--card:#fff;--fg:#1d2329;--mut:#66707a;--line:#dde1e6;--ok:#1a7f37;--bad:#cf222e;--acc:#1f6feb;--warn:#9a6700}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#111418;--card:#1a1f25;--fg:#e6e8ea;--mut:#98a2ad;--line:#2c333b;--ok:#3fb950;--bad:#f85149;--acc:#58a6ff;--warn:#d29922}}
:root[data-theme="dark"]{--bg:#111418;--card:#1a1f25;--fg:#e6e8ea;--mut:#98a2ad;--line:#2c333b;--ok:#3fb950;--bad:#f85149;--acc:#58a6ff;--warn:#d29922}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,sans-serif}
header{padding:12px 16px;border-bottom:1px solid var(--line);background:var(--card);display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
h1{font-size:17px;margin:0}.mut{color:var(--mut)}
main{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(280px,1fr);gap:14px;padding:14px 16px}
@media (max-width:860px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:14px}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut);margin:0 0 8px}
img{width:100%;border-radius:8px;background:#000;display:block}
.btns{display:flex;flex-wrap:wrap;gap:8px}
button{font:inherit;padding:7px 12px;border-radius:7px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.on{background:var(--warn);border-color:var(--warn);color:#fff}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:3px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--mut);font-weight:500}.ok{color:var(--ok)}.bad{color:var(--bad);font-weight:600}
ul{list-style:none;margin:0;padding:0;font-size:13px;max-height:260px;overflow:auto}li{padding:2px 0;border-bottom:1px dashed var(--line)}
.pill{font-size:12px;padding:2px 8px;border-radius:99px;background:#8882}
</style></head><body>
<header><h1>Simulated candy table</h1><span class="mut">drawn table + real YOLO vision + a pretend arm</span><span id="mode" class="pill"></span></header>
<main><div>
<div class="card"><img src="/sim/stream" alt="the simulated table, as the table camera sees it"></div>
<div class="card"><h2>Do something to the table</h2><div class="btns">
<button onclick="act('restock')">Refill the table</button>
<button onclick="act('loose')">Drop a candy between the boxes</button>
<button id="bFail" onclick="act('fail')">Next grip slips</button>
<button onclick="act('clear')">Clear the table</button></div>
<p class="mut" style="margin:8px 0 0">A candy put on O0 is taken by the "visitor" after a few seconds.
Green boxes in the picture are what the YOLO model sees.</p></div>
</div><div>
<div class="card"><h2>Really there vs. what the robot is told</h2><table id="cmp"></table></div>
<div class="card"><h2>Arm</h2><div id="arm" class="mut"></div></div>
<div class="card"><h2>What happened</h2><ul id="ev"></ul></div>
</div></main>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const SAME={NutcremePear:'Pearnut',Nutcreme:'Whitenut',VanillaToffee:'VanillaFudge'};
const canon=n=>SAME[n]||n;
async function act(a){try{await fetch('/sim/do/'+a,{method:'POST'})}catch(e){}poll(true)}
let timer=null;
async function poll(once){
 try{const s=await (await fetch('/sim/state')).json();draw(s)}catch(e){$('arm').textContent='simulator not reachable'}
 if(!once){clearTimeout(timer);timer=setTimeout(()=>poll(),700)}}
function draw(s){
 $('mode').textContent='vision: '+s.vision+(s.calibrating?' · calibrating':'');
 $('bFail').classList.toggle('on',s.fail_next);
 const tags=[];for(const r of 'DCBA')for(const c of '123')tags.push(r+c);
 const extra=Object.keys(s.truth).concat(Object.keys(s.seen)).filter(k=>!tags.includes(k));
 const rows=tags.concat([...new Set(extra)].sort());
 $('cmp').innerHTML='<tr><th>Where</th><th>Really there</th><th>Robot is told</th><th></th></tr>'+rows.map(k=>{
  const t=(s.truth[k]||[]),seen=s.seen[k]||'';
  const tt=t.map(canon).join(', '),ok=(t.length<=1&&canon(t[0]||'')==canon(seen))||k=='in gripper'||(k=='O0'&&!seen);
  return `<tr><td>${esc(k)}</td><td>${esc(t.join(', ')||'–')}</td><td>${esc(seen||'–')}</td><td class="${ok?'ok':'bad'}">${ok?'✓':'✗'}</td></tr>`}).join('');
 $('arm').innerHTML=(s.moving?'<b>moving</b>':'idle')+(s.queue.length?' · queue: '+s.queue.map(m=>esc(m[0]+' → '+m[1])).join(', '):'')+
  (s.fail_next?' · <span class="bad">next grip will slip</span>':'')+
  (s.misses&&s.misses.dir?`<br>Frames YOLO got wrong, saved with the true labels: <b>${s.misses.saved}</b> <span class="mut">(${esc(s.misses.dir)})</span>`:'');
 $('ev').innerHTML=s.events.slice().reverse().map(([t,x])=>`<li><span class="mut">${new Date(t*1000).toLocaleTimeString()}</span> ${esc(x)}</li>`).join('');
}
poll();
</script></body></html>"""
