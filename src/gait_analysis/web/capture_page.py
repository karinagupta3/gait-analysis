"""Two-phone capture UI for accurate (Pose2Sim) gait trials.

Single self-contained page string ``CAPTURE_BODY``, rendered through the existing
``_shell()`` in :mod:`gait_analysis.web.app` so it inherits BASE_CSS and the nav. It
mirrors the single-phone ``_RECORD_BODY`` flow (live preview + MediaRecorder, mp4/webm
sniffing, multipart upload) but coordinates TWO phones around a shared 6-char SESSION
CODE so cam1 + cam2 clips land in the same trial.

The operator opens this page on EACH phone:
  1. make/enter the SESSION CODE (shown big so it can be typed into the second phone),
  2. pick CAMERA ROLE (cam1 / cam2) and CLIP KIND (calibration / trial),
  3. record (rear camera, audio ON so the clap can audio-sync the two videos),
  4. on stop the clip POSTs to ``/capture-upload`` as multipart
     (fields: code, role, kind, video).

The page then polls ``GET /capture-status?code=...`` to show which of the 4 clips
(cam1/cam2 x calibration/trial) have arrived; once all 4 are present it offers a
"Process trial" button that POSTs the code to ``/capture-run``.

This module only defines the page string; the routes (/capture, /capture-upload,
/capture-status, /capture-run) are wired in app.py. No existing files are modified.
"""

from __future__ import annotations

# Self-contained capture page body. Pass to _shell("Capture (2 phones)", CAPTURE_BODY,
# "capture"). Uses only classes already defined in app.BASE_CSS (.hero/.lead/.banner/
# .card/.row2/.btn/.note/.steps/.badge/.ok/.bad/.empty/code) so the look matches.
CAPTURE_BODY = """<section class="hero"><h1>Two-phone capture</h1>
<p class="lead">Accurate (Pose2Sim) mode: two phones film the same walk from different
angles. Open this page ON EACH phone, share one <b>session code</b>, then record the four
clips below. Everything uploads automatically &mdash; no files to manage.</p></section>

<div class="steps">
<h3>Clinic setup (read before capturing)</h3>
<ul>
<li><b>Two phones, ~60&deg; apart</b> on tripods, both seeing the <b>whole body</b>
(head to feet) for the entire movement; don't pan or zoom. Start both before the subject moves.</li>
<li><b>Calibration first:</b> with both cameras rolling, slowly move a <b>printed
checkerboard</b> through the capture space so it is clearly visible to <b>both</b> cameras.
This is required for metric (real-world) scale.</li>
<li><b>Trial:</b> have the subject <b>CLAP once on camera at the very start</b>, then walk
4&ndash;6 strides. The clap is the audio marker that lets the two videos be synced &mdash;
that is why audio stays ON.</li>
<li>Use the <b>same session code on both phones</b>. On phone&nbsp;1 set role <b>cam1</b>;
on phone&nbsp;2 set role <b>cam2</b>. Record the <b>calibration</b> clip and the
<b>trial</b> clip on each phone &mdash; four clips total.</li>
</ul></div>

<div class="banner" style="background:#f0f9ff;border-color:#bae6fd">
<b>Session code</b>
<p class="note" style="margin:0 0 10px">Make a code on the first phone, then type the same
code into the second phone so both clips join the same trial.</p>
<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
<input id="code" maxlength="6" autocapitalize="characters" autocomplete="off" spellcheck="false"
 placeholder="ABC123" style="max-width:220px;font-size:30px;font-weight:700;letter-spacing:.18em;text-align:center;text-transform:uppercase">
<button class="btn" id="newcode" type="button" style="margin:0">New code</button>
</div>
<p class="note" id="codehint" style="margin:10px 0 0">Enter or generate a 6-character code to begin.</p>
</div>

<div class="card">
  <div class="row2">
    <div><label>Camera role</label>
      <select id="role">
        <option value="cam1">cam1 (this is phone 1)</option>
        <option value="cam2">cam2 (this is phone 2)</option>
      </select></div>
    <div><label>Clip kind</label>
      <select id="kind">
        <option value="calibration">calibration (checkerboard)</option>
        <option value="trial">trial (the walk + clap)</option>
      </select></div>
  </div>
  <label>Camera</label>
  <video id="preview" autoplay muted playsinline style="width:100%;border-radius:10px;background:#000;max-height:60vh"></video>
  <div style="display:flex;gap:10px;margin-top:12px">
    <button class="btn" id="start" type="button" style="margin:0">&#9679; Record</button>
    <button class="btn" id="stop" type="button" disabled style="margin:0;background:#b91c1c">&#9632; Stop &amp; upload</button>
  </div>
  <p class="note" id="st" style="margin-top:12px">Requesting camera&hellip;</p>
</div>

<h2 class="section">Clips received for this code</h2>
<div class="banner" id="checklist" style="background:#f8fafc">
<div class="note">Enter a session code to see which clips have arrived.</div>
</div>
<div id="runwrap" style="display:none">
  <form id="runform" action="/capture-run" method="post">
    <input type="hidden" name="code" id="runcode">
    <button class="btn" type="submit">Process trial &rarr;</button>
  </form>
  <p class="note" style="margin-top:8px">All four clips are in. Processing syncs the two
  videos on the clap, calibrates from the checkerboard, and builds the report.</p>
</div>

<script>
const $=id=>document.getElementById(id), st=$('st');
let rec, chunks=[], stream, statusTimer;

// 6-char code: A-Z + 2-9 (no 0/O/1/I to avoid mistyping across phones).
function makeCode(){
  const a='ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; let s='';
  for(let i=0;i<6;i++) s+=a[Math.floor(Math.random()*a.length)];
  return s;
}
function currentCode(){ return ($('code').value||'').trim().toUpperCase(); }
$('code').addEventListener('input',()=>{
  $('code').value=$('code').value.toUpperCase().replace(/[^A-Z0-9]/g,'').slice(0,6);
  refreshStatus();
});
$('newcode').onclick=()=>{ $('code').value=makeCode(); refreshStatus(); };

async function init(){
  try{
    // audio:true on purpose — the start-of-trial clap is the audio sync marker.
    stream = await navigator.mediaDevices.getUserMedia(
      {video:{facingMode:'environment',width:{ideal:1280},height:{ideal:720}},audio:true});
    $('preview').srcObject = stream; st.textContent='Ready. Set the code + role + kind, then Record.';
  }catch(e){ st.innerHTML='Camera unavailable ('+e.message+'). Use <a href="/process">Process video</a> to upload a clip instead.'; }
}

$('start').onclick=()=>{
  if(!stream) return;
  if(currentCode().length!==6){ st.textContent='Enter a 6-character session code first.'; return; }
  chunks=[]; let mt='video/webm';
  if(window.MediaRecorder && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported('video/mp4')) mt='video/mp4';
  try{ rec=new MediaRecorder(stream,{mimeType:mt}); }catch(e){ rec=new MediaRecorder(stream); }
  rec.ondataavailable=e=>{ if(e.data&&e.data.size) chunks.push(e.data); };
  rec.start(); $('start').disabled=true; $('stop').disabled=false;
  st.textContent='Recording '+$('role').value+' / '+$('kind').value+'…';
};

$('stop').onclick=()=>{
  if(!rec) return;
  rec.onstop=async()=>{
    const type=(chunks[0]&&chunks[0].type)||'video/webm';
    const ext=type.indexOf('mp4')>=0?'mp4':'webm';
    const blob=new Blob(chunks,{type});
    const fd=new FormData();
    fd.append('code', currentCode());
    fd.append('role', $('role').value);
    fd.append('kind', $('kind').value);
    fd.append('video', blob, $('role').value+'_'+$('kind').value+'.'+ext);
    st.textContent='Uploading '+$('role').value+' / '+$('kind').value+'…';
    try{
      const r=await fetch('/capture-upload',{method:'POST',body:fd});
      if(!r.ok) throw new Error('HTTP '+r.status);
      st.textContent='Uploaded '+$('role').value+' / '+$('kind').value+'. Record the next clip or switch phones.';
      refreshStatus();
    }catch(e){ st.textContent='Upload failed: '+e.message; }
  };
  rec.stop(); $('start').disabled=false; $('stop').disabled=true;
};

// --- checklist of the 4 expected clips (cam1/cam2 x calibration/trial) ---
const EXPECTED=[
  ['cam1','calibration'],['cam2','calibration'],
  ['cam1','trial'],['cam2','trial']
];
function renderChecklist(received){
  let html='', have=0;
  for(const [role,kind] of EXPECTED){
    const key=role+'_'+kind, ok=!!(received&&received[key]);
    if(ok) have++;
    const mark=ok?'&#10003;':'&#9633;';
    const cls=ok?'ok':'note';
    html+='<div class="'+cls+'" style="margin:3px 0">'+mark+' '+role+' &middot; '+kind+
          (ok?' &mdash; received':' &mdash; waiting')+'</div>';
  }
  $('checklist').innerHTML=html+'<div class="note" style="margin-top:8px">'+have+' of 4 clips received.</div>';
  const all=have===4;
  $('runwrap').style.display=all?'block':'none';
  $('runcode').value=currentCode();
}
async function refreshStatus(){
  const code=currentCode();
  if(code.length!==6){
    $('checklist').innerHTML='<div class="note">Enter a session code to see which clips have arrived.</div>';
    $('runwrap').style.display='none';
    $('codehint').textContent='Enter or generate a 6-character code to begin.';
    return;
  }
  $('codehint').textContent='Using session code '+code+' — type this same code on the other phone.';
  try{
    const r=await fetch('/capture-status?code='+encodeURIComponent(code));
    const j=await r.json();
    renderChecklist(j.received||{});
  }catch(e){ /* keep last view on transient errors */ }
}

// Poll status so the operator sees the other phone's uploads appear live.
function startPolling(){ clearInterval(statusTimer); statusTimer=setInterval(refreshStatus,4000); }

init();
startPolling();
refreshStatus();
</script>"""
