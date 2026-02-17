#!/usr/bin/env python3

import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# ================= ROS NODE =================
class WebCommandNode(Node):
    def __init__(self):
        super().__init__('web_command_node')
        self.pub = self.create_publisher(String, '/robot_cmd', 10)

    def send(self, cmd):
        msg = String()
        msg.data = cmd
        self.pub.publish(msg)

# ================= FASTAPI =================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ros_node = None

# ================= UI =================
@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
<title>Robot Drive HMI</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<style>
body { background:#111; color:#eee; font-family:Arial; text-align:center; touch-action:none; }
button { width:120px; height:44px; font-size:15px; margin:4px; }
.active { outline:4px solid #fff; }

#park { background:#900; }
#neutral { background:#555; }
#drive { background:#090; }
#reverse { background:#046; }

.panel { border:1px solid #333; padding:10px; margin:10px; }
input[type=range] { width:340px; }

.value { font-size:18px; margin-top:4px; }
.wheel { font-size:16px; margin-top:4px; }
.warn { color:#ff6666; font-size:16px; margin-top:6px; }

.biasBtn { width:60px; font-size:20px; }
</style>
</head>

<body>
<h2>Robot Drive Control</h2>

<div class="panel">
<h3>Gear</h3>
<button id="park">PARK</button>
<button id="neutral">NEUTRAL</button>
<button id="drive">DRIVE</button>
<button id="reverse">REVERSE</button>
<div class="warn" id="gearWarn"></div>
</div>

<div class="panel">
<h3>Speed</h3>
<input type="range" min="0" max="100" value="0" id="speed">
<div class="value">Base Speed: <span id="speedVal">0</span></div>
</div>

<div class="panel">
<h3>Bias (±)</h3>
Left
<button class="biasBtn" id="lbMinus">−</button>
<button class="biasBtn" id="lbPlus">+</button>
<span id="lbVal">0</span>

<br><br>

Right
<button class="biasBtn" id="rbMinus">−</button>
<button class="biasBtn" id="rbPlus">+</button>
<span id="rbVal">0</span>
</div>

<div class="panel">
<h3>Steering Wheel</h3>
<input type="range" min="-30" max="30" value="0" id="steer">
<div class="value">Steer: <span id="steerVal">0</span></div>
<div class="wheel">Left Wheel: <span id="lw">0</span></div>
<div class="wheel">Right Wheel: <span id="rw">0</span></div>
</div>

<script>
/* ---------- STATE ---------- */
let gear='P', speed=0, lb=0, rb=0, steer=0;
let biasTimer=null;

/* ---------- DOM ---------- */
const speedSlider = document.getElementById("speed");
const steerSlider = document.getElementById("steer");
const speedVal = document.getElementById("speedVal");
const steerVal = document.getElementById("steerVal");
const lbVal = document.getElementById("lbVal");
const rbVal = document.getElementById("rbVal");
const lw = document.getElementById("lw");
const rw = document.getElementById("rw");
const gearWarn = document.getElementById("gearWarn");

const gearBtns = {
  P: document.getElementById("park"),
  N: document.getElementById("neutral"),
  D: document.getElementById("drive"),
  R: document.getElementById("reverse")
};

/* ---------- SERIAL ---------- */
function send(cmd){ fetch(`/cmd?c=${encodeURIComponent(cmd)}`); }

/* ---------- GEAR ---------- */
function highlightGear(g){
  Object.values(gearBtns).forEach(b=>b.classList.remove("active"));
  gearBtns[g].classList.add("active");
}
function requestGear(t){
  gearWarn.innerText="";
  if((gear==='D'&&t==='R')||(gear==='R'&&t==='D')){
    gearWarn.innerText="⚠ Shift to NEUTRAL before reversing";
    applyGear('N'); return;
  }
  applyGear(t);
}
function applyGear(g){
  gear=g; highlightGear(g);
  if(g==='P'||g==='N'){
    speed=0; speedSlider.value=0; speedVal.innerText=0;
    send("STOP");
  }
  compute();
}

/* ---------- SPEED ---------- */
speedSlider.oninput=()=>{
  speed=parseInt(speedSlider.value);
  speedVal.innerText=speed;
  compute();
};

/* ---------- BIAS (CORRECT FIX) ---------- */
function setupBias(btn, side, delta){
  btn.addEventListener("pointerdown", e=>{
    e.preventDefault();
    btn.setPointerCapture(e.pointerId);
    if(biasTimer) return;
    biasTimer=setInterval(()=>{
      if(side==='lb') lb+=delta;
      else rb+=delta;
      lbVal.innerText=lb;
      rbVal.innerText=rb;
      compute();
    },120);
  });

  btn.addEventListener("pointerup", stopBias);
  btn.addEventListener("pointercancel", stopBias);
  btn.addEventListener("pointerleave", stopBias);
}

function stopBias(){
  if(biasTimer){
    clearInterval(biasTimer);
    biasTimer=null;
  }
}

/* ---------- STEERING ---------- */
steerSlider.oninput=()=>{
  steer=parseInt(steerSlider.value);
  steerVal.innerText=steer;
  compute();
};
steerSlider.onpointerup=steerSlider.onpointerleave=()=>{
  steer=0; steerSlider.value=0; steerVal.innerText=0;
  compute();
};

/* ---------- COMPUTE ---------- */
function compute(){
  if(speed===0||gear==='P'||gear==='N'){
    lw.innerText=0; rw.innerText=0;
    send("STOP"); return;
  }
  let left=speed+lb-steer;
  let right=speed+rb+steer;
  left=Math.max(0,Math.min(100,left));
  right=Math.max(0,Math.min(100,right));
  lw.innerText=left; rw.innerText=right;
  if(gear==='D'){ send(`LF ${left}`); send(`RF ${right}`); }
  else{ send(`LR ${left}`); send(`RR ${right}`); }
}

/* ---------- INIT ---------- */
gearBtns.P.onclick=()=>requestGear('P');
gearBtns.N.onclick=()=>requestGear('N');
gearBtns.D.onclick=()=>requestGear('D');
gearBtns.R.onclick=()=>requestGear('R');
highlightGear('P');

setupBias(lbMinus,'lb',-1);
setupBias(lbPlus,'lb',1);
setupBias(rbMinus,'rb',-1);
setupBias(rbPlus,'rb',1);
</script>

</body>
</html>
""")

@app.get("/cmd")
def cmd(c:str):
    if ros_node: ros_node.send(c)
    return {"ok":True}

def ros_spin(): rclpy.spin(ros_node)

def main():
    global ros_node
    rclpy.init()
    ros_node=WebCommandNode()
    threading.Thread(target=ros_spin,daemon=True).start()
    uvicorn.run(app,host="0.0.0.0",port=8000)
    ros_node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()
