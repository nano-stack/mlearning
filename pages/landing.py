"""
Landing page — animación 3b1b style en canvas HTML/JS + bienvenida.
"""
import streamlit as st
import streamlit.components.v1 as components


MANIM_ANIMATION = """
<style>
  body { margin:0; background:#0d1b2a; }
  canvas { display:block; }
</style>
<canvas id="c"></canvas>
<script>
const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
canvas.width  = window.innerWidth  || 900;
canvas.height = window.innerHeight || 520;
const W = canvas.width, H = canvas.height;

// ── Paleta 3b1b ──────────────────────────────────────────────
const GOLD    = '#f0a500';
const BLUE    = '#4a7ec8';
const GREEN   = '#2e7d5a';
const RED     = '#c0392b';
const WHITE   = 'rgba(255,255,255,0.85)';
const MUTED   = 'rgba(255,255,255,0.18)';
const BG      = '#0d1b2a';

// ── Estado global ─────────────────────────────────────────────
let t      = 0;       // tiempo global
let phase  = 0;       // 0=regression 1=clusters 2=pca
let phaseT = 0;       // tiempo dentro de la fase
const PHASE_DUR = 280;

// ── Datos sintéticos ──────────────────────────────────────────
function seededRand(seed) {
  let s = seed;
  return () => { s=(s*16807+0)%2147483647; return (s-1)/2147483646; };
}

const rng = seededRand(42);
const N = 40;
const pts = Array.from({length:N}, (_,i) => ({
  x: (rng()*2-1)*0.72,
  y: (rng()*2-1)*0.72,
  ox: 0, oy: 0,
  cx: 0, cy: 0,
  cluster: 0
}));

// Regression: y ≈ 0.7x + noise
pts.forEach(p => { p.ox=p.x; p.oy = 0.7*p.x + (rng()-0.5)*0.35; });

// Clusters
const clusterCenters = [[-0.45,-0.35],[0.35,0.38],[0.05,-0.52]];
const clusterColors  = [BLUE, GREEN, RED];
pts.forEach((p,i) => {
  const ci = i % 3;
  p.cluster = ci;
  const [ccx,ccy] = clusterCenters[ci];
  p.cx = ccx + (rng()-0.5)*0.28;
  p.cy = ccy + (rng()-0.5)*0.28;
});

// ── Helpers ────────────────────────────────────────────────────
function toScreen(nx, ny) {
  return [W/2 + nx*(W*0.38), H/2 - ny*(H*0.38)];
}

function lerp(a,b,t){ return a+(b-a)*Math.min(1,Math.max(0,t)); }
function easeInOut(t){ return t<.5?2*t*t:(4-2*t)*t-1; }

function drawAxes(alpha=1) {
  ctx.save();
  ctx.globalAlpha = alpha * 0.35;
  ctx.strokeStyle = '#4a7ec8';
  ctx.lineWidth   = 1.5;
  // X axis
  const [x0,y0] = toScreen(-1, 0);
  const [x1,y1] = toScreen( 1, 0);
  ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x1,y1); ctx.stroke();
  // Y axis
  const [ax,ay] = toScreen(0,-1);
  const [bx,by] = toScreen(0, 1);
  ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(bx,by); ctx.stroke();
  // Ticks
  for(let i=-4;i<=4;i++){
    if(i===0) continue;
    const [tx,ty] = toScreen(i*0.25, 0);
    ctx.beginPath(); ctx.moveTo(tx,ty-4); ctx.lineTo(tx,ty+4); ctx.stroke();
    const [ux,uy] = toScreen(0, i*0.25);
    ctx.beginPath(); ctx.moveTo(ux-4,uy); ctx.lineTo(ux+4,uy); ctx.stroke();
  }
  ctx.restore();
}

function drawDot(nx, ny, color, r=5, alpha=1) {
  const [sx,sy] = toScreen(nx,ny);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle   = color;
  ctx.shadowColor = color;
  ctx.shadowBlur  = 10;
  ctx.beginPath();
  ctx.arc(sx, sy, r, 0, Math.PI*2);
  ctx.fill();
  ctx.restore();
}

function drawLine(x0,y0,x1,y1, color, width=2, alpha=1) {
  const [sx0,sy0] = toScreen(x0,y0);
  const [sx1,sy1] = toScreen(x1,y1);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth   = width;
  ctx.shadowColor = color;
  ctx.shadowBlur  = 8;
  ctx.beginPath(); ctx.moveTo(sx0,sy0); ctx.lineTo(sx1,sy1); ctx.stroke();
  ctx.restore();
}

function drawText(text, nx, ny, color, size=14, alpha=1, align='center') {
  const [sx,sy] = toScreen(nx,ny);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle   = color;
  ctx.font        = `600 ${size}px "SF Pro Display", system-ui, sans-serif`;
  ctx.textAlign   = align;
  ctx.fillText(text, sx, sy);
  ctx.restore();
}

function drawStar(nx, ny, color, r=8, alpha=1) {
  const [sx,sy] = toScreen(nx,ny);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle   = color;
  ctx.shadowColor = color;
  ctx.shadowBlur  = 16;
  ctx.beginPath();
  for(let i=0;i<10;i++){
    const angle = (i*Math.PI/5) - Math.PI/2;
    const rad   = i%2===0 ? r : r*0.45;
    const px = sx + Math.cos(angle)*rad;
    const py = sy + Math.sin(angle)*rad;
    i===0 ? ctx.moveTo(px,py) : ctx.lineTo(px,py);
  }
  ctx.closePath(); ctx.fill();
  ctx.restore();
}

function drawEquation(text, x, y, alpha, size=15) {
  ctx.save();
  ctx.globalAlpha = alpha * 0.55;
  ctx.fillStyle   = WHITE;
  ctx.font        = `italic ${size}px "Georgia", serif`;
  ctx.textAlign   = 'left';
  ctx.fillText(text, x, y);
  ctx.restore();
}

// ── Fases ──────────────────────────────────────────────────────

function phaseRegression(pt) {
  const appear = easeInOut(Math.min(1, pt/60));
  const lineGrow = easeInOut(Math.min(1, Math.max(0,(pt-40)/80)));
  const resid  = easeInOut(Math.min(1, Math.max(0,(pt-90)/60)));
  const labelA = easeInOut(Math.min(1, Math.max(0,(pt-130)/40)));

  drawAxes(appear);

  // Residual lines
  pts.forEach(p => {
    const predicted = 0.7 * p.ox;
    const [sx,sy]   = toScreen(p.ox, p.oy);
    const [px,py]   = toScreen(p.ox, predicted);
    ctx.save();
    ctx.globalAlpha = resid * 0.4;
    ctx.strokeStyle = RED;
    ctx.lineWidth   = 1;
    ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(sx,sy); ctx.lineTo(px,py); ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  });

  // Dots
  pts.forEach((p,i) => {
    const da = easeInOut(Math.min(1, Math.max(0,(pt - i*2)/30)));
    drawDot(p.ox, p.oy, BLUE, 5, appear * da);
  });

  // Regression line
  if(lineGrow > 0) {
    const x0 = lerp(0, -0.95, lineGrow);
    const x1 = lerp(0,  0.95, lineGrow);
    drawLine(x0, 0.7*x0, x1, 0.7*x1, GOLD, 2.5, lineGrow);
  }

  // Label
  drawText('ŷ = β₀ + β₁x', 0.55, 0.65, GOLD, 14, labelA);
  drawText('Regresión Lineal', 0, 0.95, WHITE, 17, labelA);

  // MSE label
  if(resid > 0.5) {
    drawText('MSE = Σ(y - ŷ)²', -0.75, 0.85, RED, 12, resid*0.8);
  }
}

function phaseClusters(pt) {
  const appear = easeInOut(Math.min(1, pt/50));
  const morph  = easeInOut(Math.min(1, pt/90));
  const centA  = easeInOut(Math.min(1, Math.max(0,(pt-80)/60)));
  const labelA = easeInOut(Math.min(1, Math.max(0,(pt-110)/40)));

  drawAxes(appear * 0.5);

  pts.forEach(p => {
    const nx = lerp(p.ox, p.cx, morph);
    const ny = lerp(p.oy, p.cy, morph);
    const col = clusterColors[p.cluster];
    drawDot(nx, ny, col, 5, appear);
  });

  // Centroids
  clusterCenters.forEach(([cx,cy],i) => {
    drawStar(cx, cy, clusterColors[i], 10, centA);
    // Circle around cluster
    const [sx,sy] = toScreen(cx,cy);
    ctx.save();
    ctx.globalAlpha = centA * 0.15;
    ctx.strokeStyle = clusterColors[i];
    ctx.lineWidth   = 1.5;
    ctx.beginPath();
    ctx.arc(sx, sy, H*0.13, 0, Math.PI*2);
    ctx.stroke();
    ctx.restore();
  });

  drawText('K-Means Clustering', 0, 0.95, WHITE, 17, labelA);
  drawText('J = Σ||xᵢ − μₖ||²', 0, -0.9, GOLD, 14, labelA);

  // Lines from points to centroids
  if(centA > 0.3) {
    pts.forEach(p => {
      const [cx,cy] = clusterCenters[p.cluster];
      ctx.save();
      ctx.globalAlpha = centA * 0.12;
      ctx.strokeStyle = clusterColors[p.cluster];
      ctx.lineWidth   = 0.8;
      const [sx,sy]  = toScreen(p.cx,p.cy);
      const [ex,ey]  = toScreen(cx,cy);
      ctx.beginPath(); ctx.moveTo(sx,sy); ctx.lineTo(ex,ey); ctx.stroke();
      ctx.restore();
    });
  }
}

function phasePCA(pt) {
  const appear = easeInOut(Math.min(1, pt/50));
  const pc1A   = easeInOut(Math.min(1, Math.max(0,(pt-50)/60)));
  const pc2A   = easeInOut(Math.min(1, Math.max(0,(pt-90)/50)));
  const projA  = easeInOut(Math.min(1, Math.max(0,(pt-130)/60)));
  const labelA = easeInOut(Math.min(1, Math.max(0,(pt-120)/40)));

  drawAxes(appear * 0.5);

  // Original dots
  pts.forEach(p => {
    const nx = lerp(p.cx, p.ox*0.7+p.oy*0.12, appear);
    const ny = lerp(p.cy, p.ox*0.12+p.oy*0.6,  appear);
    drawDot(nx, ny, BLUE, 5, appear * (1 - projA*0.7));
  });

  // PC1 arrow (diagonal)
  if(pc1A > 0) {
    const len = 0.8 * pc1A;
    drawLine(-len*0.707, -len*0.707, len*0.707, len*0.707, GOLD, 3, pc1A);
    drawText('PC₁', len*0.707+0.07, len*0.707+0.05, GOLD, 14, pc1A);
  }

  // PC2 arrow (perpendicular)
  if(pc2A > 0) {
    const len = 0.4 * pc2A;
    drawLine(-len*0.707, len*0.707, len*0.707, -len*0.707, RED, 2, pc2A);
    drawText('PC₂', len*0.707+0.07, -len*0.707-0.05, RED, 13, pc2A);
  }

  // Projected dots on PC1
  if(projA > 0) {
    pts.forEach(p => {
      const v  = p.ox*0.707 + p.oy*0.707;
      const px = v * 0.707;
      const py = v * 0.707;
      drawDot(px, py, GOLD, 4.5, projA * 0.85);
      // projection line
      const [sx,sy] = toScreen(p.ox*0.7+p.oy*0.12, p.ox*0.12+p.oy*0.6);
      const [ex,ey] = toScreen(px, py);
      ctx.save();
      ctx.globalAlpha = projA * 0.25;
      ctx.strokeStyle = WHITE;
      ctx.lineWidth   = 0.8;
      ctx.setLineDash([3,3]);
      ctx.beginPath(); ctx.moveTo(sx,sy); ctx.lineTo(ex,ey); ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    });
  }

  drawText('PCA — Reducción Dimensional', 0, 0.95, WHITE, 17, labelA);
  drawText('X = UΣVᵀ', 0, -0.9, GOLD, 14, labelA);
}

// ── Ecuaciones flotantes de fondo ──────────────────────────────
const floatEqs = [
  { text:'σ(z)=1/(1+e⁻ᶻ)', x:0.06, y:0.06, spd:0.0007, amp:12 },
  { text:'∇L(w)=0',          x:0.78, y:0.08, spd:0.0009, amp:10 },
  { text:'R²=1−SSᵣ/SSₜ',    x:0.82, y:0.82, spd:0.0006, amp:14 },
  { text:'max 2/||w||',       x:0.05, y:0.85, spd:0.0008, amp:11 },
];

// ── Loop principal ─────────────────────────────────────────────
function draw() {
  ctx.clearRect(0,0,W,H);

  // Fondo gradiente
  const grad = ctx.createRadialGradient(W/2,H/2,0, W/2,H/2,W*0.7);
  grad.addColorStop(0, '#0f2035');
  grad.addColorStop(1, '#070f1a');
  ctx.fillStyle = grad;
  ctx.fillRect(0,0,W,H);

  // Ecuaciones flotantes de fondo
  floatEqs.forEach(eq => {
    const fy = Math.sin(t * eq.spd * 1000 + eq.x * 10) * eq.amp;
    const alpha = 0.07 + 0.04 * Math.sin(t * eq.spd * 500);
    drawEquation(eq.text, eq.x * W, eq.y * H + fy, alpha, 13);
  });

  // Fase actual
  const pt = phaseT;
  if(phase === 0)      phaseRegression(pt);
  else if(phase === 1) phaseClusters(pt);
  else                 phasePCA(pt);

  // Transición suave entre fases (fade out al final)
  if(phaseT > PHASE_DUR - 40) {
    const fadeOut = (phaseT - (PHASE_DUR-40)) / 40;
    ctx.save();
    ctx.globalAlpha = Math.min(1, fadeOut);
    ctx.fillStyle   = BG;
    ctx.fillRect(0,0,W,H);
    ctx.restore();
  }

  t      += 0.016;
  phaseT += 1;
  if(phaseT >= PHASE_DUR) {
    phaseT = 0;
    phase  = (phase + 1) % 3;
  }

  requestAnimationFrame(draw);
}

draw();
</script>
"""


def render():
    # ── Layout: animación izquierda, texto derecha ─────────────────────────
    st.markdown("""
    <style>
    .stApp { background: #0d1b2a !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    section[data-testid="stSidebar"] { display:none; }
    </style>
    """, unsafe_allow_html=True)

    col_anim, col_text = st.columns([1.15, 1])

    with col_anim:
        components.html(MANIM_ANIMATION, height=560, scrolling=False)

    with col_text:
        RIGHT_PANEL = """
        <style>
          * { box-sizing: border-box; margin:0; padding:0; font-family: system-ui, -apple-system, sans-serif; }
          body { background: transparent; }
          .panel { padding: 60px 32px 20px 8px; }
          .badge {
            display: inline-block;
            background: rgba(240,165,0,0.12);
            border: 1px solid rgba(240,165,0,0.35);
            border-radius: 20px;
            padding: 4px 16px;
            font-size: 11px; font-weight: 700;
            color: #f0a500; letter-spacing: 1.5px;
            text-transform: uppercase; margin-bottom: 22px;
          }
          .logo { font-size: 54px; font-weight: 900; color: white; letter-spacing: -2px; line-height: 1.05; margin-bottom: 14px; }
          .logo span { color: #f0a500; }
          .tagline { font-size: 18px; font-weight: 600; color: rgba(255,255,255,0.85); line-height: 1.5; margin-bottom: 10px; }
          .sub { font-size: 13px; color: rgba(255,255,255,0.45); line-height: 1.75; margin-bottom: 32px; }
          .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 36px; }
          .feat {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px; padding: 14px 16px;
          }
          .feat-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 5px; }
          .feat-desc  { font-size: 12px; color: rgba(255,255,255,0.55); line-height: 1.5; }
          .btn {
            display: inline-block;
            background: #f0a500; color: #0d1b2a;
            border: none; border-radius: 12px;
            font-size: 16px; font-weight: 700;
            padding: 14px 40px; cursor: pointer;
            box-shadow: 0 8px 32px rgba(240,165,0,0.3);
            text-decoration: none;
          }
          .btn:hover { background: #e09400; }
        </style>
        <div class="panel">
          <div class="badge">Machine Learning · Visual · Interactivo</div>
          <div class="logo">ML<span>earning</span></div>
          <div class="tagline">La app definitiva para aprender Machine Learning</div>
          <div class="sub">Entrena modelos reales · Visualiza cómo aprenden · Comprende cada hiperparámetro · Genera código listo para usar.</div>
          <div class="grid">
            <div class="feat">
              <div class="feat-label" style="color:#4a7ec8">Visualización</div>
              <div class="feat-desc">Animaciones matemáticas estilo 3Blue1Brown</div>
            </div>
            <div class="feat">
              <div class="feat-label" style="color:#f0a500">Hiperparámetros</div>
              <div class="feat-desc">Cada parámetro explicado con efecto y regla práctica</div>
            </div>
            <div class="feat">
              <div class="feat-label" style="color:#2e7d5a">Código</div>
              <div class="feat-desc">Script Python generado y descargable</div>
            </div>
            <div class="feat">
              <div class="feat-label" style="color:#a8c4e0">Sin API externa</div>
              <div class="feat-desc">Todo el conocimiento cargado localmente</div>
            </div>
          </div>
        </div>
        """
        components.html(RIGHT_PANEL, height=500, scrolling=False)

        if st.button("Comenzar →", key="landing_continue", use_container_width=False):
            st.session_state.page = "select_type"
            st.rerun()

        st.markdown("""
        <style>
        div[data-testid="stButton"] > button {
          background: #f0a500 !important;
          color: #0d1b2a !important;
          border: none !important;
          border-radius: 12px !important;
          font-size: 16px !important;
          font-weight: 700 !important;
          padding: 14px 48px !important;
          box-shadow: 0 8px 28px rgba(240,165,0,0.35) !important;
          margin-top: -8px;
        }
        div[data-testid="stButton"] > button:hover {
          background: #d99400 !important;
          transform: translateY(-1px) !important;
          box-shadow: 0 12px 36px rgba(240,165,0,0.45) !important;
        }
        </style>
        """, unsafe_allow_html=True)
