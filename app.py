from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler
)
from PIL import Image
from pathlib import Path
import torch, io, base64, os, sys, uvicorn, traceback, time, re

# Gráficos
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000", "http://127.0.0.1:5000"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["ACCELERATE_DISABLE_WEIGHTS_CACHE"] = "1"

BASE_DIR_DEFAULT = r"C:\Users\Linds\Documentos\Arquivos Mauá\Programação\gerador_imagens\models\sdxl\stable-diffusion-xl-base-1.0"
MODEL_BASE_PATH = Path(os.environ.get("MODEL_DIR", BASE_DIR_DEFAULT)).expanduser().resolve()
_raw_refiner = os.environ.get("REFINER_DIR", "").strip()
REFINER_DIR = Path(_raw_refiner).expanduser().resolve() if _raw_refiner else None
TEST_IMAGE_PATH = Path(os.environ.get("TEST_IMAGE_PATH", Path(__file__).parent / "static" / "teste.png"))

DEVICE = os.environ.get("DEVICE", "cpu")
DEFAULT_WIDTH = int(os.environ.get("IMG_WIDTH", "768"))
DEFAULT_HEIGHT = int(os.environ.get("IMG_HEIGHT", "512"))
DEFAULT_STEPS = int(os.environ.get("STEPS", "24"))
DEFAULT_GUIDANCE = float(os.environ.get("GUIDANCE", "5.0"))

STYLE_PRESETS = {
    "infographic": "clean educational infographic, flat minimalist diagram, vector-like shapes, balanced spacing, white background",
    "diagram": "technical instructional diagram, crisp arrows, schematic minimal shapes, limited soft colors, white background",
    "flat": "flat vector minimal illustration, uniform stroke, simple color blocks, high clarity, white background",
    "outline": "thin outline vector diagram, monochrome with one accent color, instructional style, white background",
    "photo": "ultra detailed professional studio photograph, realistic lighting, natural colors, high clarity",
    "macro": "extreme macro photography, shallow depth of field, crisp texture, bokeh background",
    "painterly": "rich painterly illustration, textured brush strokes, dramatic lighting, vibrant harmonious palette",
    "digital_art": "high detail digital concept art, dynamic lighting, soft gradients, cinematic composition"
}

QUALITY_PRESETS = {
    "standard": {"guidance": 5.0, "steps": 24},
    "focused": {"guidance": 4.8, "steps": 32},
    "rapid": {"guidance": 4.2, "steps": 18},
}

NEGATIVE_CORE = (
    "low quality, bad quality, blurry, noisy, chaotic layout, distorted, deformed, misshapen, watermark, logo, "
    "long paragraph text, grain, jpeg artifacts, clutter, oversaturated, 3d render, realistic shadow"
)
NEGATIVE_SHAPES = "broken geometry, overlapping elements, duplicated parts, extra limbs, malformed hands, distorted arrows"
NEGATIVE_TEXT = "large text block, random letters, messy handwriting, illegible text, numbers scattered"
DEFAULT_NEGATIVE = f"{NEGATIVE_CORE}, {NEGATIVE_SHAPES}, {NEGATIVE_TEXT}"

class GenReq(BaseModel):
    prompt: str
    subject: str | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance: float | None = None
    style: str | None = "infographic"
    quality: str | None = "standard"
    negative: str | None = None
    seed: int | None = None
    highres: bool | None = False
    refiner: bool | None = True
    precise: bool | None = False
    mode: str | None = None
    ultra: bool | None = False
    ensemble: int | None = 1
    raw: bool | None = False

def _dtype():
    return torch.float16 if (DEVICE == "cuda" and torch.cuda.is_available()) else torch.float32

def _has_model_index(p: Path) -> bool:
    return p.is_dir() and (p / "model_index.json").exists()

def load_base(path: Path):
    if not path.exists():
        raise RuntimeError(f"Modelo base não encontrado: {path}")
    print(f"[LOAD] BASE: {path}")
    if path.is_dir():
        pipe = StableDiffusionXLPipeline.from_pretrained(
            str(path),
            torch_dtype=_dtype(),
            local_files_only=True,
        )
    else:
        raise RuntimeError("Forneça pasta diffusers.")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="sde-dpmsolver++"
    )
    pipe.to("cuda" if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu")
    pipe.enable_attention_slicing()
    return pipe

def load_refiner(path: Path | None):
    if not path or not path.exists():
        print("[REFINER] não definido.")
        return None
    if not _has_model_index(path):
        print(f"[REFINER] ignorado (sem model_index.json): {path}")
        return None
    print(f"[LOAD] REFINER: {path}")
    try:
        pr = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            str(path),
            torch_dtype=_dtype(),
            local_files_only=True,
        )
        pr.to("cuda" if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu")
        return pr
    except Exception as e:
        print("[REFINER] falha:", e)
        return None

try:
    pipe_base = load_base(MODEL_BASE_PATH)
    pipe_refiner = load_refiner(REFINER_DIR)
except Exception as e:
    print("[ERRO] Falha inicial:", e, file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

print(f"[OK] Base pronta | device={DEVICE} | refiner={'on' if pipe_refiner else 'off'}")

# ---------------- Gráficos utilitários ----------------

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', facecolor='white', bbox_inches='tight', dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def chart_mru_png_b64(s0=0.0, v=2.0, t_max=10.0, title="MRU: s(t) = s0 + v·t"):
    t = np.linspace(0.0, float(t_max), 200)
    s = float(s0) + float(v) * t
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.set_facecolor('white'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.plot(t, s, color='#ED1E52', linewidth=3, label=f's(t) = {s0:g} + {v:g}·t')
    ax.set_xlabel('tempo t (s)'); ax.set_ylabel('posição s (m)')
    ax.set_title(title, fontsize=14, pad=12, fontweight='bold')
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
    ax.legend(loc='upper left', frameon=False)
    t1, t2 = 0.15 * t_max, 0.55 * t_max
    s1, s2 = s0 + v * t1, s0 + v * t2
    ax.annotate('', xy=(t2, s2), xytext=(t1, s1), arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=2))
    ax.text((t1+t2)/2, (s1+s2)/2, f'v = {v:g} m/s', color='#1f77b4', fontsize=10)
    return _fig_to_b64(fig)

def chart_mruv_png_b64(s0=0.0, v0=0.0, a=1.0, t_max=10.0, title="MRUV: s(t) e v(t)"):
    t = np.linspace(0.0, float(t_max), 200)
    s = float(s0) + float(v0) * t + 0.5 * float(a) * t**2
    v = float(v0) + float(a) * t
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), dpi=150)
    for ax in axes: ax.set_facecolor('white'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    axes[0].plot(t, s, color='#ED1E52', lw=3, label=f's(t) = {s0:g} + {v0:g}·t + 0.5·{a:g}·t²')
    axes[0].set_xlabel('t (s)'); axes[0].set_ylabel('s (m)'); axes[0].grid(True, ls='--', lw=0.6, alpha=0.6); axes[0].legend(loc='upper left', frameon=False)
    axes[1].plot(t, v, color='#1f77b4', lw=3, label=f'v(t) = {v0:g} + {a:g}·t')
    axes[1].set_xlabel('t (s)'); axes[1].set_ylabel('v (m/s)'); axes[1].grid(True, ls='--', lw=0.6, alpha=0.6); axes[1].legend(loc='upper left', frameon=False)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    return _fig_to_b64(fig)

# ---------------- Funções matemáticas (plot) ----------------

SAFE_FUNCS = {
    # numpy functions disponíveis
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "sinh": np.sin, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt,
    "abs": np.abs, "floor": np.floor, "ceil": np.ceil, "pow": np.power,
    "clip": np.clip, "maximum": np.maximum, "minimum": np.minimum,
    "pi": np.pi, "e": np.e
}

def _normalize_expr(expr: str) -> str:
    # normalizações: ^ -> ** ; vírgula decimal -> ponto ; inserir * implícitas
    s = expr.strip()
    s = s.replace('^', '**')
    # trocar vírgulas numéricas por ponto (somente números)
    s = re.sub(r'(\d),(\d)', r'\1.\2', s)
    # inserir * entre número/fecha-parêntese e x/(
    s = re.sub(r'(\d|\))\s*(x|\()', r'\1*\2', s, flags=re.IGNORECASE)
    s = re.sub(r'(x|\))\s*(\d|\()', r'\1*\2', s, flags=re.IGNORECASE)
    # remoção de caracteres proibidos
    if re.search(r"[A-Za-z_]{2,}", s):
        # permite nomes de funções definidas e 'x'
        tokens_ok = set(list(SAFE_FUNCS.keys()) + ["x"])
        for m in re.finditer(r"[A-Za-z_]+", s):
            if m.group(0) not in tokens_ok:
                # invalida nomes estranhos
                pass
    return s

def _safe_eval(expr: str, x: np.ndarray) -> np.ndarray:
    env = dict(SAFE_FUNCS)
    env["x"] = x
    return eval(expr, {"__builtins__": {}}, env)

def parse_function_from_prompt(prompt: str):
    p = prompt.lower()
    # intervalo de x: x in [a,b] | x ∈ [a,b] | de a a b
    x_min, x_max = -10.0, 10.0
    m = re.search(r"x\s*(?:∈|in)?\s*\[\s*(-?\d+(?:[.,]\d+)?)\s*,\s*(-?\d+(?:[.,]\d+)?)\s*\]", p)
    if m:
        x_min, x_max = float(m.group(1).replace(',', '.')), float(m.group(2).replace(',', '.'))
    else:
        m2 = re.search(r"de\s*(-?\d+(?:[.,]\d+)?)\s*a\s*(-?\d+(?:[.,]\d+)?)", p)
        if m2:
            x_min, x_max = float(m2.group(1).replace(',', '.')), float(m2.group(2).replace(',', '.'))

    # expressões explícitas: y=..., f(x)=..., g(x)=...
    exprs = []
    for pat in [r"y\s*=\s*([^;,\n]+)", r"f\s*\(\s*x\s*\)\s*=\s*([^;,\n]+)", r"g\s*\(\s*x\s*\)\s*=\s*([^;,\n]+)"]:
        for m in re.finditer(pat, prompt, flags=re.IGNORECASE):
            exprs.append(m.group(1).strip())

    # palavras-chave (se não há expressão explícita)
    keywords = {
        "linear": "x",
        "afim": "x",
        "quadrática": "x**2",
        "quadratica": "x**2",
        "exponencial": "exp(x)",
        "seno": "sin(x)",
        "cosseno": "cos(x)",
        "coseno": "cos(x)",
        "tangente": "tan(x)",
        "log": "log(x)",
        "logaritmo": "log(x)"
    }
    if not exprs:
        for k, v in keywords.items():
            if k in p:
                exprs.append(v)

    # se falar "funções" no plural e nenhum match, mostra trio padrão
    if ("funções" in p or "funcoes" in p or "funções" in p) and not exprs:
        exprs = ["x", "x**2", "exp(x)"]

    # clean e normaliza
    exprs = [_normalize_expr(e) for e in exprs if e.strip()]

    return exprs, x_min, x_max

def chart_function_png_b64(exprs: list[str], x_min=-10.0, x_max=10.0, title="Função"):
    # múltiplas curvas com legenda
    x = np.linspace(float(x_min), float(x_max), 1000)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.set_facecolor('white'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    colors = ['#ED1E52', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf']
    any_plotted = False
    for i, expr in enumerate(exprs[:6]):
        try:
            y = _safe_eval(expr, x)
            # remove valores não finitos
            mask = np.isfinite(y)
            if mask.sum() == 0:
                continue
            ax.plot(x[mask], y[mask], color=colors[i % len(colors)], lw=2.4, label=f"y = {expr}")
            any_plotted = True
        except Exception as e:
            print(f"[WARN] falha ao avaliar '{expr}': {e}")

    # eixos e grade
    ax.axhline(0, color='#999999', lw=1, alpha=0.6)
    ax.axvline(0, color='#999999', lw=1, alpha=0.6)
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    if any_plotted:
        ax.legend(loc='upper left', frameon=False)
    return _fig_to_b64(fig)

# ---------------- Detecção de gráficos ----------------

class ChartTask(BaseModel):
    kind: str
    params: dict

def _num_from(text: str, key: str, default: float) -> float:
    m = re.search(rf"{key}\s*[:=]?\s*(-?\d+[.,]?\d*)", text)
    if not m: return default
    try: return float(m.group(1).replace(',', '.'))
    except: return default

def detect_chart(prompt: str) -> ChartTask | None:
    p = prompt.lower()

    # Funções: prioridade alta
    if any(w in p for w in ["função", "funcoes", "funções", "f(x)", "g(x)", "y="]):
        exprs, x_min, x_max = parse_function_from_prompt(prompt)
        if exprs:
            return ChartTask(kind="function", params={"exprs": exprs, "x_min": x_min, "x_max": x_max})

    # Gráficos cinemática
    wants_graph = any(w in p for w in ["gráfico", "grafico", "plot", "diagrama", "s vs t", "s x t", "v vs t", "v x t"])
    if ("mruv" in p) or ("uniformemente variado" in p):
        v0 = _num_from(p, "v0", 0.0); a = _num_from(p, "a", 1.0); s0 = _num_from(p, "s0", 0.0)
        tmax = _num_from(p, "tmax", _num_from(p, "t", 10.0))
        return ChartTask(kind="mruv", params={"s0": s0, "v0": v0, "a": a, "t_max": tmax})
    if ("mru" in p) or ("uniforme" in p) or (wants_graph and any(w in p for w in ["s x t", "s vs t", "posição x tempo", "posicao x tempo"])):
        s0 = _num_from(p, "s0", 0.0); v = _num_from(p, "v", 2.0)
        tmax = _num_from(p, "tmax", _num_from(p, "t", 10.0))
        return ChartTask(kind="mru", params={"s0": s0, "v": v, "t_max": tmax})
    if wants_graph and any(w in p for w in ["v x t", "v vs t", "velocidade x tempo"]):
        v0 = _num_from(p, "v0", 2.0); a = _num_from(p, "a", 0.0)
        tmax = _num_from(p, "tmax", _num_from(p, "t", 10.0))
        return ChartTask(kind="mruv", params={"s0": 0.0, "v0": v0, "a": a, "t_max": tmax})
    return None

# ---------------- API ----------------

@app.get("/health")
def health():
    return {"ok": True, "base": str(MODEL_BASE_PATH), "refiner": bool(pipe_refiner), "device": DEVICE}

def _snap(v: int, mult: int = 64, minimum: int = 384, maximum: int = 1024) -> int:
    v = max(minimum, min(maximum, v)); return (v // mult) * mult

def expand_subject(subject: str) -> list[str]:
    s = subject.lower()
    if "newton" in s: return ["inertia object at rest", "cart acceleration F = m * a", "action-reaction skaters pushing"]
    if "eletric" in s or "elet" in s: return ["battery", "resistor", "lamp", "current arrows", "voltage difference"]
    if "cinemat" in s: return ["uniform motion cart", "accelerated motion ramp", "velocity-time graph"]
    if "funções" in s or "functions" in s: return ["linear graph", "quadratic parabola", "exponential curve"]
    if "óptica" in s or "optica" in s or "optics" in s: return ["light ray reflection", "refraction prism", "angle markers"]
    parts = re.split(r"[;,]+", subject); return [p.strip() for p in parts if p.strip()][:6]

_SIMPLE_OBJECT_WORDS = {
    "maçã","maca","apple","cachorro","dog","gato","cat","flor","flower","carro","car","árvore","arvore","tree",
    "livro","book","planeta","planet","estrela","star","montanha","mountain","oceano","ocean","pássaro","passaro","bird",
    "cavaleiro","knight","castelo","castle","dragão","dragon","cidade","city","personagem","character"
}

def is_simple_object_prompt(text: str) -> bool:
    t = text.lower().strip()
    if len(t.split()) <= 22 and not any(w in t for w in ["gráfico","grafico","diagrama","equação","lei","formula","infográfico","infografico"]):
        return any(w in t for w in _SIMPLE_OBJECT_WORDS)
    return False

def enhance_prompt(base_prompt: str, subject: str | None, style_key: str, layout_hint: str = "horizontal multi-panel") -> str:
    components = expand_subject(subject) if subject else []
    comp_text = " Components: " + "; ".join(components) + "." if components else ""
    style_txt = STYLE_PRESETS.get(style_key, STYLE_PRESETS["infographic"])
    template = (
        f"{style_txt}. Educational clarity, balanced spacing, minimal labels, consistent stroke."
        f" Layout: {layout_hint}. Avoid clutter.{comp_text} Simplified shapes only."
    )
    return f"{template} {base_prompt}"

def build_full_prompt(raw: str, style_key: str, subject: str | None, force_raw: bool) -> str:
    if force_raw or is_simple_object_prompt(raw):
        style_txt = STYLE_PRESETS.get(style_key, "")
        if style_txt and style_key not in {"infographic","diagram","outline","flat"}:
            return f"{style_txt}, {raw}"
        return raw
    return enhance_prompt(raw, subject, style_key)

def build_negative(user_neg: str | None, style_key: str) -> str:
    neg = (user_neg + ", " if user_neg else "") + DEFAULT_NEGATIVE
    if style_key in {"photo","macro"}:
        neg = neg.replace("grain, ", "").replace("realistic shadow", "")
    if style_key == "painterly":
        neg += ", flat dull coloring, washed out colors"
    return neg

@app.post("/generate")
def generate(req: GenReq):
    t0 = time.time()
    try:
        raw_prompt = (req.prompt or "").strip()
        if not raw_prompt:
            return JSONResponse({"error": "Prompt vazio"}, status_code=400)

        if raw_prompt.lower() in {"teste", "flor"}:
            if TEST_IMAGE_PATH.exists():
                with open(TEST_IMAGE_PATH, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                return {"imagem_base64": b64, "from_test_image": True, "mode": "test"}
            return JSONResponse({"error": "Imagem de teste não encontrada"}, status_code=404)

        # 1) Roteamento para gráficos
        mode = (req.mode or "").lower().strip()
        chart = None

        # Funções via modo explícito do front
        if mode in {"chart_fn", "function"}:
            exprs, x_min, x_max = parse_function_from_prompt(raw_prompt)
            if not exprs:
                return JSONResponse({"error": "Não consegui identificar a função. Use y=..., f(x)=... ou palavras como linear/quadrática/exponencial."}, status_code=400)
            b64 = chart_function_png_b64(exprs, x_min, x_max, title="Função" if len(exprs)==1 else "Funções")
            return {"imagem_base64": b64, "type": "chart", "chart": "function", "exprs": exprs, "x_min": x_min, "x_max": x_max, "mode": "chart"}

        # Detecção automática (inclui funções)
        if mode != "ai":
            chart = detect_chart(raw_prompt)

        if chart:
            if chart.kind == "function":
                b64 = chart_function_png_b64(chart.params["exprs"], chart.params["x_min"], chart.params["x_max"], title="Função" if len(chart.params["exprs"])==1 else "Funções")
                return {"imagem_base64": b64, "type": "chart", "chart": "function", "params": chart.params, "mode": "chart"}
            if chart.kind == "mru":
                s0 = chart.params.get("s0", 0.0); v = chart.params.get("v", 2.0); t_max = chart.params.get("t_max", 10.0)
                b64 = chart_mru_png_b64(s0, v, t_max)
                return {"imagem_base64": b64, "type": "chart", "chart": "mru", "params": chart.params, "mode": "chart"}
            if chart.kind == "mruv":
                s0 = chart.params.get("s0", 0.0); v0 = chart.params.get("v0", 0.0); a = chart.params.get("a", 1.0); t_max = chart.params.get("t_max", 10.0)
                b64 = chart_mruv_png_b64(s0, v0, a, t_max)
                return {"imagem_base64": b64, "type": "chart", "chart": "mruv", "params": chart.params, "mode": "chart"}

        # 2) IA (difusão) — permanece igual
        style_key = (req.style or "infographic").lower()
        quality_key = (req.quality or "standard").lower()
        quality_cfg = QUALITY_PRESETS.get(quality_key, QUALITY_PRESETS["standard"])
        ultra = bool(req.ultra)
        ensemble = max(1, min(int(req.ensemble or 1), 6))

        if ultra:
            steps = int(req.steps or 140); guidance = float(req.guidance or 6.2)
            width = _snap(req.width or 1024); height = _snap(req.height or 1024)
            use_highres = True
        elif req.precise:
            steps = int(req.steps or 56); guidance = float(req.guidance or 5.3)
            width = _snap(req.width or 1024); height = _snap(req.height or 768)
            use_highres = True
        else:
            steps = int(req.steps or quality_cfg["steps"]); guidance = float(req.guidance or quality_cfg["guidance"])
            width = _snap(req.width or DEFAULT_WIDTH); height = _snap(req.height or DEFAULT_HEIGHT)
            use_highres = bool(req.highres)

        negative = build_negative(req.negative, style_key)
        full_prompt = build_full_prompt(raw_prompt, style_key, req.subject, bool(req.raw))
        base_w, base_h = (896, 896) if ultra else ((640, 640) if use_highres else (width, height))

        print(f"[GEN] ultra={ultra} ensemble={ensemble} raw={req.raw} simple={is_simple_object_prompt(raw_prompt)} style={style_key} steps={steps} guidance={guidance} size={width}x{height}")
        print(f"[PROMPT] {full_prompt}")

        images = []
        for i in range(ensemble):
            seed_final = (req.seed + i) if req.seed is not None else None
            generator = torch.Generator(device=pipe_base.device.type).manual_seed(int(seed_final)) if seed_final is not None else None
            t1 = time.time()
            out = pipe_base(
                prompt=full_prompt,
                negative_prompt=negative,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=base_w,
                height=base_h,
                generator=generator,
            )
            img = out.images[0]
            if use_highres and (width != base_w or height != base_h):
                img = img.resize((width, height), Image.LANCZOS)
            if pipe_refiner and req.refiner:
                try:
                    img = pipe_refiner(
                        prompt=full_prompt,
                        image=img,
                        negative_prompt=negative,
                        num_inference_steps=(18 if ultra else (12 if req.precise else 8)),
                        guidance_scale=max(3.0, guidance - 1.0),
                        generator=generator
                    ).images[0]
                except Exception as e:
                    print("[WARN] refiner falhou:", e)
            images.append((img, seed_final))
            print(f"[PASS {i+1}/{ensemble}] {(time.time()-t1):.1f}s seed={seed_final}")

        def _score(im: Image.Image):
            arr = np.array(im.resize((128, 128))).astype(np.float32)
            return float(arr.std())

        best_img, best_seed = max(images, key=lambda x: _score(x[0]))
        if best_img.mode != "RGB":
            best_img = best_img.convert("RGB")
        buf = io.BytesIO()
        best_img.save(buf, format="PNG", optimize=False)
        b64 = base64.b64encode(buf.getvalue()).decode()

        return {
            "imagem_base64": b64,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": guidance,
            "style": style_key,
            "quality": quality_key,
            "subject": req.subject,
            "seed": best_seed,
            "ensemble": ensemble,
            "ultra": ultra,
            "highres": use_highres,
            "refiner_used": bool(pipe_refiner and req.refiner),
            "mode": "ai",
        }
    except Exception as e:
        print("[ERRO] geração:", e, file=sys.stderr)
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5001, log_level="debug")