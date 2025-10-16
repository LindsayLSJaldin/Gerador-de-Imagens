from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image
from pathlib import Path
import torch, io, base64, os, sys, uvicorn, traceback, time

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000", "http://127.0.0.1:5000"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Modo offline e sem offloading automático
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["ACCELERATE_DISABLE_WEIGHTS_CACHE"] = "1"

# Caminho do modelo (Diffusers SDXL base)
BASE_DIR_DEFAULT = r"C:\Users\Linds\Documentos\Arquivos Mauá\Programação\gerador_imagens\models\sdxl\stable-diffusion-xl-base-1.0"
MODEL_BASE_PATH = Path(os.environ.get("MODEL_DIR", BASE_DIR_DEFAULT)).expanduser().resolve()
TEST_IMAGE_PATH = Path(os.environ.get("TEST_IMAGE_PATH", Path(__file__).parent / "static" / "teste.png"))

# Defaults (512x512 / 30 steps)
DEVICE = os.environ.get("DEVICE", "cpu")  # "cpu" ou "cuda"
DEFAULT_WIDTH = int(os.environ.get("IMG_WIDTH", "512"))
DEFAULT_HEIGHT = int(os.environ.get("IMG_HEIGHT", "512"))
DEFAULT_STEPS = int(os.environ.get("STEPS", "30"))
DEFAULT_GUIDANCE = float(os.environ.get("GUIDANCE", "7.0"))

def _assert_diffusers_folder(path: Path):
    required = ["model_index.json", "scheduler", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "unet", "vae"]
    missing = [p for p in required if not (path / p).exists()]
    if missing:
        raise RuntimeError(f"Estrutura do modelo incompleta em {path}. Faltam: {', '.join(missing)}")

def _dtype():
    return torch.float16 if (DEVICE == "cuda" and torch.cuda.is_available()) else torch.float32

def load_base(model_path: Path):
    print(f"Carregando SDXL BASE: {model_path}")
    if model_path.is_dir():
        _assert_diffusers_folder(model_path)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            str(model_path),
            torch_dtype=_dtype(),
            local_files_only=True,
            low_cpu_mem_usage=False,
            device_map=None,
        )
    elif model_path.is_file() and model_path.suffix.lower() in {".safetensors", ".ckpt"}:
        pipe = StableDiffusionXLPipeline.from_single_file(
            str(model_path),
            torch_dtype=_dtype(),
            local_files_only=True,
            low_cpu_mem_usage=False,
            device_map=None,
        )
    else:
        raise RuntimeError(f"Caminho inválido para BASE: {model_path}")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda" if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu")
    pipe.enable_attention_slicing()
    return pipe

try:
    pipe_base = load_base(MODEL_BASE_PATH)
except Exception as e:
    print("[ERRO] Falha ao carregar modelo base:", e, file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

print(f"Modelo base=OK | device={DEVICE}")

class GenReq(BaseModel):
    prompt: str
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance: float | None = None

@app.get("/health")
def health():
    return {"ok": True, "base": str(MODEL_BASE_PATH), "device": DEVICE}

def _snap_to_multiple(v: int, mult: int = 8, minimum: int = 64, maximum: int = 1536) -> int:
    v = max(minimum, min(maximum, v))
    return (v // mult) * mult

@app.post("/generate")
def generate(req: GenReq):
    t0 = time.time()
    try:
        prompt = (req.prompt or "").strip()
        if not prompt:
            return JSONResponse({"error": "Prompt vazio"}, status_code=400)

        # >>> BLOCO ESPECIAL PARA PROMPT 'flor'
        if prompt.lower() == "flor":
            if TEST_IMAGE_PATH.exists():
                with open(TEST_IMAGE_PATH, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                print(f"[GEN] retorno imagem teste ({TEST_IMAGE_PATH}) len={len(b64)}")
                return {"imagem_base64": b64, "from_test_image": True}
            else:
                return JSONResponse({"error": f"Imagem de teste não encontrada em {TEST_IMAGE_PATH}"}, status_code=404)

        width = _snap_to_multiple(req.width or DEFAULT_WIDTH, 8)
        height = _snap_to_multiple(req.height or DEFAULT_HEIGHT, 8)
        steps = int(req.steps or DEFAULT_STEPS)
        guidance = float(req.guidance or DEFAULT_GUIDANCE)

        negative = "low quality, blurry, distorted, watermark, text, logo, artifacts"

        print(f"[GEN] prompt='{prompt[:60]}...' w={width} h={height} steps={steps} guidance={guidance}")
        t1 = time.time()
        out = pipe_base(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
        )
        image: Image.Image = out.images[0]
        t2 = time.time()
        print(f"[GEN] difusão: {(t2 - t1):.1f}s")

        if image.mode != "RGB":
            image = image.convert("RGB")

        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=False)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        t3 = time.time()
        print(f"[GEN] encode: {(t3 - t2):.1f}s | total: {(t3 - t0):.1f}s | b64 len: {len(b64)}")

        return {"imagem_base64": b64}
    except Exception as e:
        print("[ERRO] Geração falhou:", e, file=sys.stderr)
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5001, log_level="debug")