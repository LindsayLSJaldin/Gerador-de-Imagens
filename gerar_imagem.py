from diffusers import StableDiffusionXLPipeline
import os

# Caminho local pro modelo
modelo_local = r"C:\Users\Linds\Documentos\Arquivos Mauá\Programação\gerador_imagens\stable-diffusion-xl-base-1.0"

# Cria pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    modelo_local,
    torch_dtype=None  # evita aviso de dtype
)

# Detecta e usa DirectML se disponível, senão CPU
use_device = "cpu"
try:
    import torch_directml
    import torch
    dml_device = torch_directml.device()
    pipe.to(dml_device)
    use_device = "DirectML"
except ImportError:
    print("DirectML não disponível, usando CPU")
    pipe.to("cpu")
except RuntimeError:
    print("Erro ao usar DirectML, usando CPU")
    pipe.to("cpu")

# Habilita offload para economizar memória
try:
    pipe.enable_model_cpu_offload()
except RuntimeError:
    print("Accelerate não configurado ou offload não suportado, continuando sem offload")

print(f"Pipeline carregado usando: {use_device}")

# Função para gerar e salvar imagem
def gerar_imagem(prompt: str, steps: int = 30, scale: float = 7.5, nome_arquivo: str = "saida.png"):
    imagem = pipe(prompt, num_inference_steps=steps, guidance_scale=scale).images[0]
    imagem.save(nome_arquivo)
    print(f"Imagem gerada e salva em {os.path.abspath(nome_arquivo)}")

# Exemplo de uso
if __name__ == "__main__":
    prompt = "Um gato astronauta no espaço, estilo pintura digital"
    gerar_imagem(prompt)
