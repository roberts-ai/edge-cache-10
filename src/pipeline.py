import torch
from PIL.Image import Image
from diffusers import StableDiffusionXLPipeline
from pipelines.models import TextToImageRequest
from torch import Generator
from sfast.compilers.diffusion_pipeline_compiler import (compile,
                                                         CompilationConfig)

import packaging.version
import torch

if packaging.version.parse(torch.__version__) >= packaging.version.parse('1.12.0'):
    torch.backends.cuda.matmul.allow_tf32 = True

INF_STEP= 18
def load_pipeline() -> StableDiffusionXLPipeline:
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "./models/newdream-sdxl-21/",
        torch_dtype=torch.float16, 
        use_safetensors=True, 
        local_files_only=True,
    )
    pipeline.to("cuda")
    config = CompilationConfig.Default()

    try:
        import triton
        config.enable_triton = True
    except ImportError:
        print('Triton not installed, skip')

    config.enable_cuda_graph = True
    
    
    pipeline  = compile(pipeline, config)
    
    kwarg_inputs = dict(
        prompt=
        '(masterpiece:1,2), best quality, masterpiece, best detailed face, a beautiful girl',
        height=512,
        width=512,
        num_inference_steps=INF_STEP,
        num_images_per_prompt=1,
    )

    for _ in range(3):
        output_image = pipeline(**kwarg_inputs).images[0]
    pipeline(prompt="")
    return pipeline

def infer(request: TextToImageRequest, pipeline: StableDiffusionXLPipeline) -> Image:
    generator = Generator(pipeline.device).manual_seed(request.seed) if request.seed else None
    return pipeline(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        generator=generator,
        num_inference_steps=INF_STEP,
        num_images_per_prompt=1,
    ).images[0] 
    return pipeline.tgate(
        prompt=request.prompt,
        negative_prompt=f"{request.negative_prompt}",
        width=request.width,
        height=request.height,
        generator=generator,
        num_inference_steps=INF_STEP,
        gate_step=GATE_STEP,
    ).images[0]

