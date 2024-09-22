import torch
from PIL.Image import Image
from diffusers import StableDiffusionXLPipeline,DPMSolverMultistepScheduler
from pipelines.models import TextToImageRequest
from torch import Generator
from tgate import TgateSDXLDeepCacheLoader

GATE_STEP = 10
INF_STEP= 30

def load_pipeline() -> StableDiffusionXLPipeline:
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "./models/newdream-sdxl-21/",
        torch_dtype=torch.float16, 
        use_safetensors=True, 
        local_files_only=True,
    )

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    pipeline = TgateSDXLDeepCacheLoader(
                   pipeline,
                          cache_interval=3,
                                 cache_branch_id=0,
                                 ).to("cuda")
    pipeline(prompt="")
    return pipeline

def infer(request: TextToImageRequest, pipeline: StableDiffusionXLPipeline) -> Image:
    generator = Generator(pipeline.device).manual_seed(request.seed) if request.seed else None
    return pipeline.tgate(
        prompt=request.prompt,
        negative_prompt=f"{request.negative_prompt}",
        width=request.width,
        height=request.height,
        generator=generator,
        num_inference_steps=INF_STEP,
        gate_step=GATE_STEP,
    ).images[0]

