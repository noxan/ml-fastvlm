from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from fastapi import FastAPI, File, Form, UploadFile

from predict import predict as model_predict

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@dataclass
class Args:
    model_path: str = "./llava-v1.5-13b"
    model_base: str | None = None
    image_file: BinaryIO | None = None
    prompt: str = "Describe the image."
    conv_mode: str = "qwen_2"
    temperature: float = 0.2
    top_p: float | None = None
    num_beams: int = 1


@app.post("/predict")
async def predict(image: UploadFile = File(...), prompt: str = Form(...)):
    model_path = Path(__file__).parent / "checkpoints" / "llava-fastvithd_0.5b_stage3"

    output = model_predict(
        args=Args(
            model_path=model_path.as_posix(),
            image_file=image.file,
            prompt=prompt,
        )
    )

    return {"result": output}
