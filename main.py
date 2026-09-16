from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import numpy as np

# Import your modular backend processor
from pipeline.processor import process_nail_image

app = FastAPI()

# Configure CORS so your Three.js frontend can communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    # Read image into memory and convert to numpy array for the pipeline
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    image_np = np.array(image)
    
    # Run your modular computer vision pipeline
    results = process_nail_image(image_np)
    
    return {"success": True, "data": results}