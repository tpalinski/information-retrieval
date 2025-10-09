import os
import zmq
import msgpack
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

SOCKET_ADDRESS = 'ipc:///tmp/imgsearch.ipc'

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", force_download=False)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", force_download=False)
model.eval()

def generateTextEmbedding(query: str) -> torch.Tensor:
  print(f'Generating embedding for: {query}')
  inputs = processor(text=[query], return_tensors="pt", padding=True)
  with torch.no_grad():
    embedding = model.get_text_features(**inputs).cpu()
  return embedding.squeeze().numpy()

def generateImageEmbedding(path: str) -> torch.Tensor:
  print(f'Generating embedding for: {path}')
  if not os.path.exists(path):
    raise FileNotFoundError("Could not locate image")
  image = Image.open(path).convert("RGB")
  inputs = processor(images=[image], return_tensors="pt")
  with torch.no_grad():
    embedding = model.get_image_features(**inputs).cpu()
  return embedding.squeeze().numpy()

ctx = zmq.Context()
sock = ctx.socket(zmq.REP)
sock.bind(SOCKET_ADDRESS)
print("Starting listening")
while True:
  msg = sock.recv()
  req = msgpack.unpackb(msg, raw=False)
  kind = req["type"]

  try:
    if kind == "text":
      text = req["payload"]
      embedding = generateTextEmbedding(text)

    elif kind == "image":
      image_path = req["payload"]
      embedding = generateImageEmbedding(image_path)

    else:
        raise ValueError(f"Unknown type: {kind}")

    # ---- Serialize embedding ----
    payload = {
        "shape": embedding.shape,
        "dtype": str(embedding.dtype),
        "data": embedding.tobytes(),
    }

  except Exception as e:
        print("Error:", e)
        payload = {"error": str(e)}

  sock.send(msgpack.packb(payload, use_bin_type=True))
