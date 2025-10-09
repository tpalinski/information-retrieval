# Semantic Image Search (SIS)
A semantic-simmilarity based image search engine using DIY FAiSS inspired indexing
Project for Information Retrieval course for University of Pavia 2025/2026 - computer engineering

## Running the app
The app consists of three elements:
- an indexing search engine backend - cpp code
- inference provider - python application (see `inference_server.py`)
- a user gui in the form of a react SPA (see `frontend` directory)

### Local deployment
To run the entire app, all three parts of the application must be running at the same time (unless you want to run it headless, then ignore the frontend part)

#### Inference server
This python application leverages `torch` and `sentence-transformers` packages. For interprocess communication, `zmq` and `msgpack` are also required. To run the server, simply run `python inference_server.py`.

#### Indexing engine
First of all, make sure you have cmake and libtorch installed on your system. To run the service, put your images dataset in `data/img` directory (only .jpg images are supported). After building the app, simply run the application at `build/main`.

#### Frontend (optional)
After navigating to `frontend` directory, run `npm i` and `npm run dev`. The gui should be available at `http://127.0.0.1:5173`.

### Docker
Due to my lack of experience with CPP and cmake in particular, there is unfortunately no Docker configuration to run for this app

