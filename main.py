import json
import os.path
import tempfile
import uuid
import traceback
from typing import List, Optional
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, staticfiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from transformers import WhisperFeatureExtractor, AutoTokenizer
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token
from flow_inference import AudioDecoder
import requests
from pydantic import BaseModel
import sys
import logging
import ssl

sys.path.insert(0, "./cosyvoice")
sys.path.insert(0, "./third_party/Matcha-TTS")

app = FastAPI(title="GLM-4-Voice API")

# Serve static files (frontend)
app.mount("/static", staticfiles.StaticFiles(directory="frontend"), name="static")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Remaining connections: {len(self.active_connections)}")

manager = ConnectionManager()

class GlobalConfig:
    def __init__(self):
        self.flow_path = "./glm-4-voice-decoder"
        self.model_path = "THUDM/glm-4-voice-9b"
        self.tokenizer_path = "THUDM/glm-4-voice-tokenizer"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.flow_config = os.path.join(self.flow_path, "config.yaml")
        self.flow_checkpoint = os.path.join(self.flow_path, 'flow.pt')
        self.hift_checkpoint = os.path.join(self.flow_path, 'hift.pt')
        
        self.glm_tokenizer = None
        self.audio_decoder = None
        self.whisper_model = None
        self.feature_extractor = None
        self.system_prompt = ("User will provide you with a {type} instruction. "
                            "Do it step by step. First, think about the instruction "
                            "and respond in a interleaved manner, with 13 text token "
                            "followed by 26 audio tokens.")

config = GlobalConfig()

class GenerationParams(BaseModel):
    temperature: float = 0.2
    top_p: float = 0.8
    max_new_tokens: int = 2000

def initialize_models():
    if config.audio_decoder is not None:
        return

    # Initialize models
    config.glm_tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, 
        trust_remote_code=True
    )

    config.audio_decoder = AudioDecoder(
        config_path=config.flow_config,
        flow_ckpt_path=config.flow_checkpoint,
        hift_ckpt_path=config.hift_checkpoint,
        device=config.device
    )

    config.whisper_model = WhisperVQEncoder.from_pretrained(
        config.tokenizer_path
    ).eval().to(config.device)
    
    config.feature_extractor = WhisperFeatureExtractor.from_pretrained(
        config.tokenizer_path
    )

@app.on_event("startup")
async def startup_event():
    initialize_models()

async def process_audio_input(audio_data: bytes, history: list) -> tuple[str, str]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_data)
        audio_path = f.name

    history.append({"role": "user", "content": {"path": audio_path}})
    
    audio_tokens = extract_speech_token(
        config.whisper_model, 
        config.feature_extractor, 
        [audio_path]
    )[0]
    
    if len(audio_tokens) == 0:
        raise ValueError("No audio tokens extracted")
        
    audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
    audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
    
    return audio_tokens, config.system_prompt.format(type="speech")

async def process_text_input(text: str, history: list) -> tuple[str, str]:
    history.append({"role": "user", "content": text})
    return text, config.system_prompt.format(type="text")

async def process_audio_chunk(
    websocket: WebSocket,
    audio_tokens: list,
    config: GlobalConfig,
    uuid_str: str,
    flow_prompt_speech_token: torch.Tensor,
    prompt_speech_feat: torch.Tensor,
    tts_speechs: list,
    tts_mels: list,
    prev_mel: Optional[torch.Tensor],
    is_finalize: bool
) -> tuple[torch.Tensor, list, list]:
    """Process a chunk of audio tokens and send to the client"""
    block_size = 20 if len(audio_tokens) >= 10 else 10
    tts_token = torch.tensor(audio_tokens, device=config.device).unsqueeze(0)

    if prev_mel is not None:
        prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)

    tts_speech, tts_mel = config.audio_decoder.token2wav(
        tts_token,
        uuid=uuid_str,
        prompt_token=flow_prompt_speech_token.to(config.device),
        prompt_feat=prompt_speech_feat.to(config.device),
        finalize=is_finalize
    )

    # Add to collections
    tts_speechs.append(tts_speech.squeeze())
    tts_mels.append(tts_mel)
    
    # Convert audio to bytes and send
    audio_data = tts_speech.squeeze().cpu().numpy()
    await websocket.send_json({
        "type": "audio_chunk",
        "sample_rate": 22050,
        "data": audio_data.tolist()
    })
    
    # Update prompt token
    flow_prompt_speech_token = torch.cat(
        (flow_prompt_speech_token, tts_token), 
        dim=-1
    )
    
    return flow_prompt_speech_token, tts_speechs, tts_mels

async def finalize_response(
    websocket: WebSocket,
    tts_speechs: list,
    complete_tokens: list,
    text_tokens: list,
    history: list,
    config: GlobalConfig
):
    """Finalize the response and send completion message"""
    # Combine all audio chunks
    tts_speech = torch.cat(tts_speechs, dim=-1).cpu()
    
    # Decode complete text
    complete_text = config.glm_tokenizer.decode(
        complete_tokens, 
        spaces_between_special_tokens=False
    )
    
    # Save final audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        torchaudio.save(f.name, tts_speech.unsqueeze(0), 22050, format="wav")
        
    # Update history
    history.append({
        "role": "assistant", 
        "content": {
            "path": f.name, 
            "type": "audio/wav"
        }
    })
    history.append({
        "role": "assistant", 
        "content": config.glm_tokenizer.decode(
            text_tokens, 
            ignore_special_tokens=False
        )
    })
    
    # Send final response
    await websocket.send_json({
        "type": "complete",
        "text": complete_text,
        "history": history
    })

async def process_model_response(
    websocket: WebSocket,
    inputs: str,
    params: GenerationParams,
    history: list,
    config: GlobalConfig
):
    """Process the model response and handle audio generation"""
    with torch.no_grad():
        response = requests.post(
            "http://64.247.196.76:10000/generate_stream",
            data=json.dumps({
                "prompt": inputs,
                "temperature": params.temperature,
                "top_p": params.top_p,
                "max_new_tokens": params.max_new_tokens,
            }),
            stream=True
        )
        
        text_tokens, audio_tokens = [], []
        audio_offset = config.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
        end_token_id = config.glm_tokenizer.convert_tokens_to_ids('<|user|>')
        complete_tokens = []
        
        # Initialize audio processing variables
        prompt_speech_feat = torch.zeros(1, 0, 80).to(config.device)
        flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(config.device)
        this_uuid = str(uuid.uuid4())
        tts_speechs = []
        tts_mels = []
        prev_mel = None
        is_finalize = False
        
        for chunk in response.iter_lines():
            if not chunk:
                continue
                
            try:
                token_id = json.loads(chunk)["token_id"]
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode chunk: {chunk}")
                continue
            
            if token_id == end_token_id:
                is_finalize = True
                
            if len(audio_tokens) >= 10 or (is_finalize and audio_tokens):
                flow_prompt_speech_token, tts_speechs, tts_mels = await process_audio_chunk(
                    websocket, 
                    audio_tokens, 
                    config, 
                    this_uuid,
                    flow_prompt_speech_token,
                    prompt_speech_feat,
                    tts_speechs,
                    tts_mels,
                    prev_mel,
                    is_finalize
                )
                prev_mel = tts_mels[-1] if tts_mels else None
                audio_tokens = []
                
            if not is_finalize:
                complete_tokens.append(token_id)
                if token_id >= audio_offset:
                    audio_tokens.append(token_id - audio_offset)
                else:
                    text_tokens.append(token_id)

        # Final processing
        if tts_speechs:
            await finalize_response(
                websocket,
                tts_speechs,
                complete_tokens,
                text_tokens,
                history,
                config
            )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("New WebSocket connection attempt")
    await manager.connect(websocket)
    
    try:
        # Send initial connection success message
        await websocket.send_json({
            "type": "connection_status",
            "status": "connected"
        })
        
        history = []
        previous_input_tokens = ""
        previous_completion_tokens = ""
        
        while True:
            try:
                # Receive message from client
                ws_message = await websocket.receive()
                logger.info(f"Received WebSocket message type: {ws_message.get('type')}")
                
                # Handle different WebSocket message types
                if ws_message["type"] == "websocket.disconnect":
                    logger.info("Client disconnected normally")
                    manager.disconnect(websocket)
                    break
                    
                elif ws_message["type"] == "websocket.receive":
                    # Parse the actual message content
                    try:
                        if "text" in ws_message:
                            message = json.loads(ws_message["text"])
                        elif "bytes" in ws_message:
                            message = {"type": "audio", "data": ws_message["bytes"]}
                        else:
                            raise ValueError("Invalid message format")
                            
                        logger.info(f"Parsed message type: {message.get('type', 'unknown')}")

                        # Handle ping messages
                        if message.get("type") == "ping":
                            await websocket.send_json({"type": "pong"})
                            continue
                        
                        # Process the message based on its type
                        params = GenerationParams(**(message.get("params", {}) or {}))
                        
                        if message.get("type") == "audio":
                            # For audio type, we expect the audio data in the current message
                            audio_data = message.get("data") or await websocket.receive_bytes()
                            user_input, system_prompt = await process_audio_input(audio_data, history)
                        elif message.get("type") == "text":
                            user_input, system_prompt = await process_text_input(
                                message.get("text", ""), 
                                history
                            )
                        else:
                            logger.warning(f"Unhandled message type: {message.get('type')}")
                            continue

                        # Prepare input for model
                        inputs = previous_input_tokens + previous_completion_tokens
                        inputs = inputs.strip()
                        if "<|system|>" not in inputs:
                            inputs += f"<|system|>\n{system_prompt}"
                        inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"

                        # Generate and process response
                        await process_model_response(
                            websocket, 
                            inputs, 
                            params, 
                            history,
                            config
                        )
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse message: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid message format"
                        })
                        
                else:
                    logger.warning(f"Unhandled WebSocket message type: {ws_message['type']}")

            except WebSocketDisconnect:
                logger.info("Client disconnected normally")
                manager.disconnect(websocket)
                break
            
            except Exception as e:
                logger.error(f"Error in WebSocket connection: {str(e)}")
                logger.error(traceback.format_exc())
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
                
    except Exception as e:
        logger.error(f"Failed to establish WebSocket connection: {str(e)}")
        logger.error(traceback.format_exc())
        manager.disconnect(websocket)
        try:
            await websocket.close(code=1011, reason=str(e))
        except:
            pass

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    return FileResponse("frontend/index.html")

@app.get("/voice-chat.js")
async def read_js():
    """Serve the JavaScript file"""
    return FileResponse("frontend/voice-chat.js")

if __name__ == "__main__":
    import uvicorn
    # Generate self-signed certificate for development
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(
        certfile="ssl/certificate.crt",  # Path to certificate
        keyfile="ssl/private.key"    # Path to private key
    )
    
    # Run server with SSL
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8443,
        ssl_keyfile="ssl/private.key",
        ssl_certfile="ssl/certificate.crt"
    )