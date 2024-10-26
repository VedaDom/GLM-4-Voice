import json
import os.path
import tempfile
import uuid
import torch
import torchaudio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from transformers import WhisperFeatureExtractor, AutoTokenizer
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token
from flow_inference import AudioDecoder
import requests
from typing import Optional
from pydantic import BaseModel

app = FastAPI(title="GLM-4-Voice API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and configurations
class GlobalConfig:
    flow_path = "./glm-4-voice-decoder"
    model_path = "THUDM/glm-4-voice-9b"
    tokenizer_path = "THUDM/glm-4-voice-tokenizer"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    flow_config = os.path.join(flow_path, "config.yaml")
    flow_checkpoint = os.path.join(flow_path, 'flow.pt')
    hift_checkpoint = os.path.join(flow_path, 'hift.pt')
    
    glm_tokenizer = None
    audio_decoder = None
    whisper_model = None
    feature_extractor = None

config = GlobalConfig()

class GenerationParams(BaseModel):
    temperature: float = 0.2
    top_p: float = 0.8
    max_new_tokens: int = 2000

def initialize_models():
    if config.audio_decoder is not None:
        return

    # GLM
    config.glm_tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, 
        trust_remote_code=True
    )

    # Flow & Hift
    config.audio_decoder = AudioDecoder(
        config_path=config.flow_config,
        flow_ckpt_path=config.flow_checkpoint,
        hift_ckpt_path=config.hift_checkpoint,
        device=config.device
    )

    # Speech tokenizer
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
    # Save audio data to temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_data)
        audio_path = f.name

    history.append({"role": "user", "content": {"path": audio_path}})
    
    # Extract speech tokens
    audio_tokens = extract_speech_token(
        config.whisper_model, 
        config.feature_extractor, 
        [audio_path]
    )[0]
    
    if len(audio_tokens) == 0:
        raise ValueError("No audio tokens extracted")
        
    audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
    audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
    
    system_prompt = ("User will provide you with a speech instruction. "
                    "Do it step by step. First, think about the instruction "
                    "and respond in a interleaved manner, with 13 text token "
                    "followed by 26 audio tokens.")
    
    return audio_tokens, system_prompt

async def process_text_input(text: str, history: list) -> tuple[str, str]:
    history.append({"role": "user", "content": text})
    
    system_prompt = ("User will provide you with a text instruction. "
                    "Do it step by step. First, think about the instruction "
                    "and respond in a interleaved manner, with 13 text token "
                    "followed by 26 audio tokens.")
    
    return text, system_prompt

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    history = []
    previous_input_tokens = ""
    previous_completion_tokens = ""
    
    try:
        while True:
            # Receive message from client
            message = await websocket.receive_json()
            
            input_type = message.get("type", "text")
            params = GenerationParams(**message.get("params", {}))
            
            try:
                if input_type == "audio":
                    # Receive audio data in next message
                    audio_data = await websocket.receive_bytes()
                    user_input, system_prompt = await process_audio_input(audio_data, history)
                else:
                    user_input, system_prompt = await process_text_input(
                        message["text"], 
                        history
                    )

                # Prepare input for model
                inputs = previous_input_tokens + previous_completion_tokens
                inputs = inputs.strip()
                if "<|system|>" not in inputs:
                    inputs += f"<|system|>\n{system_prompt}"
                inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"

                # Generate response
                with torch.no_grad():
                    response = requests.post(
                        "http://localhost:10000/generate_stream",
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
                    block_size = 10
                    
                    # Process response stream
                    for chunk in response.iter_lines():
                        token_id = json.loads(chunk)["token_id"]
                        
                        if token_id == end_token_id:
                            is_finalize = True
                            
                        if len(audio_tokens) >= block_size or (is_finalize and audio_tokens):
                            block_size = 20
                            tts_token = torch.tensor(audio_tokens, device=config.device).unsqueeze(0)

                            if prev_mel is not None:
                                prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)

                            tts_speech, tts_mel = config.audio_decoder.token2wav(
                                tts_token,
                                uuid=this_uuid,
                                prompt_token=flow_prompt_speech_token.to(config.device),
                                prompt_feat=prompt_speech_feat.to(config.device),
                                finalize=is_finalize
                            )
                            prev_mel = tts_mel

                            # Send audio chunk to client
                            tts_speechs.append(tts_speech.squeeze())
                            tts_mels.append(tts_mel)
                            
                            # Convert audio to bytes and send
                            audio_data = tts_speech.squeeze().cpu().numpy()
                            await websocket.send_json({
                                "type": "audio_chunk",
                                "sample_rate": 22050,
                                "data": audio_data.tolist()
                            })
                            
                            flow_prompt_speech_token = torch.cat(
                                (flow_prompt_speech_token, tts_token), 
                                dim=-1
                            )
                            audio_tokens = []
                            
                        if not is_finalize:
                            complete_tokens.append(token_id)
                            if token_id >= audio_offset:
                                audio_tokens.append(token_id - audio_offset)
                            else:
                                text_tokens.append(token_id)

                    # Final processing
                    tts_speech = torch.cat(tts_speechs, dim=-1).cpu()
                    complete_text = config.glm_tokenizer.decode(
                        complete_tokens, 
                        spaces_between_special_tokens=False
                    )
                    
                    # Save final audio
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        torchaudio.save(f, tts_speech.unsqueeze(0), 22050, format="wav")
                        
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

            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
                
    except Exception as e:
        print(f"WebSocket error: {e}")