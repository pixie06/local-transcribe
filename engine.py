"""Local VibeVoice-ASR, NF4, serial chunks with recoverable results."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / 'cache'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
import sys, json, subprocess, datetime, time
sys.path.insert(0, str(ROOT / 'vendor' / 'VibeVoice'))

def decode_audio(path, start=0, duration=None):
    import imageio_ffmpeg, numpy as np
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-ss', str(start), '-i', str(path)]
    if duration: cmd += ['-t', str(duration)]
    cmd += ['-f', 'f32le', '-ac', '1', '-ar', '24000', 'pipe:1']
    result = subprocess.run(cmd, capture_output=True, check=True)
    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
    # AAC decoding may return a few padded samples past -t; do not let
    # those advance the next chunk's source offset.
    return audio[:round(duration * 24000)] if duration else audio

def stamp(t, subtitle=False):
    ms = max(0, round(float(t)*1000))
    h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f'{h:02}:{m:02}:{s:02}' + (f',{ms:03}' if subtitle else '')

class Engine:
    def __init__(self):
        import torch
        from transformers import BitsAndBytesConfig
        from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
        from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
        if not torch.cuda.is_available(): raise RuntimeError('NVIDIA GPU unavailable; check driver.')
        torch.set_num_threads(4)
        self.torch = torch
        model_dir = ROOT / 'models' / 'VibeVoice-ASR'
        self.processor = VibeVoiceASRProcessor.from_pretrained(str(model_dir),
            language_model_pretrained_name=str(ROOT/'models'/'QwenTokenizer'), local_files_only=True)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=['acoustic_tokenizer','semantic_tokenizer','acoustic_connector','semantic_connector'])
        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            str(model_dir), quantization_config=quant, device_map={'':0},
            torch_dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).eval()
        # 2 seconds = 48000 samples, exactly 15 encoder strides (3200).
        # Non-aligned chunks lose latent frames at streaming boundaries.
        original_encode = self.model.encode_speech
        def encode_low_memory(*args, **kwargs):
            kwargs['streaming_segment_duration'] = 2.0
            result = original_encode(*args, **kwargs)
            torch.cuda.empty_cache()
            return result
        self.model.encode_speech = encode_low_memory
        # Generation consumes only the last-token logits. Avoid materializing
        # full prompt x vocabulary logits and a dequantized full output matrix.
        self.model.lm_head.register_forward_pre_hook(lambda module, args: (args[0][:, -1:, :],))
        # On 8 GB WDDM GPUs, retained prefill/conv allocations cause severe
        # shared-memory paging. Release unused allocations before each decode.
        self.model.register_forward_hook(lambda module, args, output: torch.cuda.empty_cache())
        torch.cuda.empty_cache()
        print(f'Model ready. Allocated GPU: {torch.cuda.memory_allocated()/2**30:.2f} GiB', flush=True)

    def transcribe(self, audio, context=''):
        t = self.torch
        inputs = self.processor(audio=audio, sampling_rate=24000, return_tensors='pt',
                                context_info=context or None)
        inputs = {k:v.to('cuda') if isinstance(v,t.Tensor) else v for k,v in inputs.items()}
        max_tokens = max(512, min(8192, int(len(audio)/24000*12+256)))
        with t.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                pad_token_id=self.processor.pad_id, eos_token_id=self.processor.tokenizer.eos_token_id)
        new_ids = ids[0, inputs['input_ids'].shape[1]:]
        raw = self.processor.decode(new_ids, skip_special_tokens=True)
        segments = self.processor.post_process_transcription(raw)
        if len(new_ids) >= max_tokens: raise RuntimeError('Output token limit reached; use shorter chunks.')
        del inputs, ids, new_ids
        t.cuda.empty_cache()
        if not segments: raise RuntimeError('Unparseable model output: '+raw[:1000])
        return segments, raw

def run(path, context='', chunk_seconds=30, start=0, limit=0):
    path = Path(path).expanduser().resolve()
    if not path.is_file(): raise FileNotFoundError(str(path))
    job = ROOT / 'results' / (datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '-' + path.stem)
    job.mkdir(parents=True)
    meta = dict(source=str(path), model='microsoft/VibeVoice-ASR', quantization='NF4',
                audio_encoder_dtype='bfloat16', audio_encoder_chunk_seconds=2,
                chunk_seconds=chunk_seconds, start=start, limit=limit, context=context,
                speaker_scope='Each chunk only. IDs do not identify the same person across chunks.', status='running')
    def save(): (job/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    save()
    yield f'正在加载本地模型。结果目录：{job}', str(job)
    try:
        engine = Engine()
        all_segments=[]; offset=float(start); chunk_id=0
        while True:
            duration=min(chunk_seconds, start+limit-offset) if limit else chunk_seconds
            if duration<=0: break
            audio=decode_audio(path, offset, duration)
            if len(audio)<2400: break
            chunk_id+=1
            yield f'正在转写第 {chunk_id} 段，起点 {stamp(offset)}', str(job)
            before=time.monotonic()
            segments, raw=engine.transcribe(audio, context)
            for seg in segments:
                seg['start_time']=float(seg.get('start_time',0))+offset
                seg['end_time']=float(seg.get('end_time',len(audio)/24000))+offset
                seg['chunk_id']=chunk_id
            (job/f'chunk-{chunk_id:04}.json').write_text(json.dumps(dict(offset=offset,raw=raw,segments=segments,seconds=time.monotonic()-before),ensure_ascii=False,indent=2),encoding='utf-8')
            all_segments.extend(segments)
            (job/'transcript.json').write_text(json.dumps(all_segments,ensure_ascii=False,indent=2),encoding='utf-8')
            lines=[f"[{stamp(s['start_time'])}–{stamp(s['end_time'])}] 段{s['chunk_id']}/Speaker {s.get('speaker_id','?')}: {s.get('text','')}" for s in all_segments]
            (job/'transcript.txt').write_text('\n'.join(lines),encoding='utf-8')
            srt=[f"{i}\n{stamp(s['start_time'],True)} --> {stamp(s['end_time'],True)}\n[Chunk {s['chunk_id']}, Speaker {s.get('speaker_id','?')}] {s.get('text','')}\n" for i,s in enumerate(all_segments,1)]
            (job/'transcript.srt').write_text('\n'.join(srt),encoding='utf-8')
            meta['processed_seconds']=offset+len(audio)/24000-start; save()
            yield '\n'.join(lines[-15:]), str(job)
            actual=len(audio)/24000
            offset+=actual
            if actual<duration-0.1: break
        meta['status']='complete'; meta['processed_seconds']=offset-start; save()
        yield f'完成。已处理 {stamp(offset-start)}，结果保存在：{job}', str(job)
    except Exception as exc:
        meta['status']='failed';meta['error']=str(exc);save()
        raise
    finally:
        if 'engine' in locals():
            del engine
            import gc, torch
            gc.collect(); torch.cuda.empty_cache()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('audio');p.add_argument('--start',type=float,default=0)
    p.add_argument('--limit',type=float,default=0);p.add_argument('--chunk',type=int,default=30)
    p.add_argument('--context',default='')
    a=p.parse_args()
    for message,folder in run(a.audio,a.context,a.chunk,a.start,a.limit): print(message,flush=True)
