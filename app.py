from engine import ROOT, run
import os
os.environ['GRADIO_TEMP_DIR']=str(ROOT/'cache'/'uploads')
import gradio as gr

def transcribe(path,upload,context,chunk,preview,start):
    source=path.strip().strip('"') if path.strip() else upload
    if not source: raise gr.Error('请选择音频或填写本地路径。')
    try:
        for status,folder in run(source,context,int(chunk),start=max(0,float(start)),limit=60 if preview else 0):
            yield status,folder
    except Exception as e:
        raise gr.Error(str(e))

with gr.Blocks(title='LocalTranscribe - VibeVoice-ASR') as app:
    gr.Markdown('# 本地语音转文字 · VibeVoice-ASR\n录音在本机处理。建议先试转 1 分钟，再转全文。')
    path=gr.Textbox(label='本地音频完整路径（优先使用，避免复制大文件）')
    upload=gr.File(label='或选择音频文件',type='filepath',file_types=['audio'])
    context=gr.Textbox(label='录音语言、背景与专业术语（可选）',placeholder='例如：German university office hours. Fachbegriffe: ...')
    with gr.Row():
        chunk=gr.Dropdown([20,30,60],value=30,label='每段秒数（默认 30；显存不足可用 20）')
        preview=gr.Checkbox(value=True,label='仅试转 1 分钟')
    start=gr.Number(value=0,label='从第几秒开始（通常为 0，也可用于跳过已完成内容）',minimum=0)
    gr.Markdown('说话人编号仅在每段内有效。跨段身份、公式和听不清的内容需要回听核对。')
    button=gr.Button('开始本地转写',variant='primary')
    status=gr.Textbox(label='进度与最近结果',lines=16)
    folder=gr.Textbox(label='结果目录（TXT / SRT / JSON）')
    button.click(transcribe,[path,upload,context,chunk,preview,start],[status,folder],concurrency_limit=1)
if __name__=='__main__':
    app.queue(default_concurrency_limit=1,max_size=1).launch(server_name='127.0.0.1',server_port=7864,
        share=False,inbrowser=False,show_error=True)
