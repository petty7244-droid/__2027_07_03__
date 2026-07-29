#這個寫的是Web Server
#這是decorator寫法 配置\多個事件按鈕

import gradio as gr

with gr.Blocks() as demo:
    a = gr.Number(label="數值A")
    b = gr.Number(label="數值B")

    abtn = gr.Button("將 A 的值加1後 填入B")
    bbtn = gr.Button("將 B 的值加1後 填入A")

    @abtn.click(inputs=a , outputs= b)
    @bbtn.click(inputs=b , outputs= a)
    def increase(num):
        return num+1

demo.launch()