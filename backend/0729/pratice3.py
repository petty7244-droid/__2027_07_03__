#這個寫的是Web Server
#這是decorator寫法 用Change方法 可以即時顯示

import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("請在下方輸入您的姓名<輸出將即時更新")
    inp = gr.Textbox(placeholder="請問您的姓名?")
    out = gr.Textbox(label="歡迎詞")

    @inp.change(inputs=inp, outputs=out)
    def Welcome(name):
        return f'歡迎來到Gradio ~ {name} !'

demo.launch()