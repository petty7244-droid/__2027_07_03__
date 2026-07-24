import gradio as gr

def greet(name, intensity):
    return "Hello," + name + "!" * int(intensity)

#建立Interface實體
demo = gr.Interface(
    fn = greet,
    inputs = ['text','slider'],
    outputs = ['text']

)

#執行DEMO實體
demo.launch('share=True')