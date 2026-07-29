#輸出多個組件(Return List)
import gradio as gr

with gr.Blocks() as demo:
    food_box = gr.Number(value=10, label="剩餘食物數量")
    feed_box = gr.Number(label="餵食數量")
    status_box = gr.Textbox(label="寵物狀態")

    @gr.Button("餵食寵物").click(inputs=[food_box,feed_box] , outputs=[food_box,status_box])
    def eat(food,feed):
        if food >= feed:
            return food-feed, "飽足"
        elif food < feed and food > 0 :
            return food-feed, "飽足"
        else:
            return food, "肚子餓了"

demo.launch()