import gradio as gr

# 全域變數,所有使用者共享

scores = [] #設一個空字串來接收資料

def track_score(score):
    scores.append(score)
    top_scores = sorted(scores, reverse=True)[:3]
    return top_scores

# gradio的呈現寫法
demo = gr.Interface(
    title = "🏆 全域分數排行榜",
    description = "請輸入您的分數！本系統會追蹤所有使用者的前 3 名最高分數。請開啟多個瀏覽器分頁同時測試，觀察資料如何共享。",
    inputs = gr.Number(label="您的分數"),
    outputs = gr.JSON(label="前 3 名最高分數排行榜"),
    fn = track_score,
    
)

demo.launch(share=True)