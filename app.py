import os
import gradio as gr

from apscheduler.schedulers.background import BackgroundScheduler
from dabstep_benchmark.content import TITLE, INTRODUCTION_TEXT, SUBMISSION_TEXT, CITATION_BUTTON_TEXT, CITATION_BUTTON_LABEL, VALIDATION_GUIDELINES
from dabstep_benchmark.leaderboard import *


def restart_space():
    HF_API.restart_space(repo_id=HF_LEADERBOARD)
    

def download_leaderboard(type):
    verified_lb, unverified_lb = generate_leaderboard_df()
    if type == "verified":
        df_to_download = verified_lb
    if type == "unverified":
        df_to_download = unverified_lb

    path = f"data/{type}_leaderboard.csv"
    if os.path.exists(path):
        os.remove(path)
    df_to_download.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    os.makedirs("data/task_scores", exist_ok=True)
    refresh(only_leaderboard=False)

    demo = gr.Blocks()
    with demo:
        gr.Markdown(TITLE)
        gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")
        
        # Generate initial leaderboard data
        validated_lb, unvalidated_lb = generate_leaderboard_df()

        with gr.Tab("Validated"):
            verified_table = gr.Dataframe(
                value=validated_lb,
                datatype=["markdown", "str", "str", "str", "markdown", "str", "str", "str"],
                interactive=False,
                column_widths=["20%"],
                wrap=True,
            )
            verified_download = gr.DownloadButton(
                label="Download Leaderboard",
                elem_id="download-verified-lb",
            )
        
        with gr.Tab("Unvalidated"):
            unverified_table = gr.Dataframe(
                value=unvalidated_lb,
                datatype=["markdown", "str", "str", "str", "markdown", "str", "str", "str"],
                interactive=False,
                column_widths=["20%"],
                wrap=True,
            )
            unverified_download = gr.DownloadButton(
                label="Download Leaderboard",
                elem_id="download-unverified-lb",
            )
        # create a Gradio event listener that runs when the page is loaded to populate the dataframe
        demo.load(generate_leaderboard_df, inputs=None, outputs=[verified_table, unverified_table])

        verified_download.click(
            download_leaderboard,
            inputs=[gr.Textbox(value="verified", visible=False)],
            outputs=[verified_download]
        )
        unverified_download.click(
            download_leaderboard,
            inputs=[gr.Textbox(value="unverified", visible=False)],
            outputs=[unverified_download]
        )

        refresh_button = gr.Button("Refresh")
        refresh_button.click(
            refresh,
            inputs=[
                gr.Checkbox(value=True, visible=False)
            ],
            outputs=[
                verified_table, unverified_table
            ],
        )
        gr.Markdown(VALIDATION_GUIDELINES, elem_classes="markdown-text")
                    
        with gr.Row():
            with gr.Accordion("📙 Citation", open=False):
                citation_button = gr.Textbox(
                    value=CITATION_BUTTON_TEXT,
                    label=CITATION_BUTTON_LABEL,
                    lines=len(CITATION_BUTTON_TEXT.split("\n")),
                    elem_id="citation-button",
                )  # .style(show_copy_button=True)

        with gr.Accordion("Submit new agent answers for evaluation"):
            with gr.Row():
                gr.Markdown(SUBMISSION_TEXT, elem_classes="markdown-text")
            with gr.Row():
                with gr.Column():
                    split = gr.Radio(["all"], value="all", label="Split", visible=False)
                    agent_name_textbox = gr.Textbox(label="Agent name")
                    model_family_textbox = gr.Textbox(label="Model family")
                    system_prompt_textbox = gr.Textbox(label="System prompt example")
                    repo_url_textbox = gr.Textbox(label="Repo URL with agent code")
                with gr.Column():
                    organisation = gr.Textbox(label="Organisation")
                    mail = gr.Textbox(
                        label="Contact email (will be stored privately, & used if there is an issue with your submission)")
                    file_output = gr.File()

            with gr.Row():
                gr.LoginButton()
                submit_button = gr.Button("Submit answers")
            submission_result = gr.Markdown()
            submit_button.click(
                process_submission,
                [
                    split,
                    agent_name_textbox,
                    model_family_textbox,
                    repo_url_textbox,
                    file_output,
                    organisation,
                    mail
                ],
                submission_result,
            )

    scheduler = BackgroundScheduler()
    scheduler.add_job(restart_space, "interval", seconds=3600*24)
    scheduler.start()
    demo.launch(debug=True)