import json
import os
import urllib.request

from textual import work
from textual.app import App, ComposeResult
from textual.containers import HorizontalGroup
from textual.widgets import Static, Input, Header, Markdown

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:9b"

MAX_TOOL_OUTPUT = 4000  # keep tool results from flooding the context
MAX_STEPS = 10  # stop a confused model from looping forever
# sent only in agent mode, not stored in history; without it qwen denies having tools or an agent mode
AGENT_PROMPT = {"role": "system", "content": (
    "Agent mode is ON. You are running on the user's computer with tools that read their local files. "
    f"The current directory is {os.getcwd()}. Use the tools instead of saying you can't access files."
)}


def list_dir(path="."):
    return "\n".join(sorted(os.listdir(path)))


def read_file(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


TOOLS = {"list_dir": list_dir, "read_file": read_file}
TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List the files and folders in a directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory path, default '.'"}}},
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file and return its contents.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path"}}, "required": ["path"]},
    }},
]


def run_tool(call):
    fn = call["function"]
    try:
        result = str(TOOLS[fn["name"]](**fn["arguments"]))
    except Exception as e:  # hand errors back so the model can correct itself
        result = f"error: {e}"
    return result[:MAX_TOOL_OUTPUT]


def chat(history, tools=None, think=False):
    """Stream message chunks from Ollama. history: list of {"role", "content", ...} dicts."""
    body = {"model": MODEL, "messages": history, "stream": True, "think": think}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        for line in resp:  # one JSON object per line
            chunk = json.loads(line)
            if chunk.get("done"):
                break
            yield chunk["message"]


def bulleted(widget, row_class):
    """Put a bullet column left of widget so wrapped lines stay indented."""
    return HorizontalGroup(Static("•", classes="bullet"), widget, classes=row_class)


class TextUpdateApp(App):
    CSS_PATH = "main.tcss"

    def __init__(self):
        super().__init__()
        self.history = []
        self.messages = []  # every prompt/response/tool widget, never removed
        self.turn = 0
        self.agent = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static("Prompt Qwen for a response.", id="qwen_response_init")
        yield Input(placeholder="Ask Qwen... (/agent toggles agent mode)", id="qwen_input")

    def on_mount(self) -> None:
        self.title = "Qwen CLI"
        self.sub_title = "Version 1.0.0"

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.submit()

    def submit(self) -> None:
        box = self.query_one("#qwen_input", Input)
        prompt = box.value.strip()
        if not prompt:
            return
        box.value = ""
        if prompt == "/agent":
            self.agent = not self.agent
            self.sub_title = "agent mode" if self.agent else ""
            return
        # query() matches nothing once it's gone, so this only removes it on the first prompt
        self.query("#qwen_response_init").remove()
        self.history.append({"role": "user", "content": prompt})
        self.turn += 1
        # markup=False so brackets in prompts aren't parsed as Textual markup
        asked = Static(prompt, id=f"prompt_{self.turn}", classes="user_prompt", markup=False)
        out = Markdown("Thinking...", id=f"response_{self.turn}", classes="qwen_reply")
        self.messages += [asked, out]
        self.mount(bulleted(asked, "prompt_row"), bulleted(out, "reply_row"), before=box)
        self.call_after_refresh(self.screen.scroll_end)
        self.ask_qwen(out, self.agent)

    @work(thread=True, exclusive=True)
    def ask_qwen(self, out: Markdown, agent: bool) -> None:
        # plain mode: no tools, so the loop ends after one round
        tools = TOOL_SPECS if agent else None
        reply = ""
        start = len(self.history)
        try:
            for _ in range(MAX_STEPS):
                content, calls = "", []
                messages = [AGENT_PROMPT] + self.history if agent else self.history
                for msg in chat(messages, tools, think=agent):
                    piece = msg.get("content", "")
                    if piece:  # thinking chunks have no content
                        # first chunk replaces "Thinking...", the rest append so we don't re-parse
                        self.call_from_thread(out.append if reply else out.update, piece)
                        reply += piece
                        content += piece
                    calls += msg.get("tool_calls", [])
                if not calls:
                    self.history.append({"role": "assistant", "content": content})
                    return
                self.history.append({"role": "assistant", "content": content, "tool_calls": calls})
                for call in calls:
                    fn = call["function"]
                    args = ", ".join(f"{k}={v}" for k, v in fn["arguments"].items())
                    line = Static(f"→ {fn['name']}({args})", classes="tool_call", markup=False)
                    self.messages.append(line)
                    self.call_from_thread(self.mount, line, before=out.parent)  # above the reply's row
                    self.history.append({"role": "tool", "tool_name": fn["name"], "content": run_tool(call)})
            self.call_from_thread(out.append, f"\n\n*Stopped after {MAX_STEPS} steps.*")
        except OSError as e:
            self.call_from_thread(out.update, f"Couldn't reach Ollama: {e}")
            del self.history[start - 1:]  # drop the unanswered prompt and any partial tool rounds

if __name__ == "__main__":
    app = TextUpdateApp()
    app.run()
