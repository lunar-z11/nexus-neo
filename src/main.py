import json
import os
import threading
import urllib.request

from textual import work
from textual.app import App, ComposeResult
from textual.containers import HorizontalGroup, Vertical
from textual.screen import ModalScreen
from textual.widgets import Static, Input, Header, Markdown, Button
from textual.worker import get_current_worker

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:9b"

MAX_TOOL_OUTPUT = 4000  # keep tool results from flooding the context
MAX_STEPS = 10  # stop a confused model from looping forever
NUM_CTX = 8192  # ollama defaults to 4096, which tool results fill fast
PREVIEW_LINES = 20  # how much of a pending write the approval dialog shows
ROOT = os.getcwd()  # writes are confined to the folder the app was started from
# sent only in agent mode, not stored in history; without it qwen denies having tools or an agent mode
AGENT_PROMPT = {"role": "system", "content": (
    "Agent mode is ON. You are running on the user's computer with tools that read and write their local files. "
    f"The current directory is {ROOT}; you can only write inside it. Use the tools instead of saying you can't "
    "access files. The user approves each write; if they decline one, don't retry it."
)}


def list_dir(path="."):
    return "\n".join(sorted(os.listdir(path)))


def read_file(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def inside_root(path):
    # realpath follows symlinks and "..", so neither can escape ROOT
    return os.path.commonpath([ROOT, os.path.realpath(path)]) == ROOT


def wants_append(args):
    return str(args.get("append", False)).lower() == "true"  # models sometimes send "true" as a string


def write_file(path, content, append=False):
    if not inside_root(path):
        raise PermissionError(f"{path} is outside {ROOT}")
    os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
    mode = "a" if wants_append({"append": append}) else "w"
    with open(path, mode, encoding="utf-8") as f:
        f.write(content)
    return f"{'appended' if mode == 'a' else 'wrote'} {len(content)} characters to {path}"


TOOLS = {"list_dir": list_dir, "read_file": read_file, "write_file": write_file}
NEEDS_APPROVAL = {"write_file"}
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
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write text to a file, creating it if needed. Overwrites unless append is true.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path"},
            "content": {"type": "string", "description": "Text to write"},
            "append": {"type": "boolean", "description": "Add to the end instead of overwriting, default false"}},
            "required": ["path", "content"]},
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
    body = {"model": MODEL, "messages": history, "stream": True, "think": think, "options": {"num_ctx": NUM_CTX}}
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


class ApproveWrite(ModalScreen[bool]):
    """Show a pending write_file call; dismisses with True to allow it."""

    BINDINGS = [("y", "choose(True)", "Allow"), ("n,escape", "choose(False)", "Deny")]

    def __init__(self, args):
        super().__init__()
        self.args = args

    def compose(self) -> ComposeResult:
        path = str(self.args.get("path", ""))
        if wants_append(self.args):
            action = "append to"
        else:
            action = "overwrite" if os.path.exists(path) else "create"
        lines = str(self.args.get("content", "")).splitlines()
        preview = "\n".join(lines[:PREVIEW_LINES])
        if len(lines) > PREVIEW_LINES:
            preview += f"\n... ({len(lines) - PREVIEW_LINES} more lines)"
        with Vertical(id="approve_box"):
            yield Static(f"Qwen wants to {action} {path}", id="approve_title", markup=False)
            yield Static(preview or "(empty)", id="approve_preview", markup=False)
            with HorizontalGroup(id="approve_buttons"):
                yield Button("Allow (y)", variant="success", id="allow")
                yield Button("Deny (n)", variant="error", id="deny")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "allow")

    def action_choose(self, allow: bool) -> None:
        self.dismiss(allow)


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

    def approve(self, call) -> bool:
        """Block the worker thread until the user allows or denies call."""
        fn = call["function"]
        if fn["name"] not in NEEDS_APPROVAL or not inside_root(str(fn["arguments"].get("path", ""))):
            return True  # out-of-root writes are refused by the tool itself, no need to ask
        answer, done = [], threading.Event()
        self.call_from_thread(self.push_screen, ApproveWrite(fn["arguments"]), lambda ok: (answer.append(ok), done.set()))
        while not done.wait(0.1):
            if get_current_worker().is_cancelled:  # app is quitting
                return False
        return answer[0]

    @work(thread=True, exclusive=True)
    def ask_qwen(self, out: Markdown, agent: bool) -> None:
        # plain mode: no tools, so the loop ends after one round
        tools = TOOL_SPECS if agent else None
        reply = ""
        start = len(self.history)
        think = False  # tool-picking rounds skip thinking; only the final answer gets it
        try:
            for _ in range(MAX_STEPS):
                content, calls, restart = "", [], False
                messages = [AGENT_PROMPT] + self.history if agent else self.history
                for msg in chat(messages, tools, think):
                    piece = msg.get("content", "")
                    if not reply and not piece.strip():
                        piece = ""  # leading whitespace, often emitted before a tool call
                    if piece and agent and not think and not calls:
                        # prose means it's answering rather than calling tools: drop it, redo with thinking
                        restart = True
                        break
                    if piece:  # thinking chunks have no content
                        # first chunk replaces "Thinking...", the rest append so we don't re-parse
                        self.call_from_thread(out.append if reply else out.update, piece)
                        reply += piece
                        content += piece
                    calls += msg.get("tool_calls", [])
                if restart:
                    think = True
                    continue
                if not calls:
                    self.history.append({"role": "assistant", "content": content})
                    return
                self.history.append({"role": "assistant", "content": content, "tool_calls": calls})
                for call in calls:
                    fn = call["function"]
                    # shorten values so a write_file call doesn't dump the whole file into the chat
                    args = ", ".join(f"{k}={str(v)[:40]!r}" if len(str(v)) > 40 else f"{k}={v}"
                                     for k, v in fn["arguments"].items())
                    line = Static(f"→ {fn['name']}({args})", classes="tool_call", markup=False)
                    self.messages.append(line)
                    self.call_from_thread(self.mount, line, before=out.parent)  # above the reply's row
                    result = run_tool(call) if self.approve(call) else "error: the user declined this action"
                    self.history.append({"role": "tool", "tool_name": fn["name"], "content": result})
                think = False  # back to picking tools
            self.call_from_thread(out.append, f"\n\n*Stopped after {MAX_STEPS} steps.*")
        except OSError as e:
            self.call_from_thread(out.update, f"Couldn't reach Ollama: {e}")
            del self.history[start - 1:]  # drop the unanswered prompt and any partial tool rounds

if __name__ == "__main__":
    app = TextUpdateApp()
    app.run()
