import argparse
import queue
import re
import threading
import tkinter as tk

from backend.intent_model import IntentModel
from backend.paths import intent_model_dir, model_dir
from backend.rag_engine import MargRagEngine


BG = "#f6f1e8"
CARD = "#fffdf8"
TEXT = "#24211c"
MUTED = "#766c5f"
ACCENT = "#8f542d"
ACCENT_DARK = "#6f3f21"
DISABLED_BG = "#ddd5c9"
DISABLED_FG = "#9b9288"
BORDER = "#e3d8c8"
USER_BG = "#eadfce"


class FlatButton(tk.Frame):
    def __init__(self, parent, text, command, bg, fg, activebackground=None, font=None, padx=18, pady=8):
        super().__init__(parent, bg=bg, cursor="hand2")
        self.command = command
        self.enabled = True
        self.normal_bg = bg
        self.normal_fg = fg
        self.active_bg = activebackground or bg
        self.disabled_bg = DISABLED_BG
        self.disabled_fg = DISABLED_FG
        self.label = tk.Label(self, text=text, bg=bg, fg=fg, font=font, padx=padx, pady=pady, cursor="hand2")
        self.label.pack(fill=tk.BOTH, expand=True)
        for widget in (self, self.label):
            widget.bind("<Button-1>", self._click)
            widget.bind("<Enter>", self._enter)
            widget.bind("<Leave>", self._leave)

    def _click(self, _event=None):
        self.invoke()

    def _enter(self, _event=None):
        if self.enabled:
            self.configure(bg=self.active_bg)

    def _leave(self, _event=None):
        if self.enabled:
            self.configure(bg=self.normal_bg)

    def configure(self, cnf=None, **kwargs):
        if "state" in kwargs:
            self.enabled = kwargs.pop("state") != tk.DISABLED
        if "bg" in kwargs:
            self.normal_bg = kwargs.pop("bg")
        if "fg" in kwargs:
            self.normal_fg = kwargs.pop("fg")
        if "activebackground" in kwargs:
            self.active_bg = kwargs.pop("activebackground")
        if "cursor" in kwargs:
            cursor = kwargs.pop("cursor")
            super().configure(cursor=cursor)
            self.label.configure(cursor=cursor)
        bg = self.normal_bg if self.enabled else self.disabled_bg
        fg = self.normal_fg if self.enabled else self.disabled_fg
        super().configure(bg=bg, **kwargs)
        self.label.configure(bg=bg, fg=fg)

    config = configure

    def __getitem__(self, key):
        if key == "state":
            return tk.NORMAL if self.enabled else tk.DISABLED
        if key == "bg":
            return self.normal_bg if self.enabled else self.disabled_bg
        if key == "fg":
            return self.normal_fg if self.enabled else self.disabled_fg
        return self.label[key]

    def invoke(self):
        if self.enabled and self.command:
            return self.command()
        return None


class MargDesktopApp:
    def __init__(self, root, engine=None, intent_model=None):
        self.root = root
        self.root.title("Marg")
        self.root.geometry("920x680")
        self.root.minsize(720, 540)
        self.engine = engine or MargRagEngine.load(model_dir())
        self.intent_model = intent_model or IntentModel.load(intent_model_dir())
        self.answer_queue = queue.Queue()
        self.is_answering = False
        self.transcript = []
        self._build_ui()
        self.root.after(100, self.focus_question_input)

    def _build_ui(self):
        self.root.configure(bg=BG)
        shell = tk.Frame(self.root, bg=BG, padx=30, pady=24)
        shell.pack(fill=tk.BOTH, expand=True)
        shell.grid_columnconfigure(0, weight=1)
        shell.grid_rowconfigure(1, weight=1)

        header = tk.Frame(shell, bg=BG)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        header.grid_columnconfigure(1, weight=1)

        logo = tk.Canvas(header, width=42, height=42, bg=BG, highlightthickness=0)
        logo.create_oval(3, 3, 39, 39, fill=ACCENT, outline=ACCENT)
        logo.create_text(21, 21, text="M", fill="white", font=("Helvetica", 19, "bold"))
        logo.grid(row=0, column=0, rowspan=2, padx=(0, 12), sticky="w")

        tk.Label(header, text="Marg", bg=BG, fg=TEXT, font=("Helvetica", 24, "bold")).grid(row=0, column=1, sticky="w")
        tk.Label(header, text="Offline Bhagavad Gita assistant", bg=BG, fg=MUTED, font=("Helvetica", 13)).grid(row=1, column=1, sticky="w")

        self.new_chat_button = FlatButton(
            header,
            text="New chat",
            command=self.start_fresh_chat,
            bg="#efe5d7",
            fg=TEXT,
            activebackground="#e5d5c0",
            font=("Helvetica", 12, "bold"),
            padx=22,
            pady=10,
        )
        self.new_chat_button.grid(row=0, column=2, rowspan=2, sticky="e")

        chat_shell = tk.Frame(shell, bg=BG)
        chat_shell.grid(row=1, column=0, sticky="nsew")
        chat_shell.grid_rowconfigure(0, weight=1)
        chat_shell.grid_columnconfigure(0, weight=1)

        self.chat = tk.Canvas(
            chat_shell,
            bg=BG,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
        )
        self.chat.grid(row=0, column=0, sticky="nsew")
        scrollbar = tk.Scrollbar(chat_shell, command=self.chat.yview, relief=tk.FLAT, width=10)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.chat.configure(yscrollcommand=scrollbar.set)
        self.chat_messages = tk.Frame(self.chat, bg=BG)
        self.chat_window = self.chat.create_window((0, 0), window=self.chat_messages, anchor="nw")
        self.chat_messages.bind("<Configure>", self._sync_chat_scroll)
        self.chat.bind("<Configure>", self._sync_chat_width)
        self._show_welcome()

        composer = tk.Frame(shell, bg=BG)
        composer.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        composer.grid_columnconfigure(0, weight=1)

        input_shell = tk.Frame(
            composer,
            bg="#fffaf1",
            highlightthickness=1,
            highlightbackground="#cdbfaa",
            highlightcolor=ACCENT,
        )
        input_shell.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        input_shell.grid_columnconfigure(0, weight=1)

        self.question = tk.Text(
            input_shell,
            height=3,
            wrap=tk.WORD,
            bg="#fffaf1",
            fg=TEXT,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
            insertbackground=TEXT,
            font=("Helvetica", 14),
            padx=14,
            pady=10,
            takefocus=1,
        )
        self.question.grid(row=0, column=0, sticky="ew")
        self.question.bind("<Command-Return>", lambda _event: self.ask())
        self.question.bind("<Control-Return>", lambda _event: self.ask())

        self.ask_button = FlatButton(
            composer,
            text="Ask",
            command=self.ask,
            bg=ACCENT,
            fg="white",
            activebackground="#74411f",
            font=("Helvetica", 13, "bold"),
            padx=30,
            pady=10,
        )
        self.ask_button.grid(row=0, column=1, sticky="ns")

        self.status = tk.Label(shell, text="Ready", bg=BG, fg=MUTED, font=("Helvetica", 11))
        self.status.grid(row=3, column=0, sticky="w", pady=(8, 0))
        self._set_ask_button_enabled(True)

    def _show_welcome(self):
        self._clear_chat()
        self._append_marg(
            "Ask a question about duty, anger, peace, focus, work, money, or daily life.\n"
            "Example: How can I control anger?"
        )

    def _clear_chat(self):
        self.transcript = []
        for child in self.chat_messages.winfo_children():
            child.destroy()
        self.chat.yview_moveto(0)

    def _append_user(self, question):
        self._append_message("You", question, align="right")

    def _append_marg(self, answer):
        self._append_message("Marg", answer, align="left")

    def _append_message(self, sender, body, align):
        self.transcript.append((sender, body.strip()))
        outer = tk.Frame(self.chat_messages, bg=BG)
        outer.pack(fill=tk.X, pady=(0, 14), padx=10)
        outer.grid_columnconfigure(0, weight=1)
        column = 1 if align == "right" else 0
        outer.grid_columnconfigure(column, weight=0)

        bubble_bg = USER_BG if align == "right" else CARD
        sender_fg = MUTED if align == "right" else ACCENT
        card = tk.Frame(outer, bg=bubble_bg, padx=16, pady=12, highlightthickness=1, highlightbackground=BORDER)
        card.grid(row=0, column=column, sticky="e" if align == "right" else "w")

        tk.Label(card, text=sender, bg=bubble_bg, fg=sender_fg, font=("Helvetica", 11, "bold")).pack(anchor="w")
        tk.Label(
            card,
            text=body.strip(),
            bg=bubble_bg,
            fg=TEXT,
            font=("Helvetica", 14),
            justify=tk.LEFT,
            wraplength=720,
        ).pack(anchor="w", pady=(8, 0))
        self.root.after_idle(lambda: self.chat.yview_moveto(1.0))

    def get_chat_text(self):
        return "\n".join(f"{sender}\n{body}" for sender, body in self.transcript)

    def _sync_chat_scroll(self, _event=None):
        self.chat.configure(scrollregion=self.chat.bbox("all"))

    def _sync_chat_width(self, event):
        self.chat.itemconfigure(self.chat_window, width=max(event.width - 12, 320))

    def start_fresh_chat(self):
        self.question.delete("1.0", tk.END)
        self._show_welcome()
        self.status.config(text="Ready")
        self._set_ask_button_enabled(True)
        self.focus_question_input()

    def focus_question_input(self):
        self.question.focus_force()
        self.question.mark_set(tk.INSERT, "1.0")

    def ask(self):
        if self.is_answering:
            self.status.config(text="Marg is still answering...")
            return

        question = self.question.get("1.0", tk.END).strip()
        if not question:
            self.status.config(text="Please enter a question.")
            self.focus_question_input()
            return

        self.question.delete("1.0", tk.END)
        self._append_user(question)

        if not is_clear_question(question):
            self._append_marg("Please ask a clear Bhagavad Gita question. Example: How can I control anger?")
            self.status.config(text="Waiting for a clear question")
            self.focus_question_input()
            return

        self.is_answering = True
        self._set_ask_button_enabled(False)
        self.status.config(text="Thinking offline...")
        threading.Thread(target=self._answer_question, args=(question,), daemon=True).start()
        self.root.after(30, self._poll_answer_queue)

    def _answer_question(self, question):
        try:
            response = self.engine.ask(question, intent_model=self.intent_model)
            self.answer_queue.put(("ok", response))
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self.answer_queue.put(("error", exc))

    def _poll_answer_queue(self):
        try:
            status, payload = self.answer_queue.get_nowait()
        except queue.Empty:
            if self.is_answering:
                self.root.after(30, self._poll_answer_queue)
            return

        if status == "ok":
            self._render_response(payload)
        else:
            self._render_error(payload)

    def _render_response(self, response):
        self._append_marg(format_chat_answer(response))
        self.status.config(text="Answered offline")
        self.is_answering = False
        self._set_ask_button_enabled(True)
        self.focus_question_input()

    def _render_error(self, error):
        self._append_marg(f"Marg could not answer this question. {error}")
        self.status.config(text="Answer failed")
        self.is_answering = False
        self._set_ask_button_enabled(True)
        self.focus_question_input()

    def _set_ask_button_enabled(self, enabled):
        if enabled:
            self.ask_button.config(
                state=tk.NORMAL,
                bg=ACCENT,
                fg="white",
                activebackground=ACCENT_DARK,
                cursor="hand2",
            )
        else:
            self.ask_button.config(
                state=tk.DISABLED,
                bg=DISABLED_BG,
                fg=DISABLED_FG,
                activebackground=DISABLED_BG,
                cursor="arrow",
            )


def format_user_question(question):
    return f"You\n{question.strip()}"


def should_focus_question_input_on_start():
    return True


def format_chat_answer(response):
    answer = clean_answer_text(response.get("answer", "").strip())
    sources = response.get("sources", [])
    if sources:
        answer += f"\n\nReference: {sources[0].get('reference', 'Bhagavad Gita')}"
    return answer


def clean_answer_text(answer):
    cleaned = re.sub(r"\s+", " ", answer.strip())
    cleaned = re.sub(r" Related support also appears.*$", "", cleaned)
    cleaned = re.sub(r"^(Chapter \d+, Verse \d+) says:\s*", r"According to Bhagavad Gita \1:\n", cleaned)
    cleaned = re.sub(r"^(Chapter \d+, Verse \d+) gives the main direction:\s*", r"According to Bhagavad Gita \1:\n", cleaned)
    cleaned = cleaned.replace(" For anger,", "\nFor anger,")
    cleaned = cleaned.replace(" For earning", "\nFor earning")
    cleaned = cleaned.replace(" For peace", "\nFor peace")
    cleaned = cleaned.replace(" For duty", "\nFor duty")
    cleaned = cleaned.replace(" The practical lesson", "\nThe practical lesson")
    return cleaned


def is_clear_question(text):
    cleaned = text.strip().lower()
    if cleaned in {"who am i", "who am i?"}:
        return True
    words = re.findall(r"[a-zA-Z]{2,}", cleaned)
    spiritual_terms = {
        "god",
        "krishna",
        "atma",
        "paramatma",
        "parmatma",
        "soul",
        "worship",
        "pray",
        "prayer",
        "bhakti",
        "devotion",
    }
    if "?" in cleaned and any(word in spiritual_terms for word in words):
        return True
    if len(words) < 3:
        return False
    question_words = {"how", "what", "why", "when", "where", "who", "can", "should", "do", "does", "is", "are"}
    has_question_shape = "?" in cleaned or any(word in question_words for word in words)
    if not has_question_shape:
        return False

    vowel_rich_words = sum(1 for word in words if sum(char in "aeiou" for char in word) >= 1)
    repeated_noise = any(len(set(word)) <= 2 and len(word) > 5 for word in words)
    long_unknown_blob = any(len(word) > 16 for word in words)
    return vowel_rich_words >= 2 and not repeated_noise and not long_unknown_blob


def run_self_test(question="How can I control anger?"):
    engine = MargRagEngine.load(model_dir())
    intent_model = IntentModel.load(intent_model_dir())
    response = engine.ask(question, intent_model=intent_model)
    first_source = response["sources"][0]["reference"] if response["sources"] else "no source"
    return f"Marg self-test OK\n{first_source}\n{response['answer']}"


def main():
    parser = argparse.ArgumentParser(description="Run Marg desktop app.")
    parser.add_argument("--self-test", action="store_true", help="Load the packaged model, answer once, then exit.")
    args = parser.parse_args()

    if args.self_test:
        print(run_self_test())
        return

    root = tk.Tk()
    MargDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
