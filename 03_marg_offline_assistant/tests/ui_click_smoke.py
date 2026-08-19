import tkinter as tk
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_marg_desktop import MargDesktopApp


class FakeEngine:
    def ask(self, question, intent_model=None):
        return {
            "question": question,
            "answer": "Chapter 2, Verse 47 says: Test answer for the UI click path.",
            "sources": [{"reference": "Chapter 2, Verse 47"}],
            "intent": "money_work",
        }


class FakeIntentModel:
    def predict(self, question):
        return {"intent": "money_work", "confidence": 0.99, "probabilities": {}}


class SlowEngine(FakeEngine):
    def ask(self, question, intent_model=None):
        time.sleep(0.35)
        return super().ask(question, intent_model=intent_model)


def main():
    root = tk.Tk()
    app = MargDesktopApp(root, engine=FakeEngine(), intent_model=FakeIntentModel())
    root.update()

    assert str(app.ask_button["state"]) in {"normal", "active"}, app.ask_button["state"]
    assert app.ask_button["bg"] == "#8f542d"
    assert app.ask_button["fg"] == "white"
    app.question.insert("1.0", "How should I work?")
    app.ask_button.invoke()

    for _ in range(20):
        root.update()
        text = app.get_chat_text()
        if "Test answer for the UI click path" in text:
            print("UI click smoke OK")
            root.destroy()
            return
        root.after(50)

    root.destroy()
    raise AssertionError("Ask button did not render the fake answer")


def async_click_main():
    root = tk.Tk()
    app = MargDesktopApp(root, engine=SlowEngine(), intent_model=FakeIntentModel())
    root.update()

    app.question.insert("1.0", "How should I work?")
    start = time.monotonic()
    app.ask_button.invoke()
    elapsed = time.monotonic() - start

    for _ in range(30):
        root.update()
        if str(app.ask_button["state"]) == "normal":
            break
        root.after(50)

    assert elapsed < 0.1, f"Ask click blocked the UI for {elapsed:.3f}s"
    assert str(app.ask_button["state"]) == "normal"
    root.destroy()


def repeated_click_main():
    root = tk.Tk()
    app = MargDesktopApp(root, engine=FakeEngine(), intent_model=FakeIntentModel())
    root.update()

    for index in range(6):
        app.question.insert("1.0", f"How should I work number {index}?")
        app.ask_button.invoke()
        for _ in range(20):
            root.update()
            if str(app.ask_button["state"]) == "normal":
                break
            root.after(50)
        text = app.get_chat_text()
        assert text.count("Test answer for the UI click path") >= index + 1
        assert str(app.ask_button["state"]) == "normal"

    root.destroy()
    print("Repeated Ask click smoke OK")


if __name__ == "__main__":
    main()
    async_click_main()
    repeated_click_main()
