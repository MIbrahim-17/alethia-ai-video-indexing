# AI chat history

## Claude Code

**Claude Code** (the VS Code extension) is the only AI coding tool used for this task. Every line of code in the repository was written in Claude Code.

`claude_code/` holds the Claude Code `.jsonl` session files for this project, copied unedited from `~/.claude/projects/` by `tools/export_chats.py`. The only change the brief allows is replacing passwords and API tokens with `[REMOVED]`, and none were found in the transcript. The session was continued after context compaction, so the full history is in the same file.

## ChatGPT (web, incognito, not logged in)

Several ChatGPT chats in the browser were also used, for two things only:

- **Drafting prompts for Claude Code,** so the prompts were clear and effective.
- **Finding out about finer-grained features** of libraries such as PyAV and TensorRT.

This kept the main Claude Code session's token use down and made the session easier to manage. ChatGPT wrote no code in this repository.

These chats were held in incognito browser windows without logging in. That allowed several windows open at once without session or weekly usage limits. The logged-in ChatGPT account was kept for unrelated research work, which needed its whole limit.

Because these chats weren't logged in, ChatGPT offers no Share link for them, and they were lost when the windows closed. So there is no `links.txt`, and the chats can't be included here.

**A note on the Claude Code transcript:** on 21 Nov 2026 (14:44 UTC), Claude Code asked which other AI tools had been used, such as ChatGPT on the web, and I answered "None". That answer left out the ChatGPT use described above. This README gives the complete account.
