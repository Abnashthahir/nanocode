#!/usr/bin/env python3
"""nanocode - minimal claude code alternative"""

import glob as globlib, json, os, re, subprocess, urllib.request, sys, time, hashlib
from pathlib import Path
from datetime import datetime
from urllib.error import URLError, HTTPError
from functools import lru_cache

# Load config
def load_config():
    config_paths = [".nanocoderc", os.path.expanduser("~/.nanocoderc")]
    for path in config_paths:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return {}

CONFIG = load_config()
API_URL = os.environ.get("API_URL", CONFIG.get("api_url", "https://api.anthropic.com/v1/messages"))
MODEL = os.environ.get("MODEL", CONFIG.get("model", "claude-opus-4-5"))
API_KEY = os.environ.get("API_KEY", CONFIG.get("api_key", os.environ.get("ANTHROPIC_API_KEY", "")))

# ANSI colors
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
BLUE, CYAN, GREEN, YELLOW, RED = "\033[34m", "\033[36m", "\033[32m", "\033[33m", "\033[31m"

# File operation history for undo
FILE_HISTORY = []
ACTIVE_TRANSACTION = []

# Caching
FILE_CACHE = {}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_CONVERSATION_TOKENS = 150000  # Rough estimate


# --- Validation helpers ---


def validate_path(path, must_exist=False, must_be_file=False):
    """Validate file paths"""
    if not path or ".." in path:
        raise ValueError("Invalid path")
    
    p = Path(path)
    
    if must_exist and not p.exists():
        raise ValueError(f"Path does not exist: {path}")
    
    if must_be_file and p.exists() and not p.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    return str(p)


def get_file_hash(path):
    """Get file hash for cache invalidation"""
    try:
        stat = os.stat(path)
        return f"{path}:{stat.st_mtime}:{stat.st_size}"
    except:
        return None


def read_file_cached(path):
    """Read file with caching"""
    file_hash = get_file_hash(path)
    
    if file_hash and file_hash in FILE_CACHE:
        return FILE_CACHE[file_hash]
    
    # Check file size
    size = os.path.getsize(path)
    if size > MAX_FILE_SIZE:
        raise ValueError(f"File too large ({size // 1024 // 1024}MB), max {MAX_FILE_SIZE // 1024 // 1024}MB")
    
    with open(path) as f:
        content = f.read()
    
    # Cache if reasonable size
    if size < 1024 * 1024:  # Cache files < 1MB
        FILE_CACHE[file_hash] = content
        # Limit cache size
        if len(FILE_CACHE) > 50:
            FILE_CACHE.pop(next(iter(FILE_CACHE)))
    
    return content


def invalidate_cache(path):
    """Invalidate cache for a file"""
    keys_to_remove = [k for k in FILE_CACHE.keys() if k.startswith(path + ":")]
    for k in keys_to_remove:
        del FILE_CACHE[k]


# --- Tool implementations ---


def read(args):
    path = validate_path(args["path"], must_exist=True, must_be_file=True)
    
    try:
        content = read_file_cached(path)
        lines = content.splitlines(keepends=True)
    except ValueError as e:
        # File too large, read in chunks
        return str(e) + " - use offset and limit to read portions"
    
    offset = args.get("offset", 0)
    limit = args.get("limit", len(lines))
    
    if offset < 0 or offset >= len(lines):
        return f"error: offset {offset} out of range (file has {len(lines)} lines)"
    
    selected = lines[offset : offset + limit]
    return "".join(f"{offset + idx + 1:4}| {line}" for idx, line in enumerate(selected))


def write(args):
    path = validate_path(args["path"])
    content = args["content"]
    
    _backup_file(path)
    ACTIVE_TRANSACTION.append({"type": "write", "path": path})
    
    with open(path, "w") as f:
        f.write(content)
    
    invalidate_cache(path)
    return "ok"


def edit(args):
    path = validate_path(args["path"], must_exist=True, must_be_file=True)
    
    try:
        text = read_file_cached(path)
    except ValueError as e:
        return f"error: {e}"
    
    old, new = args["old"], args["new"]
    
    if not old:
        return "error: old_string cannot be empty"
    
    if old not in text:
        return "error: old_string not found"
    
    count = text.count(old)
    if not args.get("all") and count > 1:
        return f"error: old_string appears {count} times, must be unique (use all=true)"
    
    _backup_file(path)
    ACTIVE_TRANSACTION.append({"type": "edit", "path": path})
    replacement = text.replace(old, new) if args.get("all") else text.replace(old, new, 1)
    
    with open(path, "w") as f:
        f.write(replacement)
    
    invalidate_cache(path)
    
    # Show diff preview
    diff_lines = []
    for line in old.split("\n")[:3]:
        diff_lines.append(f"- {line}")
    for line in new.split("\n")[:3]:
        diff_lines.append(f"+ {line}")
    preview = "\n".join(diff_lines)
    if len(old.split("\n")) > 3:
        preview += "\n..."
    
    return f"ok\n{preview}"


def insert(args):
    path = validate_path(args["path"], must_exist=True, must_be_file=True)
    
    try:
        content = read_file_cached(path)
        lines = content.splitlines(keepends=True)
    except ValueError as e:
        return f"error: {e}"
    
    line_num = args["line"]
    new_content = args["content"]
    
    if line_num < 1 or line_num > len(lines) + 1:
        return f"error: line {line_num} out of range (file has {len(lines)} lines)"
    
    if not new_content.endswith("\n"):
        new_content += "\n"
    
    _backup_file(path)
    ACTIVE_TRANSACTION.append({"type": "insert", "path": path})
    lines.insert(line_num - 1, new_content)
    
    with open(path, "w") as f:
        f.writelines(lines)
    
    invalidate_cache(path)
    return f"ok (inserted at line {line_num})"


@lru_cache(maxsize=10)
def glob_cached(pattern, path="."):
    """Cached glob results"""
    full_pattern = (path + "/" + pattern).replace("//", "/")
    files = globlib.glob(full_pattern, recursive=True)
    files = sorted(files, key=lambda f: os.path.getmtime(f) if os.path.isfile(f) else 0, reverse=True)
    return tuple(files)  # tuple for caching


def glob(args):
    files = glob_cached(args["pat"], args.get("path", "."))
    return "\n".join(files) or "none"


def grep(args):
    pattern = re.compile(args["pat"])
    hits = []
    for filepath in globlib.glob(args.get("path", ".") + "/**", recursive=True):
        try:
            for line_num, line in enumerate(open(filepath), 1):
                if pattern.search(line):
                    # Show context: file:line with surrounding context
                    hits.append(f"{filepath}:{line_num}:{line.rstrip()}")
        except Exception:
            pass
    return "\n".join(hits[:50]) or "none"


def tree(args):
    path = args.get("path", ".")
    max_depth = args.get("depth", 3)
    ignore = [".git", "__pycache__", "node_modules", ".venv", "venv"]
    
    lines = []
    def walk(dir_path, prefix="", depth=0):
        if depth >= max_depth:
            return
        try:
            entries = sorted(Path(dir_path).iterdir(), key=lambda p: (not p.is_dir(), p.name))
            entries = [e for e in entries if e.name not in ignore]
            
            for i, entry in enumerate(entries):
                is_last = i == len(entries) - 1
                lines.append(f"{prefix}{'└── ' if is_last else '├── '}{entry.name}")
                if entry.is_dir():
                    walk(entry, prefix + ("    " if is_last else "│   "), depth + 1)
        except PermissionError:
            pass
    
    lines.append(path)
    walk(path)
    return "\n".join(lines)


def bash(args):
    result = subprocess.run(args["cmd"], shell=True, capture_output=True, text=True, timeout=30)
    return (result.stdout + result.stderr).strip() or "(empty)"


def lint(args):
    """Run linter on file based on extension"""
    path = args["path"]
    ext = Path(path).suffix
    
    linters = {
        ".py": "flake8 --max-line-length=100",
        ".js": "eslint",
        ".jsx": "eslint",
        ".ts": "eslint",
        ".tsx": "eslint",
    }
    
    cmd = linters.get(ext)
    if not cmd:
        return f"No linter configured for {ext} files"
    
    result = subprocess.run(f"{cmd} {path}", shell=True, capture_output=True, text=True, timeout=30)
    output = (result.stdout + result.stderr).strip()
    return output or "✓ No issues found"


def git(args):
    """Execute git commands"""
    cmd = args["cmd"]
    allowed = ["status", "diff", "log", "add", "commit", "push", "pull", "branch", "checkout"]
    
    if not any(cmd.startswith(c) for c in allowed):
        return f"Git command must start with one of: {', '.join(allowed)}"
    
    result = subprocess.run(f"git {cmd}", shell=True, capture_output=True, text=True, timeout=30)
    return (result.stdout + result.stderr).strip() or "(empty)"


def search_replace(args):
    """Advanced regex-based search and replace"""
    path = validate_path(args["path"], must_exist=True, must_be_file=True)
    pattern = args["pattern"]
    replacement = args["replacement"]
    
    try:
        text = read_file_cached(path)
    except ValueError as e:
        return f"error: {e}"
    
    _backup_file(path)
    ACTIVE_TRANSACTION.append({"type": "search_replace", "path": path})
    
    try:
        new_text = re.sub(pattern, replacement, text)
        matches = len(re.findall(pattern, text))
        
        if matches == 0:
            return "error: pattern not found"
        
        with open(path, "w") as f:
            f.write(new_text)
        
        invalidate_cache(path)
        return f"ok ({matches} replacement{'s' if matches > 1 else ''})"
    except re.error as e:
        return f"error: invalid regex - {e}"


def _backup_file(path):
    """Backup file before modifying it"""
    if os.path.exists(path):
        try:
            content = read_file_cached(path)
        except:
            with open(path) as f:
                content = f.read()
        
        FILE_HISTORY.append({"path": path, "content": content, "time": datetime.now()})
        # Keep last 10 operations
        if len(FILE_HISTORY) > 10:
            FILE_HISTORY.pop(0)


# --- Tool definitions ---

TOOLS = {
    "read": (
        "Read file with line numbers (file path, not directory)",
        {"path": "string", "offset": "number?", "limit": "number?"},
        read,
    ),
    "write": (
        "Write content to file",
        {"path": "string", "content": "string"},
        write,
    ),
    "edit": (
        "Replace old with new in file (old must be unique unless all=true)",
        {"path": "string", "old": "string", "new": "string", "all": "boolean?"},
        edit,
    ),
    "insert": (
        "Insert content at specific line number",
        {"path": "string", "line": "number", "content": "string"},
        insert,
    ),
    "glob": (
        "Find files by pattern, sorted by mtime",
        {"pat": "string", "path": "string?"},
        glob,
    ),
    "grep": (
        "Search files for regex pattern",
        {"pat": "string", "path": "string?"},
        grep,
    ),
    "tree": (
        "Display directory structure",
        {"path": "string?", "depth": "number?"},
        tree,
    ),
    "bash": (
        "Run shell command",
        {"cmd": "string"},
        bash,
    ),
    "lint": (
        "Run linter on file (supports .py, .js, .jsx, .ts, .tsx)",
        {"path": "string"},
        lint,
    ),
    "git": (
        "Execute git command (status, diff, log, add, commit, push, pull, branch, checkout)",
        {"cmd": "string"},
        git,
    ),
    "search_replace": (
        "Regex-based search and replace in file",
        {"path": "string", "pattern": "string", "replacement": "string"},
        search_replace,
    ),
}


def run_tool(name, args):
    try:
        return TOOLS[name][2](args)
    except Exception as err:
        return f"error: {err}"


def make_schema():
    result = []
    for name, (description, params, _fn) in TOOLS.items():
        properties = {}
        required = []
        for param_name, param_type in params.items():
            is_optional = param_type.endswith("?")
            base_type = param_type.rstrip("?")
            properties[param_name] = {"type": "integer" if base_type == "number" else base_type}
            if not is_optional:
                required.append(param_name)
        result.append({
            "name": name,
            "description": description,
            "input_schema": {"type": "object", "properties": properties, "required": required},
        })
    return result


def call_api(messages, system_prompt, max_retries=3):
    """Call API with automatic retry on failures"""
    for attempt in range(max_retries):
        try:
            request = urllib.request.Request(
                API_URL,
                data=json.dumps({
                    "model": MODEL,
                    "max_tokens": 8192,
                    "system": system_prompt,
                    "messages": messages,
                    "tools": make_schema(),
                }).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}",
                    "anthropic-version": "2023-06-01",
                },
            )
            response = urllib.request.urlopen(request, timeout=60)
            return json.loads(response.read())
            
        except HTTPError as e:
            error_body = e.read().decode()
            if e.code == 429:  # Rate limit
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"{YELLOW}⏺ Rate limited, retrying in {wait_time}s...{RESET}")
                    time.sleep(wait_time)
                    continue
            elif e.code >= 500:  # Server error
                if attempt < max_retries - 1:
                    print(f"{YELLOW}⏺ Server error, retrying...{RESET}")
                    time.sleep(1)
                    continue
            raise Exception(f"API error {e.code}: {error_body}")
            
        except URLError as e:
            if attempt < max_retries - 1:
                print(f"{YELLOW}⏺ Connection error, retrying...{RESET}")
                time.sleep(1)
                continue
            raise Exception(f"Connection failed: {e.reason}")
            
        except Exception as e:
            if attempt < max_retries - 1 and "timeout" in str(e).lower():
                print(f"{YELLOW}⏺ Timeout, retrying...{RESET}")
                continue
            raise
    
    raise Exception("Max retries exceeded")


def separator():
    return f"{DIM}{'─' * min(os.get_terminal_size().columns, 80)}{RESET}"


def render_markdown(text):
    return re.sub(r"\*\*(.+?)\*\*", f"{BOLD}\\1{RESET}", text)


def get_system_prompt():
    custom_prompt = CONFIG.get("system_prompt", "")
    base_prompt = f"""You are a concise coding assistant. Current directory: {os.getcwd()}

Guidelines:
- Read files before editing to understand context
- Make surgical, minimal changes
- Test changes when possible (use bash to run tests)
- Use lint tool to check code quality after changes
- Ask clarifying questions for ambiguous requests
- Use edit for small changes, write for new files or rewrites
- Prefer insert for adding new code sections
- Use search_replace for regex-based transformations
- Check project structure with tree before making assumptions
- Use git for version control operations

{custom_prompt}"""
    return base_prompt


def save_conversation(messages, filename=None):
    if filename is None:
        filename = f"nanocode_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(messages, f, indent=2)
    return filename


def load_conversation(filename):
    with open(filename) as f:
        return json.load(f)


def undo_last():
    if not FILE_HISTORY:
        return "Nothing to undo"
    
    last = FILE_HISTORY.pop()
    with open(last["path"], "w") as f:
        f.write(last["content"])
    
    return f"Reverted {last['path']} to state from {last['time'].strftime('%H:%M:%S')}"


def rollback_transaction():
    """Rollback all file operations in current transaction"""
    if not ACTIVE_TRANSACTION:
        return
    
    print(f"{YELLOW}⏺ Rolling back {len(ACTIVE_TRANSACTION)} operation(s)...{RESET}")
    
    # Reverse order rollback
    for op in reversed(ACTIVE_TRANSACTION):
        if FILE_HISTORY:
            last = FILE_HISTORY.pop()
            with open(last["path"], "w") as f:
                f.write(last["content"])
            invalidate_cache(last["path"])
            print(f"{DIM}  Reverted {last['path']}{RESET}")
    
    ACTIVE_TRANSACTION.clear()
    print(f"{GREEN}⏺ Rollback complete{RESET}")


def estimate_tokens(text):
    """Rough token estimation (4 chars ≈ 1 token)"""
    return len(text) // 4


def trim_conversation(messages, max_tokens=MAX_CONVERSATION_TOKENS):
    """Trim old messages to stay under token limit"""
    total = sum(estimate_tokens(json.dumps(m)) for m in messages)
    
    if total <= max_tokens:
        return messages
    
    print(f"{YELLOW}⏺ Trimming conversation (>{max_tokens // 1000}k tokens)...{RESET}")
    
    # Keep first message (context) and recent messages
    if len(messages) <= 4:
        return messages
    
    # Keep first and last N messages
    keep_recent = 6
    trimmed = [messages[0]] + messages[-keep_recent:]
    
    print(f"{DIM}  Kept {len(trimmed)}/{len(messages)} messages{RESET}")
    return trimmed


def main():
    print(f"{BOLD}nanocode{RESET} | {DIM}{MODEL}{RESET}")
    if CONFIG:
        print(f"{DIM}Config: {', '.join(CONFIG.keys())}{RESET}")
    print(f"{DIM}CWD: {os.getcwd()}{RESET}")
    print(f"{DIM}Commands: /q (quit) /c (clear) /s (save) /l <file> (load) /u (undo) /config (show){RESET}\n")
    
    messages = []
    system_prompt = get_system_prompt()

    while True:
        try:
            print(separator())
            user_input = input(f"{BOLD}{BLUE}❯{RESET} ").strip()
            print(separator())
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input in ("/q", "exit"):
                break
            
            if user_input == "/c":
                messages = []
                FILE_HISTORY.clear()
                FILE_CACHE.clear()
                glob_cached.cache_clear()
                print(f"{GREEN}⏺ Cleared conversation and caches{RESET}")
                continue
            
            if user_input == "/s":
                filename = save_conversation(messages)
                print(f"{GREEN}⏺ Saved to {filename}{RESET}")
                continue
            
            if user_input.startswith("/l "):
                filename = user_input[3:].strip()
                messages = load_conversation(filename)
                print(f"{GREEN}⏺ Loaded {filename} ({len(messages)} messages){RESET}")
                continue
            
            if user_input == "/u":
                result = undo_last()
                print(f"{YELLOW}⏺ {result}{RESET}")
                continue
            
            if user_input == "/config":
                if CONFIG:
                    print(f"{CYAN}Current config:{RESET}")
                    for k, v in CONFIG.items():
                        display_v = v if k != "api_key" else "***"
                        print(f"  {k}: {display_v}")
                else:
                    print(f"{DIM}No config file found (.nanocoderc or ~/.nanocoderc){RESET}")
                    print(f"\n{DIM}Example config:{RESET}")
                    print(json.dumps({
                        "api_url": "https://openrouter.ai/api/v1/chat/completions",
                        "model": "anthropic/claude-opus-4",
                        "api_key": "your-key-here",
                        "system_prompt": "Additional instructions..."
                    }, indent=2))
                continue

            messages.append({"role": "user", "content": user_input})
            
            # Trim conversation if getting too long
            messages = trim_conversation(messages)

            # Agentic loop
            ACTIVE_TRANSACTION.clear()
            try:
                while True:
                    response = call_api(messages, system_prompt)
                    content_blocks = response.get("content", [])
                    tool_results = []

                    for block in content_blocks:
                        if block["type"] == "text":
                            print(f"\n{CYAN}⏺{RESET} {render_markdown(block['text'])}")

                        if block["type"] == "tool_use":
                            tool_name = block["name"]
                            tool_args = block["input"]
                            arg_preview = str(list(tool_args.values())[0])[:50]
                            print(f"\n{GREEN}⏺ {tool_name.capitalize()}{RESET}({DIM}{arg_preview}{RESET})")

                            result = run_tool(tool_name, tool_args)
                            
                            # Check for errors and offer rollback
                            if result.startswith("error:"):
                                print(f"  {RED}⎿  {result}{RESET}")
                                if ACTIVE_TRANSACTION:
                                    rollback_transaction()
                            else:
                                result_lines = result.split("\n")
                                preview = result_lines[0][:60]
                                if len(result_lines) > 1:
                                    preview += f" ... +{len(result_lines) - 1} lines"
                                elif len(result_lines[0]) > 60:
                                    preview += "..."
                                print(f"  {DIM}⎿  {preview}{RESET}")

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block["id"],
                                "content": result,
                            })

                    messages.append({"role": "assistant", "content": content_blocks})

                    if not tool_results:
                        break
                    messages.append({"role": "user", "content": tool_results})
                
                # Clear transaction on success
                ACTIVE_TRANSACTION.clear()
            
            except Exception as e:
                print(f"{RED}⏺ Error: {e}{RESET}")
                if ACTIVE_TRANSACTION:
                    rollback_transaction()
                continue

            print()

        except (KeyboardInterrupt, EOFError):
            if ACTIVE_TRANSACTION:
                print(f"\n{YELLOW}⏺ Interrupted with pending changes{RESET}")
                rollback_transaction()
            break
        except Exception as err:
            print(f"{RED}⏺ Unexpected error: {err}{RESET}")
            if ACTIVE_TRANSACTION:
                rollback_transaction()


if __name__ == "__main__":
    main()