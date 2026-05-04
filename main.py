"""
LM Studio Chat - FastAPI アプリケーション
"""

import httpx
import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from fastapi import FastAPI, Request, Depends, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import config
from rag import RAGManager

# ─── アプリケーション設定 ───────────────────────────────────────────────

app = FastAPI(title="LM Studio Chat")

# テンプレート・静的ファイルの設定
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# チャット履歴の保存先
HISTORY_FILE = BASE_DIR / "chat_history.json"

# ─── RAG 初期化 ─────────────────────────────────────────────────────────

RAG_DOCS_DIR    = BASE_DIR / "rag_docs"
RAG_CONFIG_FILE = BASE_DIR / "rag_config.json"
RAG_DOCS_DIR.mkdir(exist_ok=True)

RAG_CONFIG_DEFAULTS = {"chunk_size": 400, "chunk_overlap": 40}

def load_rag_config() -> dict:
    """rag_config.json を読み込む。存在しない場合はデフォルト値を返す"""
    if RAG_CONFIG_FILE.exists():
        try:
            return {**RAG_CONFIG_DEFAULTS, **json.loads(RAG_CONFIG_FILE.read_text(encoding="utf-8"))}
        except Exception:
            pass
    return dict(RAG_CONFIG_DEFAULTS)

rag_manager: RAGManager | None = None

@app.on_event("startup")
async def startup_event():
    global rag_manager
    try:
        cfg = load_rag_config()
        rag_manager = RAGManager(
            db_path=str(BASE_DIR / "rag_db"),
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
        )
        print(f"[RAG] 初期化完了 - chunk_size={cfg['chunk_size']} overlap={cfg['chunk_overlap']} - {rag_manager.get_status()}")
    except Exception as e:
        print(f"[RAG] 初期化失敗（RAG機能は無効）: {e}")
        rag_manager = None

# ─── モデル定義 ─────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 8196


class LoginRequest(BaseModel):
    username: str
    password: str


class SettingsUpdate(BaseModel):
    lm_studio_host: str
    lm_studio_port: int
    app_username: str
    app_password: str
    lm_studio_api_key: str


# ─── チャット履歴管理 ───────────────────────────────────────────────────


def load_history() -> dict:
    """チャット履歴を読み込む"""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_history(history: dict) -> None:
    """チャット履歴を保存する"""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def get_session_history(session_id: str) -> list[dict]:
    """セッションの履歴を取得"""
    history = load_history()
    return history.get(session_id, [])


def append_to_history(session_id: str, role: str, content: str) -> None:
    """履歴に追加"""
    history = load_history()
    if session_id not in history:
        history[session_id] = []

    history[session_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })

    # 履歴は最新100件に制限
    if len(history[session_id]) > 100:
        history[session_id] = history[session_id][-100:]

    save_history(history)


def clear_session_history(session_id: str) -> None:
    """セッション履歴をクリア"""
    history = load_history()
    if session_id in history:
        del history[session_id]
    save_history(history)


# ─── LM Studio API 通信 ─────────────────────────────────────────────────


async def fetch_models() -> list[dict]:
    """LM Studio から利用可能なモデル一覧を取得"""
    try:
        api_key = config.Config.get_api_key()
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                config.Config.get_models_endpoint(),
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
    except Exception as e:
        print(f"モデル取得エラー: {e}")
        return []


async def send_to_lm_studio(
    messages: list[dict],
    model: str = "",
    temperature: float = 0.7,
    max_tokens: int = 8196
) -> str:
    """LM Studio にチャットリクエストを送信"""
    api_key = config.Config.get_api_key()
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model:
        payload["model"] = model

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                config.Config.get_api_endpoint(),
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except httpx.TimeoutException:
        return "⚠️ タイムアウトしました。LM Studio サーバーが実行中か確認してください。"
    except httpx.HTTPStatusError as e:
        return f"⚠️ HTTPエラー: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"⚠️ エラーが発生しました: {str(e)}"


# ─── 認証ミドルウェア ───────────────────────────────────────────────────


def check_auth(request: Request) -> Optional[str]:
    """セッション認証をチェック"""
    session = request.cookies.get("session_token")
    if not session:
        return None
    return session  # 簡易的にトークン自体をユーザーIDとして使用


# ─── ルート ─────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """ログインページ / チャットページ"""
    session = request.cookies.get("session_token")
    
    if session:
        # ✅ request を context 外に明示的に指定
        return templates.TemplateResponse(
            name="chat.html",
            context={
                "authenticated": True,
                "session_id": session,
                "history": get_session_history(session),
                "models": await fetch_models(),
                "lm_studio_url": config.Config.get_lm_studio_url(),
            },
            request=request
        )
    
    return templates.TemplateResponse(
        name="login.html",
        context={
            "authenticated": False,
            "error": "ユーザー名またはパスワードが異なります。",
        },
        request=request
    )

@app.post("/login")
async def login(request: Request):
    """ログイン処理"""
    form = await request.form()
    username = form.get("username", "")
    password = form.get("password", "")

    if config.Config.is_authenticated(username, password):
        response = HTMLResponse(
            """<script>window.location.href='/';</script>"""
        )
        response.set_cookie(
            key="session_token",
            value=f"user_{username}_{datetime.now().timestamp()}",
            httponly=True,
            max_age=86400,
        )
        return response

    return templates.TemplateResponse("login.html", {
        "request": request,
        "authenticated": False,
        "error": "ユーザー名またはパスワードが異なります。",
    })


@app.get("/logout")
async def logout(request: Request):
    """ログアウト"""
    response = HTMLResponse("""<script>window.location.href='/';</script>""")
    response.delete_cookie(key="session_token")
    return response


@app.get("/api/models")
async def api_models(request: Request):
    """モデル一覧取得 API"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    models = await fetch_models()
    return JSONResponse({"models": models})


@app.post("/api/chat")
async def api_chat(request: Request):
    """チャットリクエスト API（RAG + ツール対応版）"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "")
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 8192)
    tools = body.get("tools", [])
    use_rag = body.get("use_rag", False)

    if not messages:
        return JSONResponse({"error": "メッセージが空です"}, status_code=400)

    # RAG: ユーザーの最後の発言でドキュメント検索してシステムメッセージに注入
    rag_sources = []
    if use_rag and rag_manager:
        user_query = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        if user_query:
            hits = rag_manager.search(user_query, n_results=5)
            if hits:
                context_text = "\n\n---\n\n".join(
                    f"[出典: {h['source']}]\n{h['content']}" for h in hits
                )
                rag_system = {
                    "role": "system",
                    "content": (
                        "以下の参考ドキュメントを使用して回答してください。"
                        "情報が不足する場合は、その旨を伝えてください。\n\n"
                        f"{context_text}"
                    ),
                }
                messages = [rag_system] + messages
                rag_sources = [{"source": h["source"], "score": h["score"]} for h in hits]

    assistant_reply = await chat_with_tools(
        messages=messages,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    append_to_history(session, "user", messages[-1]["content"])
    append_to_history(session, "assistant", assistant_reply)

    return JSONResponse({
        "reply": assistant_reply,
        "history": get_session_history(session),
        "rag_sources": rag_sources,
    })


# ─── MCP ツール実行 ──────────────────────────────────────────────────────────

async def call_mcp_tool(tool_name: str, tool_args: dict) -> str:
    """
    stdio MCP サーバーを起動してツールを呼び出し、結果を文字列で返す。
    呼び出しのたびに新しいプロセスを起動する（ステートレス）。
    """
    try:
        async with stdio_client(config.Config.get_mcp_server_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, tool_args)
                # result.content はリスト。TextContent を結合して返す
                parts = []
                for block in result.content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                    else:
                        parts.append(str(block))
                return "\n".join(parts)
    except Exception as e:
        return json.dumps({"error": f"MCPツール呼び出しエラー: {str(e)}"}, ensure_ascii=False)


async def chat_with_tools(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    max_tool_iterations: int = 5
) -> str:
    """
    ツール呼び出しに対応したチャット処理。
    tool_calls が返ってきたら MCP サーバーを実際に呼び出して
    tool_result を組み立て、LM Studio に再送して最終応答を得る。
    """
    current_messages = list(messages)
    tool_definitions = list(tools) if tools else []

    api_key = config.Config.get_api_key()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for iteration in range(max_tool_iterations):
        payload = {
            "messages": current_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tool_definitions:
            payload["tools"] = tool_definitions
            payload["tool_choice"] = "auto"

        print(f"[chat_with_tools] iteration={iteration}, messages={len(current_messages)}, tools={len(tool_definitions)}")
        print(f"[chat_with_tools] payload={json.dumps(payload, ensure_ascii=False)[:600]}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    config.Config.get_api_endpoint(),
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            return f"⚠️ HTTPエラー: {e.response.status_code} - {e.response.text}"
        except Exception as e:
            return f"⚠️ API通信エラー: {str(e)}"

        print(f"[chat_with_tools] response={json.dumps(data, ensure_ascii=False)[:600]}")

        choice = data["choices"][0]
        finish_reason = choice.get("finish_reason", "")
        assistant_msg = choice["message"]
        tool_calls = assistant_msg.get("tool_calls") or []

        # ── ツール呼び出しなし → 最終応答 ─────────────────────────────
        if not tool_calls:
            content = assistant_msg.get("content")
            return content if content is not None else ""

        # ── ツール呼び出しあり → MCP サーバーを実際に呼ぶ ─────────────
        # assistant メッセージを履歴に追加
        current_messages.append({
            "role": "assistant",
            "content": assistant_msg.get("content"),
            "tool_calls": tool_calls,
        })

        # 各 tool_call を MCP で実行し tool_result を追加
        for call in tool_calls:
            func_name = call["function"]["name"]
            raw_args = call["function"].get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    func_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    func_args = {}
            else:
                func_args = raw_args

            # None 文字列を実際の None に変換（モデルが "None" を渡してくる場合の対策）
            sanitized_args = {
                k: (None if v in (None, "None", "null", "") else v)
                for k, v in func_args.items()
            }

            print(f"[chat_with_tools] calling MCP tool: {func_name}({sanitized_args})")
            result_content = await call_mcp_tool(func_name, sanitized_args)
            print(f"[chat_with_tools] MCP result: {result_content[:300]}")

            current_messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": result_content,
            })

    return "⚠️ ツール呼び出しの最大反復回数を超えました。"

@app.get("/api/history/{session_id}")
async def api_get_history(session_id: str, request: Request):
    """履歴取得 API"""
    _ = check_auth(request)  # 認証チェック（トークンベース）
    return JSONResponse({"history": get_session_history(session_id)})


@app.post("/api/clear-history")
async def api_clear_history(request: Request):
    """履歴クリア API"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    clear_session_history(session)
    return JSONResponse({"status": "cleared"})


@app.get("/api/settings")
async def api_get_settings(request: Request):
    """設定取得 API"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    return JSONResponse({
        "lm_studio_host": os.getenv("LM_STUDIO_HOST", "127.0.0.1"),
        "lm_studio_port": os.getenv("LM_STUDIO_PORT", "1234"),
        "has_api_key": bool(os.getenv("LM_STUDIO_API_KEY", "")),
        "app_username": os.getenv("APP_USERNAME", "admin"),
    })


@app.post("/api/update-settings")
async def api_update_settings(request: Request):
    """設定更新 API（変更キーのみ上書き、未知のキーは保持）"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    body = await request.json()
    env_path = BASE_DIR / ".env"

    # リクエストから更新するキーと値を組み立てる（空文字は除外しない）
    updates: dict[str, str] = {
        "LM_STUDIO_HOST": str(body.get("lm_studio_host", "")),
        "LM_STUDIO_PORT": str(body.get("lm_studio_port", "")),
        "LM_STUDIO_API_KEY": str(body.get("lm_studio_api_key", "")),
        "APP_USERNAME": str(body.get("app_username", "")),
    }
    # パスワードは空送信の場合は更新しない
    if body.get("app_password"):
        updates["APP_PASSWORD"] = str(body["app_password"])

    # 既存ファイルを行単位で読み込み、該当キーだけ置換する
    existing_lines: list[str] = []
    if env_path.exists():
        existing_lines = env_path.read_text(encoding="utf-8").splitlines()

    updated_keys: set[str] = set()
    new_lines: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        # コメント・空行はそのまま保持
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue
        # KEY=VALUE 形式のみ処理
        if "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                updated_keys.add(key)
                continue
        new_lines.append(line)

    # ファイルに存在しなかったキーは末尾に追記
    for key, value in updates.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    load_dotenv(env_path, override=True)

    return JSONResponse({"status": "updated", "message": "設定を更新しました。サーバーを再起動してください。"})


# ─── Function Calling ツール設定の保存・読み込み ─────────────────────────

TOOLS_CONFIG_FILE = BASE_DIR / "tools_config.json"

@app.get("/api/tools")
async def api_get_tools(request: Request):
    """保存済みツール設定を取得"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    if TOOLS_CONFIG_FILE.exists():
        tools = json.loads(TOOLS_CONFIG_FILE.read_text(encoding="utf-8"))
    else:
        tools = []
    return JSONResponse({"tools": tools})


@app.post("/api/tools")
async def api_save_tools(request: Request):
    """ツール設定を保存"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    body = await request.json()
    tools = body.get("tools", [])
    TOOLS_CONFIG_FILE.write_text(
        json.dumps(tools, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return JSONResponse({"status": "saved", "count": len(tools)})


# ─── RAG エンドポイント ──────────────────────────────────────────────────

@app.get("/api/rag/config")
async def rag_get_config(request: Request):
    """RAG チャンク設定を取得"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    return JSONResponse(load_rag_config())


@app.post("/api/rag/config")
async def rag_save_config(request: Request):
    """RAG チャンク設定を保存し、RAGManager を再初期化する"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    global rag_manager
    body = await request.json()

    chunk_size    = int(body.get("chunk_size",    RAG_CONFIG_DEFAULTS["chunk_size"]))
    chunk_overlap = int(body.get("chunk_overlap", RAG_CONFIG_DEFAULTS["chunk_overlap"]))

    if chunk_overlap >= chunk_size:
        return JSONResponse({"error": "chunk_overlap は chunk_size より小さくしてください"}, status_code=400)

    cfg = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
    RAG_CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # RAGManager を新しい設定で再初期化（既存 DB はそのまま保持）
    try:
        rag_manager = RAGManager(
            db_path=str(BASE_DIR / "rag_db"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except Exception as e:
        return JSONResponse({"error": f"RAG再初期化失敗: {e}"}, status_code=500)

    return JSONResponse({
        "status": "saved",
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "note": "新しい設定は次回インデックス追加時から反映されます。既存ドキュメントを再インデックスするとすべての設定が適用されます。",
    })


@app.get("/api/rag/status")
async def rag_status(request: Request):
    """インデックスの状態確認"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not rag_manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)
    return JSONResponse(rag_manager.get_status())


@app.post("/api/rag/upload")
async def rag_upload(request: Request, file: UploadFile = File(...)):
    """ファイルをアップロードしてインデックスに追加"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not rag_manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".txt", ".pdf"}:
        return JSONResponse({"error": f"未対応の形式: {suffix}"}, status_code=400)

    dest = RAG_DOCS_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = rag_manager.add_document(dest)
    return JSONResponse(result)


@app.post("/api/rag/index-dir")
async def rag_index_dir(request: Request):
    """rag_docs/ ディレクトリ内の全ファイルを再インデックス"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not rag_manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)

    results = rag_manager.add_directory(RAG_DOCS_DIR)
    success = sum(1 for r in results if r.get("success"))
    return JSONResponse({
        "total": len(results),
        "success": success,
        "failed": len(results) - success,
        "details": results,
    })


@app.delete("/api/rag/document/{file_name}")
async def rag_delete_document(file_name: str, request: Request):
    """指定ファイルをインデックスから削除"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not rag_manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)

    result = rag_manager.delete_document(file_name)
    return JSONResponse(result)


@app.delete("/api/rag/clear")
async def rag_clear(request: Request):
    """インデックスを全クリア"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not rag_manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)

    result = rag_manager.clear()
    return JSONResponse(result)


@app.post("/api/rag/search")
async def rag_search(request: Request):
    """RAG検索のテスト用エンドポイント"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not rag_manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)

    body = await request.json()
    query = body.get("query", "")
    n_results = body.get("n_results", 5)
    if not query:
        return JSONResponse({"error": "queryが空です"}, status_code=400)

    hits = rag_manager.search(query, n_results=n_results)
    return JSONResponse({"results": hits})


# ─── メイン ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.Config.get_app_host(),
        port=config.Config.get_app_port(),
        reload=True,
    )
