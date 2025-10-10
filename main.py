import os, json, base64, hmac, hashlib, logging
from typing import Any, Dict, List
from fastapi import FastAPI, Request, BackgroundTasks, Response
import httpx

# ====== 基本設定 ======
app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("uvicorn.error")

LINE_CHANNEL_SECRET       = os.environ.get("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
OPENAI_API_KEY            = os.environ.get("OPENAI_API_KEY", "")

LINE_REPLY_URL   = "https://api.line.me/v2/bot/message/reply"
LINE_CONTENT_URL = "https://api-data.line.me/v2/bot/message/{messageId}/content"  # ← api-data が正解

# ====== 署名検証 ======
def verify_signature(body: bytes, signature: str) -> bool:
    if not LINE_CHANNEL_SECRET:
        log.error("ENV LINE_CHANNEL_SECRET is empty")
        return False
    mac = hmac.new(LINE_CHANNEL_SECRET.encode(), body, hashlib.sha256).digest()
    expect = base64.b64encode(mac).decode()
    ok = hmac.compare_digest(expect, signature or "")
    if not ok:
        log.error("Signature NG")
    return ok

# ====== LINE 返信 ======
async def line_reply(reply_token: str, messages: List[Dict[str, Any]]) -> None:
    if not LINE_CHANNEL_ACCESS_TOKEN:
        log.error("ENV LINE_CHANNEL_ACCESS_TOKEN is empty")
        return
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    body = {"replyToken": reply_token, "messages": messages[:5]}
    async with httpx.AsyncClient(timeout=20) as ac:
        r = await ac.post(LINE_REPLY_URL, headers=headers, json=body)
        r.raise_for_status()

# ====== 画像/動画/音声（LINEサーバ保持）のバイト取得 ======
async def download_line_content(message_id: str) -> bytes:
    url = LINE_CONTENT_URL.format(messageId=message_id)
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()      # 401/404 はここで例外
        return r.content

# ====== fx-CG50 の定型手順（問題文から推定して付け足す） ======
def cg50_steps_for_text(text: str) -> str:
    t = text.replace(" ", "")
    if ("y=-x^2+4ax+b" in t) or ("放物線" in t and "4ax+b" in t):
        return (
            "【fx-CG50 操作】\n"
            "1) MENU→GRAPH→EXE（FUNC）\n"
            "2) Y1 を選んで F1[SELECT] で＝を濃く\n"
            "3) Y1 に次を入力→EXE：\n"
            "   [(-)] [X,θ,T] [x²] [+] 4 [×] [ALPHA][log](A) [×] [X,θ,T] [+] [ALPHA][ln](B)\n"
            "   ※ [ALPHA][log]＝A、[ALPHA][ln]＝B\n"
            "4) 代入：MENU→RUN-MAT → 0.5 [SHIFT][RCL]→[ALPHA][log](A)→EXE／4 [SHIFT][RCL]→[ALPHA][ln](B)→EXE\n"
            "5) 戻って F6[DRAW]、頂点は SHIFT+F5[G-Solv]→MAX\n"
        )
    if ("勝率" in t or "確率" in t) and ("1/3" in t or "１/３" in t or "1÷3" in t):
        return (
            "【fx-CG50（RUN-MAT）】\n"
            "a) 3戦全勝： ( 1 ÷ 3 ) [SHIFT][^] 3 → EXE\n"
            "b) 5戦3勝2敗： 5C3×(1/3)^3×(2/3)^2 を計算\n"
            "   10 × ( 1 ÷ 3 ) [SHIFT][^] 3 × ( 2 ÷ 3 ) [SHIFT][^] 2 → EXE\n"
        )
    return (
        "【fx-CG50 基本】MENU→GRAPH（関数）／EQUA（方程式）／RUN-MAT（数値）\n"
        "式は Y1→EXE→F6[DRAW]、読取は SHIFT+F5[G-Solv]。係数は A,B にして STO▶ 代入→再描画。\n"
    )

# ====== OpenAI Vision で読んで解く（任意） ======
async def solve_from_image(img_bytes: bytes) -> str:
    if not OPENAI_API_KEY:
        return "（サーバ設定：OPENAI_API_KEY 未設定）"
    b64 = base64.b64encode(img_bytes).decode()
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content":
             "日本語で、最大2問。次の書式で：\n【問題】…\n【答え】…\n【考え方】…\n【電卓手順】fx-CG50のキー列（[(-)] [X,θ,T] [x²] …、EXEの位置も）。"},
            {"role": "user", "content": [
                {"type": "text", "text": "この画像の数学問題を読み取り、上の形式で日本語で出力してください。"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    async with httpx.AsyncClient(timeout=60) as ac:
        r = await ac.post("https://api.openai.com/v1/chat/completions",
                          headers=headers, json=payload)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
    return text + "\n\n" + cg50_steps_for_text(text)

# ====== 裏で走る本処理 ======
async def process_line_events(body_bytes: bytes, signature: str):
    try:
        # 署名検証（必要なら off にして検証の通りだけ見てもOK）
        if not verify_signature(body_bytes, signature):
            return
        body = json.loads(body_bytes.decode("utf-8"))
        for ev in body.get("events", []):
            if ev.get("type") != "message":
                continue
            msg         = ev["message"]
            reply_token = ev["replyToken"]
            mtype       = msg.get("type")
            # ---- 画像 ----
            if mtype == "image":
                log.info("message.id=%s", msg.get("id"))
                cp = msg.get("contentProvider", {"type": "line"})
                if cp.get("type") == "line":
                    data = await download_line_content(msg["id"])
                else:
                    # 外部URLの場合
                    async with httpx.AsyncClient(timeout=30) as ac:
                        r = await ac.get(cp["originalContentUrl"])
                        r.raise_for_status()
                        data = r.content
                # ここで Vision へ（重いなら省略可）
                try:
                    answer = await solve_from_image(data)
                except Exception as e:
                    log.exception("Vision failed")
                    answer = f"画像OK: {len(data)} bytes 取得。message.id={msg['id']}\n（解析に失敗: {e}）"
                await line_reply(reply_token, [{"type": "text", "text": answer[:4900]}])
            # ---- テキスト ----
            elif mtype == "text":
                txt   = msg.get("text", "")
                extra = cg50_steps_for_text(txt)
                await line_reply(reply_token, [{"type": "text", "text": f"受信：{txt}\n\n{extra}"}])
            else:
                await line_reply(reply_token, [{"type": "text", "text": f"{mtype} は未対応です。"}])
    except Exception:
        log.exception("process_line_events error")
# ===== 安全計算ユーティリティ =====
import ast, math, re
from typing import Any

ALLOWED_FUNCS = {
    "sin": lambda x: math.sin(math.radians(x)),
    "cos": lambda x: math.cos(math.radians(x)),
    "tan": lambda x: math.tan(math.radians(x)),
    "asin": lambda x: math.degrees(math.asin(x)),
    "acos": lambda x: math.degrees(math.acos(x)),
    "atan": lambda x: math.degrees(math.atan(x)),
    "sqrt": math.sqrt,
    "log": math.log,      # 自然対数
    "log10": math.log10,
    "abs": abs
}
ALLOWED_NAMES = {"pi": math.pi, "e": math.e}

class SafeEval(ast.NodeVisitor):
    def visit(self, node):
        if isinstance(node, ast.Expression): return self.visit(node.body)
        if isinstance(node, ast.Num): return node.n  # py<3.8
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return node.value
        if isinstance(node, ast.BinOp):
            l, r = self.visit(node.left), self.visit(node.right)
            if isinstance(node.op, ast.Add): return l + r
            if isinstance(node.op, ast.Sub): return l - r
            if isinstance(node.op, ast.Mult): return l * r
            if isinstance(node.op, ast.Div): return l / r
            if isinstance(node.op, ast.Pow): return l ** r
            if isinstance(node.op, ast.Mod): return l % r
            raise ValueError("operator not allowed")
        if isinstance(node, ast.UnaryOp):
            v = self.visit(node.operand)
            if isinstance(node.op, ast.UAdd): return +v
            if isinstance(node.op, ast.USub): return -v
            raise ValueError("unary op not allowed")
        if isinstance(node, ast.Name):
            if node.id in ALLOWED_NAMES: return ALLOWED_NAMES[node.id]
            raise ValueError("name not allowed")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name): raise ValueError("call not allowed")
            fname = node.func.id
            if fname not in ALLOWED_FUNCS: raise ValueError(f"func {fname} not allowed")
            args = [self.visit(a) for a in node.args]
            return ALLOWED_FUNCS[fname](*args)
        raise ValueError("node not allowed")

def safe_calc(expr: str) -> float:
    # 角度記号「°」に対応: 例) 30° → sin(30)等は ALLOWED_FUNCS側で度→ラジアン変換済み
    # ただし「30°」単体は radに直すとややこしいので、ユーザーには sin(30°) など関数内に入れてもらう前提。
    tree = ast.parse(expr, mode="eval")
    return SafeEval().visit(tree)

def cg50_keyseq(expr: str) -> str:
    """
    超簡易：fx-CG50 キー案内（代表的なものだけ）
    - sin(30) → [SHIFT][SETUP]で角度Deg確認 → [SIN] 3 0 [)]
    - 乗算/除算/累乗 → × / ÷ / ^ キー想定
    ※ 実機での微調整は適宜。ここではガイド用の擬似列を返す。
    """
    seq = expr
    seq = re.sub(r"\s+", "", seq)
    seq = seq.replace("sin", "[SIN]").replace("cos", "[COS]").replace("tan", "[TAN]")
    seq = seq.replace("sqrt", "[√]").replace("log10", "[LOG]10,").replace("log", "[LN]")
    seq = seq.replace("*", "×").replace("/", "÷").replace("**", "^")
    return "角度:Deg を確認 → 入力: " + seq + " → [EXE]"
# ===== webhook内（イベントループ）のテキスト/画像分岐 =====
elif msg_type == "text":
    text = event["message"]["text"].strip()
    if text.lower() == "ping":
        reply_text = "pong ✅"
    elif text.lower().startswith("calc:"):
        expr = text[5:].strip()
        try:
            val = safe_calc(expr)
            seq = cg50_keyseq(expr)
            reply_text = f"計算OK ✅\n式: {expr}\n結果: {val}\n\nfx-CG50操作ガイド:\n{seq}"
        except Exception as e:
            reply_text = f"式の解析に失敗しました ❌\n入力例: calc: sin(30)+3^2\n詳細: {e}"
    else:
        reply_text = "受信しました。計算は `calc: ...` で送ってね。例: `calc: sin(30)+3^2`"
    await reply_message(reply_token, [{"type":"text","text":reply_text}])
    return JSONResponse({"status":"ok"})

elif msg_type == "image":
    # いったん“誤答を出さない”安全運用に固定
    reply_text = (
        "画像ありがとう！📷\n"
        "まずはテキストで式を送ってね（例）\n"
        "calc: sin(30)+3^2\n"
        "※今は誤答防止のため、画像からの自動読取は一時停止中です。"
    )
    await reply_message(reply_token, [{"type":"text","text":reply_text}])
    return JSONResponse({"status":"ok"})

# ====== ルーティング ======
@app.get("/")
def health():
    return {"ok": True}

# Webhook：即200を返し、実処理はバックグラウンド
@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()
    sig  = request.headers.get("x-line-signature", "")
    background_tasks.add_task(process_line_events, body, sig)
    return Response(status_code=200)


