# main.py — LINE Bot (FastAPI) with Sympy CAS & robust calc
import os, hmac, hashlib, base64, json, re, logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse, Response
import httpx

# ===== Sympy（高度計算） =====
SYM_AVAILABLE = True
try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations,
        implicit_multiplication_application,
        convert_xor, function_exponentiation
    )
except Exception:
    SYM_AVAILABLE = False
logging.info(f"Sympy available: {SYM_AVAILABLE}")

# ====== 基本設定 ======
app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
LINE_CONTENT_URL = "https://api-data.line.me/v2/bot/message/{messageId}/content"  # ← 画像は api-data

# 角度モード（プロセス内共有）
ANGLE_MODE = {"mode": "deg"}  # "deg" or "rad"

# ===== 共通 =====
async def reply_message(reply_token: str, messages: List[Dict[str, Any]]) -> None:
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"replyToken": reply_token, "messages": messages}
    async with httpx.AsyncClient(timeout=30) as ac:
        r = await ac.post(LINE_REPLY_URL, headers=headers, json=payload)
        logging.info(f'HTTP Request: POST {LINE_REPLY_URL} "{r.http_version} {r.status_code} {r.reason_phrase}"')
        r.raise_for_status()

async def get_line_image_bytes(message_id: str) -> bytes:
    url = LINE_CONTENT_URL.format(messageId=message_id)
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    async with httpx.AsyncClient(timeout=30) as ac:
        r = await ac.get(url, headers=headers)
        logging.info(f'GET {url} "{r.status_code}"')
        r.raise_for_status()
        return r.content

def verify_signature(channel_secret: str, body: bytes, signature: str) -> bool:
    mac = hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, signature or "")

# ===== 入力正規化（全角→半角、°、√、×÷、i/j など） =====
_ZK = "０１２３４５６７８９（）＊＋－／＾，．　ｉｊｘ"
_HK = "0123456789()*/+/-^,. ijx"
TRANS = str.maketrans(_ZK, _HK)

def normalize_expr(s: str) -> str:
    s0 = s
    s = s.translate(TRANS)
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-").replace("–", "-")
    s = s.replace("π", "pi").replace("ｅ", "e").replace("Ｅ", "e")
    # sqrt
    s = re.sub(r"√\s*([0-9a-zA-Z_\(])", r"sqrt(\1", s)   # √2 → sqrt(2
    # 足りない ) は sympy が補えないのでそのまま（多くはOK）
    # 度記号：30° → (30 deg)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*°", r"(\1 deg)", s)
    # sin30, cos45° など（括弧省略）→ sin(30), sin(30 deg)
    s = re.sub(r"\b(sin|cos|tan|asin|acos|atan|sinh|cosh|tanh)\s*([0-9πpieij\.]+(?:\s*deg)?)",
               r"\1(\2)", s, flags=re.IGNORECASE)
    # 余計なスペース削除
    s = re.sub(r"\s+", "", s)
    # ^ は後で convert_xor でも対応するが、明示で
    s = s.replace("^", "**")
    # 虚数単位 i/j
    s = re.sub(r"\b([0-9\.]+)i\b", r"\1*I", s, flags=re.IGNORECASE)
    s = re.sub(r"\b([0-9\.]+)j\b", r"\1*I", s, flags=re.IGNORECASE)
    s = re.sub(r"\bi\b", "I", s, flags=re.IGNORECASE)
    s = re.sub(r"\bj\b", "I", s, flags=re.IGNORECASE)
    return s

# ===== Sympy評価系 =====
if SYM_AVAILABLE:
    # 角度→ラジアン変換ヘルパ
    def _deg(x):  # 数値/式 → ラジアンへ
        return x * sp.pi / 180

    # nCr / nPr
    def nCr(n, r): return sp.binomial(n, r)
    def nPr(n, r): return sp.factorial(n) / sp.factorial(n - r)

    # 角度モードに応じた trig ラッパ
    def _wrap_trig(func):
        def f(x):
            if ANGLE_MODE["mode"] == "deg":
                return func(_deg(x))
            return func(x)
        return f

    # sympify用ローカル辞書
    SYM_LOCALS = {
        # 基本
        "pi": sp.pi, "e": sp.E, "I": sp.I,
        "abs": sp.Abs, "sqrt": sp.sqrt, "exp": sp.exp,
        "log": sp.log, "log10": lambda x: sp.log(x, 10),
        "floor": sp.floor, "ceil": sp.ceiling,
        # 組合せ
        "nCr": nCr, "C": nCr, "comb": nCr,
        "nPr": nPr, "P": nPr, "perm": nPr,
        # 三角（モード対応）
        "sin": _wrap_trig(sp.sin), "cos": _wrap_trig(sp.cos), "tan": _wrap_trig(sp.tan),
        "asin": lambda x: sp.asin(x) if ANGLE_MODE["mode"]=="rad" else sp.deg(sp.asin(x)),
        "acos": lambda x: sp.acos(x) if ANGLE_MODE["mode"]=="rad" else sp.deg(sp.acos(x)),
        "atan": lambda x: sp.atan(x) if ANGLE_MODE["mode"]=="rad" else sp.deg(sp.atan(x)),
        "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
        # 単位: deg（数値/式に付けたときラジアン化）
        "deg": lambda x: _deg(x),
        # Matrix
        "Matrix": sp.Matrix,
    }

    TRANSFORMS = (
        standard_transformations
        + (implicit_multiplication_application,)   # 2pi, 3x, 2(x+1)
        + (convert_xor,)                           # ^ をべき
        + (function_exponentiation,)               # sin^2 x → (sin(x))**2 等
    )

    def sym_parse(expr: str):
        return parse_expr(expr, local_dict=SYM_LOCALS, transformations=TRANSFORMS, evaluate=True)

    def sym_eval_numeric(expr: str):
        e = sym_parse(expr)
        # 数値化（複素もOK）
        return sp.N(e)

# ===== fx-CG50 キーガイド（ベーシック部のみ自動生成） =====
def cg50_keyseq(expr_show: str) -> str:
    s = expr_show
    s = s.replace("**", "^").replace("*", "×").replace("/", "÷")
    s = (s.replace("asin", "[SHIFT][SIN]^-1")
           .replace("acos", "[SHIFT][COS]^-1")
           .replace("atan", "[SHIFT][TAN]^-1")
           .replace("sin", "[SIN]").replace("cos", "[COS]").replace("tan", "[TAN]")
           .replace("sqrt", "[√]").replace("log10", "[LOG]10,").replace("log", "[LN]"))
    return "角度: " + ("Deg" if ANGLE_MODE["mode"]=="deg" else "Rad") + " を確認 → 入力: " + s + " → [EXE]"

# ===== ルーティング =====
@app.get("/")
async def root():
    return {"ok": True, "message": "LINE bot (FastAPI) running", "sympy": SYM_AVAILABLE}

@app.get("/botinfo")
async def botinfo():
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    async with httpx.AsyncClient(timeout=10) as ac:
        r = await ac.get("https://api.line.me/v2/bot/info", headers=headers)
    return Response(r.text, media_type="application/json", status_code=r.status_code)

@app.post("/webhook")
async def webhook(request: Request, x_line_signature: Optional[str] = Header(default=None)):
    body_bytes = await request.body()
    if LINE_CHANNEL_SECRET and not verify_signature(LINE_CHANNEL_SECRET, body_bytes, x_line_signature or ""):
        logging.error("Signature verify failed")
        return JSONResponse({"message": "signature error"}, status_code=400)

    logging.info('POST /webhook "HTTP/1.1 200 OK"')
    body = json.loads(body_bytes.decode("utf-8"))
    events = body.get("events", [])

    for event in events:
        if event.get("type") != "message":
            continue
        reply_token = event.get("replyToken")
        m = event.get("message", {})
        msg_type = m.get("type")
        logging.info(f'message.id = {m.get("id")} type={msg_type}')

        try:
            if msg_type == "text":
                text = (m.get("text") or "").strip()

                # === モード切替 ===
                if text.lower().startswith("mode:"):
                    v = text.split(":",1)[1].strip().lower()
                    if v in ("deg","degree","degrees"):
                        ANGLE_MODE["mode"]="deg"
                        await reply_message(reply_token,[{"type":"text","text":"角度モードを Deg に設定しました"}])
                    elif v in ("rad","radian","radians"):
                        ANGLE_MODE["mode"]="rad"
                        await reply_message(reply_token,[{"type":"text","text":"角度モードを Rad に設定しました"}])
                    else:
                        await reply_message(reply_token,[{"type":"text","text":"mode:deg または mode:rad を指定してください"}])
                    continue

                # === calc ===
                if text.lower().startswith("calc:"):
                    raw = text[5:].strip()
                    if not raw:
                        await reply_message(reply_token,[{"type":"text","text":"式が空です。例: calc: sin30° + 3^2"}])
                        continue
                    expr_in = normalize_expr(raw)
                    if not SYM_AVAILABLE:
                        msg = ("Sympy が未導入のため高度計算は無効です。\n"
                               "requirements.txt に `fastapi\nuvicorn\nhttpx\nsympy` を入れて再デプロイしてください。\n"
                               f"expr: {expr_in}")
                        await reply_message(reply_token,[{"type":"text","text":msg}])
                        continue
                    try:
                        val = sym_eval_numeric(expr_in)
                        shown = str(sp.simplify(sym_parse(expr_in))).replace("**","^")
                        guide = cg50_keyseq(shown)
                        msg = f"計算OK ✅\n式: {shown}\n結果: {val}\n\nfx-CG50操作ガイド:\n{guide}"
                    except Exception as e:
                        msg = ("式の解析/評価に失敗しました ❌\n"
                               "例: calc: sin30° + 3^2,  calc: (2+3i)^2,  calc: nCr(10,3)\n"
                               f"詳細: {e}")
                    await reply_message(reply_token,[{"type":"text","text":msg}])
                    continue

                # === solve ===
                if text.lower().startswith("solve:"):
                    if not SYM_AVAILABLE:
                        await reply_message(reply_token,[{"type":"text","text":"Sympy未導入のため solve は使えません。"}]); continue
                    raw = text.split(":",1)[1]
                    if "," not in raw:
                        await reply_message(reply_token,[{"type":"text","text":"書式: solve: 方程式 , 変数\n例: solve: x^3-8=0 , x"}]); continue
                    left, var = raw.split(",",1)
                    expr = normalize_expr(left)
                    v = normalize_expr(var)
                    try:
                        # x をシンボル登録
                        sym = sp.symbols(v)
                        if "=" in expr:
                            L,R = expr.split("=",1)
                            sol = sp.solve(sp.Eq(sym_parse(L), sym_parse(R)), sym, dict=True)
                        else:
                            sol = sp.solve(sym_parse(expr), sym, dict=True)
                        await reply_message(reply_token,[{"type":"text","text":f"解: {sol}"}])
                    except Exception as e:
                        await reply_message(reply_token,[{"type":"text","text":f"solve 失敗 ❌ 詳細: {e}"}])
                    continue

                # === diff ===
                if text.lower().startswith("diff:"):
                    if not SYM_AVAILABLE:
                        await reply_message(reply_token,[{"type":"text","text":"Sympy未導入のため diff は使えません。"}]); continue
                    raw = text.split(":",1)[1]
                    if "," not in raw:
                        await reply_message(reply_token,[{"type":"text","text":"書式: diff: 式 , 変数"}]); continue
                    f, v = [normalize_expr(x) for x in raw.split(",",1)]
                    try:
                        var = sp.symbols(v)
                        res = sp.diff(sym_parse(f), var)
                        await reply_message(reply_token,[{"type":"text","text":f"d/d{v} = {sp.simplify(res)}"}])
                    except Exception as e:
                        await reply_message(reply_token,[{"type":"text","text":f"diff 失敗 ❌ 詳細: {e}"}])
                    continue

                # === int ===
                if text.lower().startswith("int:"):
                    if not SYM_AVAILABLE:
                        await reply_message(reply_token,[{"type":"text","text":"Sympy未導入のため int は使えません。"}]); continue
                    raw = text.split(":",1)[1]
                    if "," not in raw:
                        await reply_message(reply_token,[{"type":"text","text":"書式: int: 式 , 変数"}]); continue
                    f, v = [normalize_expr(x) for x in raw.split(",",1)]
                    try:
                        var = sp.symbols(v)
                        res = sp.integrate(sym_parse(f), var)
                        await reply_message(reply_token,[{"type":"text","text":f"∫ {f} d{v} = {res} + C"}])
                    except Exception as e:
                        await reply_message(reply_token,[{"type":"text","text":f"int 失敗 ❌ 詳細: {e}"}])
                    continue

                # === factor / expand ===
                if text.lower().startswith("factor:"):
                    if not SYM_AVAILABLE:
                        await reply_message(reply_token,[{"type":"text","text":"Sympy未導入のため factor は使えません。"}]); continue
                    expr = normalize_expr(text.split(":",1)[1])
                    try:
                        res = sp.factor(sym_parse(expr))
                        await reply_message(reply_token,[{"type":"text","text":f"factor: {res}"}])
                    except Exception as e:
                        await reply_message(reply_token,[{"type":"text","text":f"factor 失敗 ❌ 詳細: {e}"}])
                    continue

                if text.lower().startswith("expand:"):
                    if not SYM_AVAILABLE:
                        await reply_message(reply_token,[{"type":"text","text":"Sympy未導入のため expand は使えません。"}]); continue
                    expr = normalize_expr(text.split(":",1)[1])
                    try:
                        res = sp.expand(sym_parse(expr))
                        await reply_message(reply_token,[{"type":"text","text":f"expand: {res}"}])
                    except Exception as e:
                        await reply_message(reply_token,[{"type":"text","text":f"expand 失敗 ❌ 詳細: {e}"}])
                    continue

                # === matrix（または calc に書いてもOK） ===
                if text.lower().startswith("matrix:"):
                    if not SYM_AVAILABLE:
                        await reply_message(reply_token,[{"type":"text","text":"Sympy未導入のため matrix は使えません。"}]); continue
                    expr = normalize_expr(text.split(":",1)[1])
                    try:
                        val = sym_eval_numeric(expr)
                        await reply_message(reply_token,[{"type":"text","text":f"matrix: {val}"}])
                    except Exception as e:
                        await reply_message(reply_token,[{"type":"text","text":f"matrix 失敗 ❌ 詳細: {e}"}])
                    continue

                # それ以外
                await reply_message(reply_token,[{"type":"text","text":
                    ("受信しました。\n"
                     "主なコマンド: mode:deg/rad, calc:, solve:, diff:, int:, factor:, expand:, matrix:\n"
                     "例: calc: sin30° + 3^2   /  solve: x^3-8=0 , x")}])

            elif msg_type == "image":
                cp = m.get("contentProvider") or {}
                if cp.get("type") == "external" and cp.get("originalContentUrl"):
                    async with httpx.AsyncClient(timeout=30) as ac:
                        r = await ac.get(cp["originalContentUrl"]); r.raise_for_status()
                        img_bytes = r.content
                    logging.info(f"Downloaded external image bytes: {len(img_bytes)}")
                else:
                    img_bytes = await get_line_image_bytes(m.get("id"))
                    logging.info(f"Downloaded image bytes: {len(img_bytes)}")

                # 画像→自動解法は誤答防止のため、いったん停止
                guide = (
                    "📷 画像を受け取りました。\n"
                    "現在は誤答防止のため、画像からの自動解法を停止中です。\n"
                    "テキストで指示してください：\n"
                    "例) calc: sin30° + 3^2 / solve: x^3-8=0 , x"
                )
                await reply_message(reply_token,[{"type":"text","text":guide}])

            else:
                await reply_message(reply_token,[{"type":"text","text":f"未対応メッセージタイプ: {msg_type}"}])

        except httpx.HTTPStatusError as he:
            await reply_message(reply_token,[{"type":"text","text":f"HTTPエラー: {he.response.status_code}"}])
            logging.exception("HTTPStatusError")
        except Exception:
            await reply_message(reply_token,[{"type":"text","text":"内部エラーが発生しました。"}])
            logging.exception("Unhandled error")

    return JSONResponse({"status":"ok"})

