# main.py  — FastAPI only
import os, hmac, hashlib, base64, json, re, unicodedata, logging
from typing import Dict, Any, Tuple
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
import httpx

# ==== Calculator (sympy) ====
from sympy import sin, cos, tan, sqrt, pi, E, simplify
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    function_exponentiation,
    convert_xor,
)

TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,  # 2x, 2(x), (a)b
    function_exponentiation,              # sin^2 x
    convert_xor,                          # ^ → **
)

# ========= App =========
app = FastAPI()
logger = logging.getLogger("uvicorn.error")

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")

# 共有ステート（簡易）
ANGLE_MODE = "deg"          # "deg" / "rad"
PREC_DIGITS = 6             # 近似の桁
LAST_ERROR: Dict[str, Any] = {"msg": None, "trace": None}

# --------- Utilities ----------
def set_last_error(msg: str, trace: str = ""):
    LAST_ERROR["msg"] = msg
    LAST_ERROR["trace"] = trace
    logger.error(f"[last_error] {msg}\n{trace}")

async def line_api_get(url: str):
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    async with httpx.AsyncClient(timeout=10) as ac:
        r = await ac.get(url, headers=headers)
    r.raise_for_status()
    return r

async def line_api_post(url: str, payload: Dict[str, Any]):
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=10) as ac:
        r = await ac.post(url, headers=headers, content=json.dumps(payload, ensure_ascii=False))
    r.raise_for_status()
    return r

def verify_signature(body: bytes, signature: str) -> bool:
    if not LINE_CHANNEL_SECRET:
        return False
    mac = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, signature)

# ---- Normalization (ここが肝) ----
def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def normalize_expr(raw: str, angle_mode: str = "deg") -> str:
    """
    ユーザー入力を sympy が食べられる式に正規化する。
    - 全角/記号ゆれを統一
    - 暗黙の掛け算: 2sinx, 2(3+4), (a)b, 3π など
    - ^ → **
    - ° を rad(...) に置換（sin30°, 60° など）
    - mode:deg のときは sin30 → sin(rad(30)) のような“数値だけの引数”も度とみなす
    """
    s = nfkc(raw)
    # スペース除去
    s = re.sub(r"\s+", "", s)

    # 記号統一
    s = (
        s.replace("×", "*").replace("·", "*").replace("∙", "*")
         .replace("÷", "/")
         .replace("−", "-").replace("—", "-").replace("―", "-")
         .replace("，", ",")
         .replace("；", ";")
    )
    # べき
    s = s.replace("^", "**")

    # π, e
    s = s.replace("π", "pi").replace("Π", "pi").replace("ｅ", "e").replace("Ｅ", "E")

    # √x → sqrt(x)
    s = re.sub(r"√(?=[A-Za-z0-9\(])", "sqrt(", s)  # √3 → sqrt(3
    # √(… は上の置換で "sqrt((" になりがち、余計な "(" は parse_expr が丸めるのでOK
    # ）補完はしない（途中式は /calc_test で見える）

    # 暗黙の掛け算（数字の後に関数名/変数/括弧）
    s = re.sub(r"(?<=\d)(?=[A-Za-z\(])", "*", s)
    # 括弧閉じの後に数字/関数/変数
    s = re.sub(r"(?<=\))(?=[A-Za-z0-9\(])", ")*", s)
    # 定数 pi, e の前に係数
    s = re.sub(r"(?<=\d)(?=pi\b)", "*", s)
    s = re.sub(r"(?<=\d)(?=e\b)", "*", s)

    # --- 角度（°）の処理 ---
    # 1) sin(30°) / cos(…°) / tan(…°) → f(rad(数値))
    s = re.sub(
        r"(?<![A-Za-z0-9_])(sin|cos|tan)\(\s*(\d+(?:\.\d+)?)\s*°\s*\)",
        lambda m: f"{m.group(1)}(rad({m.group(2)}))",
        s,
    )
    # 2) sin30° のように括弧なし → f(rad(数値))
    s = re.sub(
        r"(?<![A-Za-z0-9_])(sin|cos|tan)\s*(\d+(?:\.\d+)?)\s*°",
        lambda m: f"{m.group(1)}(rad({m.group(2)}))",
        s,
    )
    # 3) mode=deg のとき、f(30) / f30 を f(rad(30)) に（引数が純数値のときだけ）
    if angle_mode == "deg":
        s = re.sub(
            r"(?<![A-Za-z0-9_])(sin|cos|tan)(?:\(\s*(\d+(?:\.\d+)?)\s*\)|\s*(\d+(?:\.\d+)?))",
            lambda m: f"{m.group(1)}(rad({m.group(2) or m.group(3)}))",
            s,
        )

    # 4) 孤立した 60° など → rad(60)
    s = re.sub(r"(\d+(?:\.\d+)?)°", r"rad(\1)", s)

    return s

def parse_and_eval(norm: str, prec_digits: int) -> Tuple[Any, Any]:
    # 安全な辞書（使う関数だけ）
    def rad(x):  # degrees → radians
        return x * pi / 180
    local = {
        "sin": sin, "cos": cos, "tan": tan,
        "sqrt": sqrt, "pi": pi, "e": E, "E": E,
        "rad": rad,
    }
    expr = parse_expr(norm, local_dict=local, transformations=TRANSFORMS, evaluate=False)
    exact = simplify(expr)
    approx = exact.evalf(prec_digits)
    return exact, approx

def make_cg50_guide(angle_mode: str, raw: str) -> str:
    # 超シンプルな“打鍵例”ガイド（説明テキスト）。厳密なキー列ではないが目安として。
    # 例: 2sin30°+60° → 角度:Deg → 入力: 2 × [SIN] 30 + 60 → [EXE]
    s = nfkc(raw)
    ex = (
        s.replace(" ", "")
         .replace("×", "×").replace("*", "×")
         .replace("^", "^")
         .replace("π", "pi")
    )
    # 関数名を大文字キー表記に
    ex = re.sub(r"sin", "[SIN]", ex, flags=re.I)
    ex = re.sub(r"cos", "[COS]", ex, flags=re.I)
    ex = re.sub(r"tan", "[TAN]", ex, flags=re.I)
    ex = ex.replace("**", "^")
    return f"fx-CG50 操作ガイド\n角度:{'Deg' if angle_mode=='deg' else 'Rad'} → 入力: {ex} → [EXE]"

def build_calc_response(raw_expr: str) -> str:
    norm = normalize_expr(raw_expr, ANGLE_MODE)
    exact, approx = parse_and_eval(norm, PREC_DIGITS)
    lines = []
    lines.append("[式]")
    lines.append(str(exact))
    lines.append(f"近似（{PREC_DIGITS}桁）: {approx}")
    lines.append("")
    lines.append(make_cg50_guide(ANGLE_MODE, raw_expr))
    return "\n".join(lines), norm

# --------- LINE webhook ----------
@app.post("/webhook")
async def webhook(request: Request):
    body = await request.body()
    sig = request.headers.get("X-Line-Signature", "")
    if not verify_signature(body, sig):
        # 署名不一致でも 200 を返してリトライ嵐を避ける
        set_last_error("signature verify failed", "")
        return PlainTextResponse("signature NG", status_code=200)

    try:
        payload = json.loads(body.decode("utf-8"))
        events = payload.get("events", [])
        for ev in events:
            if ev.get("type") != "message":
                continue
            reply_token = ev["replyToken"]
            msg = ev["message"]
            if msg.get("type") == "text":
                text = msg.get("text", "")
                lower = nfkc(text).lower()

                # ping
                if lower.startswith("ping"):
                    await reply_line(reply_token, ["pong ✅"])
                    continue

                # 角度モード
                if lower.startswith("mode:"):
                    global ANGLE_MODE
                    if "rad" in lower:
                        ANGLE_MODE = "rad"
                    else:
                        ANGLE_MODE = "deg"
                    await reply_line(reply_token, [f"角度モード: {'Deg' if ANGLE_MODE=='deg' else 'Rad'}"])
                    continue

                # 桁数
                if lower.startswith("prec:"):
                    global PREC_DIGITS
                    try:
                        PREC_DIGITS = max(3, min(20, int(lower.split(":",1)[1])))
                        await reply_line(reply_token, [f"桁数を {PREC_DIGITS} に設定しました"])
                    except Exception:
                        await reply_line(reply_token, ["桁数の指定が不正です（例: prec: 8）"])
                    continue

                # 計算
                if lower.startswith("calc:"):
                    expr = text.split(":",1)[1]
                    try:
                        out, norm = build_calc_response(expr)
                        await reply_line(reply_token, [out])
                    except Exception as e:
                        set_last_error(f"calc error: {e}", "")
                        await reply_line(reply_token, [
                            f"解析失敗: {e.__class__.__name__}\n入力: {expr}"
                        ])
                    continue

                # ヘルプ
                if lower in ("help", "usage", "使い方"):
                    await reply_line(reply_token, [
                        "使い方:\n"
                        "- mode:deg / mode:rad\n"
                        "- prec: 8  ← 近似の桁数\n"
                        "- calc: 2sin30° + 60°\n"
                        "- calc: sin30° + 3^2\n"
                        "- calc: tan45°"
                    ])
                    continue

                # その他はエコー
                await reply_line(reply_token, [f"echo: {text}"])
    except Exception as e:
        set_last_error(f"webhook exception: {e}", "")
    return PlainTextResponse("OK", status_code=200)

async def reply_line(reply_token: str, texts):
    msgs = [{"type": "text", "text": t} for t in texts]
    payload = {"replyToken": reply_token, "messages": msgs}
    await line_api_post("https://api.line.me/v2/bot/message/reply", payload)

# --------- Debug Endpoints ----------
@app.get("/")
async def root():
    return {"ok": True, "sympy": True, "latex_parser": True}

@app.get("/calc_test")
async def calc_test(expr: str):
    try:
        norm = normalize_expr(expr, ANGLE_MODE)
        exact, approx = parse_and_eval(norm, PREC_DIGITS)
        return {
            "raw": expr,
            "norm": norm,
            "exact": str(exact),
            "approx": str(approx),
            "mode": ANGLE_MODE,
            "prec_digits": PREC_DIGITS,
        }
    except Exception as e:
        set_last_error(f"calc_test error: {e}", "")
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/botinfo")
async def botinfo():
    try:
        r = await line_api_get("https://api.line.me/v2/bot/info")
        return Response(r.text, media_type="application/json", status_code=r.status_code)
    except Exception as e:
        set_last_error(f"botinfo error: {e}", "")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/envcheck")
async def envcheck():
    return {
        "access_token": (f"{len(LINE_CHANNEL_ACCESS_TOKEN)} chars" if LINE_CHANNEL_ACCESS_TOKEN else None),
        "channel_secret": (f"{len(LINE_CHANNEL_SECRET)} chars" if LINE_CHANNEL_SECRET else None),
        "angle_mode": ANGLE_MODE,
        "prec_digits": PREC_DIGITS,
    }

@app.get("/last_error")
async def last_error():
    return LAST_ERROR
