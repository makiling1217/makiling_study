# main.py — FastAPI only / LINE bot / 計算＆電卓キー案内（必要時のみ）/ 正式名称で表記
import os, hmac, hashlib, base64, json, re, unicodedata, logging
from typing import Dict, Any, Tuple, List
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
import httpx

from sympy import sin, cos, tan, sqrt, pi, E, simplify
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    function_exponentiation,
    convert_xor,
)

TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    function_exponentiation,
    convert_xor,
)

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")

# 状態
ANGLE_MODE = "deg"        # "deg" or "rad"
PREC_DIGITS = 6
GUIDE_MODE = "smart"      # "on" | "off" | "smart"
LAST_ERROR: Dict[str, Any] = {"msg": None, "trace": None}

# =============== Utility for LINE ===============
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

# =============== Calc Core ===============
def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def normalize_expr(raw: str, angle_mode: str = "deg") -> str:
    """
    仕様：
      - sin30° / sin(30°) → sin(rad(30))
      - degモードでは sin30 → sin(rad(30)) に統一
      - 単独の 60° の “°” は外す（= 60）
    """
    s = nfkc(raw)
    s = re.sub(r"\s+", "", s)

    s = (s.replace("×", "*").replace("·", "*").replace("∙", "*")
           .replace("÷", "/")
           .replace("−", "-").replace("—", "-").replace("―", "-")
           .replace("，", ",").replace("；", ";"))
    s = s.replace("^", "**")
    s = s.replace("π", "pi").replace("Π", "pi").replace("ｅ", "e").replace("Ｅ", "E")

    # √x → sqrt(x)
    s = re.sub(r"√(?=[A-Za-z0-9\(])", "sqrt(", s)

    # 暗黙の掛け算
    s = re.sub(r"(?<=\d)(?=[A-Za-z\(])", "*", s)      # 2x, 2(x)
    s = re.sub(r"(?<=\))(?=[A-Za-z0-9\(])", ")*", s)  # )( → )*
    s = re.sub(r"(?<=\d)(?=pi\b)", "*", s)
    s = re.sub(r"(?<=\d)(?=e\b)", "*", s)

    # trig(数値°)
    s = re.sub(
        r"(?<![A-Za-z0-9_])(sin|cos|tan)\(\s*(\d+(?:\.\d+)?)\s*°\s*\)",
        lambda m: f"{m.group(1)}(rad({m.group(2)}))", s)
    s = re.sub(
        r"(?<![A-Za-z0-9_])(sin|cos|tan)\s*(\d+(?:\.\d+)?)\s*°",
        lambda m: f"{m.group(1)}(rad({m.group(2)}))", s)

    # degモード：trig(数値) → trig(rad(数値))
    if angle_mode == "deg":
        s = re.sub(
            r"(?<![A-Za-z0-9_])(sin|cos|tan)(?:\(\s*(\d+(?:\.\d+)?)\s*\)|\s*(\d+(?:\.\d+)?))",
            lambda m: f"{m.group(1)}(rad({m.group(2) or m.group(3)}))", s)

    # 単独の “n°” は ° を取り除く
    s = re.sub(r"(\d+(?:\.\d+)?)°", r"\1", s)

    return s

def parse_and_eval(norm: str, prec_digits: int):
    def rad(x): return x * pi / 180
    local = {
        "sin": sin, "cos": cos, "tan": tan,
        "sqrt": sqrt, "pi": pi, "e": E, "E": E,
        "rad": rad
    }
    expr = parse_expr(norm, local_dict=local, transformations=TRANSFORMS, evaluate=False)
    exact = simplify(expr)
    approx = exact.evalf(prec_digits)
    return exact, approx

# =============== Key Guide ===============
KEY_MAP = {"+":"[+]", "-":"[-]", "*":"[×]", "/":"[÷]", "^":"[^]", "(": "[ ( ]", ")":"[ ) ]"}
FUNC_KEY = {"sin":"[SIN]", "cos":"[COS]", "tan":"[TAN]"}

def tokenize_for_keys(raw: str) -> List[str]:
    s = nfkc(raw).replace("×", "*").replace("÷", "/").replace("−", "-").replace(" ", "").replace("°", "")
    s = s.replace("π", "pi")
    s = re.sub(r"√(?=[A-Za-z0-9\(])", "sqrt(", s)
    token_re = re.compile(r"(sin|cos|tan|sqrt|pi|\d+|\^|\+|\-|\*|/|\(|\))", re.I)
    return token_re.findall(s)

def needs_angle_steps(raw: str) -> bool:
    t = nfkc(raw)
    return bool(re.search(r"(sin|cos|tan|°)", t, flags=re.I))

def key_steps(raw: str, angle_mode: str, include_angle_steps: bool) -> str:
    tokens = tokenize_for_keys(raw)
    steps: List[str] = []
    n = 1
    def push(x):
        nonlocal n; steps.append(f"{n}. {x}"); n += 1

    if include_angle_steps:
        push("（必要時）角度モードを設定： [SHIFT] → [MENUキー]（SETUP） → Angle を選択 → Deg/Rad を選ぶ → [EXEキー] → [EXITキー]")
        steps.append("   - Deg（度数法）: Angle: Deg")
        steps.append("   - Rad（弧度法）: Angle: Rad")
        steps.append("")

    # 入力キー列
    i = 0
    while i < len(tokens):
        t = tokens[i].lower()
        if t.isdigit():
            digs = [tokens[i]]
            j = i + 1
            while j < len(tokens) and tokens[j].isdigit():
                digs.append(tokens[j]); j += 1
            push("".join(f"[{d}]" for d in digs)); i = j; continue
        if t in FUNC_KEY:
            push(FUNC_KEY[t] + "（自動で '(' が入る）"); i += 1; continue
        if t == "sqrt":
            push("[√]（自動で '(' が入る）"); i += 1; continue
        if t == "pi":
            push("[π]"); i += 1; continue
        if t in KEY_MAP:
            push(KEY_MAP[t]); i += 1; continue
        push(f"[{tokens[i]}]"); i += 1

    push("[EXEキー]")
    steps.append("")
    steps.append("★困ったら： [EXITキー] で1画面戻る / [AC] で行をクリア")
    steps.append("★SIN/COS/TAN は押すと '(' が入るので引数の後に [ ) ] を押す")
    steps.append("★べき乗は [^]、平方根は [√]、円周率は [π]")
    return "\n".join(steps)

def build_calc_response(raw_expr: str) -> Tuple[str, str]:
    norm = normalize_expr(raw_expr, ANGLE_MODE)
    exact, approx = parse_and_eval(norm, PREC_DIGITS)

    out_lines = []
    out_lines.append("[式]")
    out_lines.append(str(exact))
    out_lines.append(f"近似（{PREC_DIGITS}桁）: {approx}")
    out_lines.append("")
    out_lines.append("【答え】")
    out_lines.append(f"厳密: {exact}")
    out_lines.append(f"小数: {approx}")

    include_angle = False
    if GUIDE_MODE == "on":
        include_angle = needs_angle_steps(raw_expr)
    elif GUIDE_MODE == "smart":
        include_angle = needs_angle_steps(raw_expr)
    else:  # "off"
        include_angle = False

    if GUIDE_MODE != "off":
        out_lines.append("")
        out_lines.append("fx-CG50 操作ガイド（この式向け）")
        out_lines.append(key_steps(raw_expr, ANGLE_MODE, include_angle))

    return "\n".join(out_lines), norm

# =============== Webhook ===============
@app.post("/webhook")
async def webhook(request: Request):
    body = await request.body()
    sig = request.headers.get("X-Line-Signature", "")
    if not verify_signature(body, sig):
        set_last_error("signature verify failed", "")
        return PlainTextResponse("signature NG", status_code=200)

    try:
        payload = json.loads(body.decode("utf-8"))
        for ev in payload.get("events", []):
            if ev.get("type") != "message": continue
            if ev["message"].get("type") != "text": continue
            reply_token = ev["replyToken"]
            text = ev["message"]["text"]
            lower = nfkc(text).lower()

            if lower.startswith("ping"):
                await reply_line(reply_token, ["pong ✅"]); continue

            if lower.startswith("mode:"):
                global ANGLE_MODE
                ANGLE_MODE = "rad" if "rad" in lower else "deg"
                await reply_line(reply_token, [f"角度モード: {'Deg' if ANGLE_MODE=='deg' else 'Rad'}"]); continue

            if lower.startswith("prec:"):
                global PREC_DIGITS
                try:
                    PREC_DIGITS = max(3, min(20, int(lower.split(":",1)[1])))
                    await reply_line(reply_token, [f"桁数を {PREC_DIGITS} に設定しました"])
                except Exception:
                    await reply_line(reply_token, ["桁数の指定が不正です（例: prec: 8）"])
                continue

            if lower.startswith("guide:"):
                global GUIDE_MODE
                v = lower.split(":",1)[1].strip()
                if v in ("on","off","smart"):
                    GUIDE_MODE = v
                    await reply_line(reply_token, [f"操作ガイド表示: {GUIDE_MODE}"])
                else:
                    await reply_line(reply_token, ["guide:on / guide:off / guide:smart のどれかを指定してください"])
                continue

            if lower.startswith("calc:"):
                expr = text.split(":",1)[1]
                try:
                    out, _ = build_calc_response(expr)
                    await reply_line(reply_token, [out])
                except Exception as e:
                    set_last_error(f"calc error: {e}", "")
                    await reply_line(reply_token, [f"解析失敗: {e.__class__.__name__}\n入力: {expr}"])
                continue

            if lower in ("help","usage","使い方"):
                await reply_line(reply_token, [
                    "使い方:\n"
                    "- mode:deg / mode:rad（角度）\n"
                    "- prec: 8（近似の桁数）\n"
                    "- guide:on|off|smart（操作ガイド表示）\n"
                    "- calc: 2sin30° + 60° / calc: sin30° + 3^2 / calc: tan45°"
                ])
                continue

            await reply_line(reply_token, [f"echo: {text}"])

    except Exception as e:
        set_last_error(f"webhook exception: {e}", "")
    return PlainTextResponse("OK", status_code=200)

async def reply_line(reply_token: str, texts):
    msgs = [{"type":"text","text":t} for t in texts]
    payload = {"replyToken": reply_token, "messages": msgs}
    await line_api_post("https://api.line.me/v2/bot/message/reply", payload)

# =============== Debug Endpoints ===============
@app.get("/")
async def root():
    return {"ok": True, "sympy": True, "latex_parser": True}

@app.get("/calc_test")
async def calc_test(expr: str):
    try:
        norm = normalize_expr(expr, ANGLE_MODE)
        exact, approx = parse_and_eval(norm, PREC_DIGITS)
        return {"raw": expr, "norm": norm, "exact": str(exact), "approx": str(approx),
                "mode": ANGLE_MODE, "prec_digits": PREC_DIGITS, "guide_mode": GUIDE_MODE}
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
        "guide_mode": GUIDE_MODE,
    }

@app.get("/last_error")
async def last_error():
    return LAST_ERROR
