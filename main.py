# main.py — LINE Bot: Image→OCR(Mathpix)→Sympy(CAS)→Answer & (optional) fx-CG50 keys
import os, hmac, hashlib, base64, json, re, logging, math
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse, Response
import httpx

# ===== Sympy（厳密計算） =====
SYM_AVAILABLE = True
try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations,
        implicit_multiplication_application, convert_xor, function_exponentiation
    )
    try:
        # 可能なら LaTeX パースも（Sympy 1.13+）
        from sympy.parsing.latex import parse_latex
        HAS_PARSE_LATEX = True
    except Exception:
        HAS_PARSE_LATEX = False
except Exception:
    SYM_AVAILABLE = False
    HAS_PARSE_LATEX = False

app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
MATHPIX_APP_ID = os.environ.get("MATHPIX_APP_ID", "")
MATHPIX_APP_KEY = os.environ.get("MATHPIX_APP_KEY", "")
latex_list = []
if MATHPIX_APP_ID and MATHPIX_APP_KEY:
    try:
        mp = await ocr_mathpix(cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY),95])[1].tobytes())
        latex_list = extract_latex(mp)
    except Exception as ex:
        expr_blocks.append(f"【数式OCR】Mathpix失敗: {ex}")
else:
    expr_blocks.append("【数式OCR】Mathpix未設定のためスキップしました。")

LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
LINE_CONTENT_URL = "https://api-data.line.me/v2/bot/message/{messageId}/content"  # ←画像は api-data 固定

# 角度モード（deg/rad）
ANGLE_MODE = {"mode": "deg"}  # 既定：度

# ====== ユーティリティ ======
def verify_signature(channel_secret: str, body: bytes, signature: str) -> bool:
    mac = hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, signature or "")

async def reply_message(reply_token: str, messages: List[Dict[str, Any]]) -> None:
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}", "Content-Type": "application/json"}
    payload = {"replyToken": reply_token, "messages": messages}
    async with httpx.AsyncClient(timeout=30) as ac:
        r = await ac.post(LINE_REPLY_URL, headers=headers, json=payload)
        logging.info(f'POST reply "{r.status_code}"')
        r.raise_for_status()

def chunk_text(txt: str, limit: int = 4500) -> List[str]:
    # LINEは1メッセージ最大約5000文字。余裕をみて分割
    parts = []
    while txt:
        parts.append(txt[:limit])
        txt = txt[limit:]
    return parts

async def reply_long_text(reply_token: str, txt: str) -> None:
    chunks = chunk_text(txt)
    await reply_message(reply_token, [{"type": "text", "text": c} for c in chunks])

async def get_line_image_bytes(message_id: str) -> bytes:
    url = LINE_CONTENT_URL.format(messageId=message_id)
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    async with httpx.AsyncClient(timeout=30) as ac:
        r = await ac.get(url, headers=headers)
        logging.info(f'GET {url} "{r.status_code}"')
        r.raise_for_status()
        return r.content

# ====== 入力正規化（全角→半角、°、√、×÷ など） ======
_ZK = "０１２３４５６７８９（）＊＋－／＾，．　ｉｊｘ"
_HK = "0123456789()*+-/^,. ijx"
assert len(_ZK) == len(_HK)
TRANS = str.maketrans(_ZK, _HK)

def normalize_expr(s: str) -> str:
    s = s.translate(TRANS)
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-").replace("–", "-")
    s = s.replace("π", "pi")
    # √2 → sqrt(2
    s = re.sub(r"√\s*([0-9a-zA-Z_\(])", r"sqrt(\1", s)
    # 30° → (30 deg)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*°", r"(\1 deg)", s)
    # 関数の括弧省略（sin30 → sin(30)）
    s = re.sub(r"\b(sin|cos|tan|asin|acos|atan|sinh|cosh|tanh)\s*([0-9πpieij\.]+(?:\s*deg)?)",
               r"\1(\2)", s, flags=re.IGNORECASE)
    # i/j → I
    s = re.sub(r"\b([0-9\.]+)[ij]\b", r"\1*I", s, flags=re.IGNORECASE)
    s = re.sub(r"\b[ij]\b", "I", s, flags=re.IGNORECASE)
    s = s.replace("^", "**")
    s = re.sub(r"\s+", "", s)
    return s

# ====== Sympy 準備 ======
if SYM_AVAILABLE:
    def _deg(x): return x * sp.pi / 180

    def nCr(n, r): return sp.binomial(n, r)
    def nPr(n, r): return sp.factorial(n) / sp.factorial(n - r)

    def _wrap_trig(func):
        def f(x):
            if ANGLE_MODE["mode"] == "deg":
                return func(_deg(x))
            return func(x)
        return f

    SYM_LOCALS = {
        "pi": sp.pi, "e": sp.E, "I": sp.I,
        "abs": sp.Abs, "sqrt": sp.sqrt, "exp": sp.exp,
        "log": sp.log, "log10": lambda x: sp.log(x, 10),
        "floor": sp.floor, "ceil": sp.ceiling,
        "nCr": nCr, "C": nCr, "comb": nCr,
        "nPr": nPr, "P": nPr, "perm": nPr,
        "sin": _wrap_trig(sp.sin), "cos": _wrap_trig(sp.cos), "tan": _wrap_trig(sp.tan),
        "asin": lambda x: sp.asin(x) if ANGLE_MODE["mode"]=="rad" else sp.deg(sp.asin(x)),
        "acos": lambda x: sp.acos(x) if ANGLE_MODE["mode"]=="rad" else sp.deg(sp.acos(x)),
        "atan": lambda x: sp.atan(x) if ANGLE_MODE["mode"]=="rad" else sp.deg(sp.atan(x)),
        "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
        "deg": lambda x: _deg(x),
        "Matrix": sp.Matrix,
    }
    TRANSFORMS = (standard_transformations
                  + (implicit_multiplication_application,)
                  + (convert_xor,)
                  + (function_exponentiation,))

    def sym_parse(expr: str):
        return parse_expr(expr, local_dict=SYM_LOCALS, transformations=TRANSFORMS, evaluate=True)

    def sym_eval_numeric(expr: str):
        e = sym_parse(expr)
        return sp.N(e)

# ====== fx-CG50 キー列（簡易ルール） ======
def cg50_keyseq(expr_show: str) -> str:
    s = expr_show.replace("**","^").replace("*","×").replace("/","÷")
    s = (s.replace("asin","[SHIFT][SIN]^-1")
           .replace("acos","[SHIFT][COS]^-1")
           .replace("atan","[SHIFT][TAN]^-1")
           .replace("sin","[SIN]").replace("cos","[COS]").replace("tan","[TAN]")
           .replace("sqrt","[√]").replace("log10","[LOG]10,").replace("log","[LN]"))
    return "角度:" + ("Deg" if ANGLE_MODE["mode"]=="deg" else "Rad") + " → 入力: " + s + " → [EXE]"

# ====== OCR（Mathpix） ======
async def ocr_mathpix(image_bytes: bytes) -> Dict[str, Any]:
    if not (MATHPIX_APP_ID and MATHPIX_APP_KEY):
        raise RuntimeError("Mathpix API keys not set")
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "src": f"data:image/jpeg;base64,{b64}",
        "formats": ["text", "data", "latex_simplified"],
        "rm_spaces": True,
        "math_inline_delimiters": ["$", "$"],
        "math_block_delimiters": ["$$", "$$"],
    }
    headers = {
        "app_id": MATHPIX_APP_ID,
        "app_key": MATHPIX_APP_KEY,
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient(timeout=45) as ac:
        r = await ac.post("https://api.mathpix.com/v3/text", headers=headers, json=payload)
        r.raise_for_status()
        return r.json()

def extract_expressions(mp: Dict[str, Any]) -> List[str]:
    exprs: List[str] = []
    # 1) data.items から latex を拾う
    items = (mp.get("data") or {}).get("items") or []
    for it in items:
        if it.get("type") in ("latex", "asciimath", "mathml") and it.get("value"):
            exprs.append(it["value"])
    # 2) latex_simplified があれば追加
    if isinstance(mp.get("latex_simplified"), str) and mp["latex_simplified"].strip():
        exprs.append(mp["latex_simplified"].strip())
    # 3) テキスト内の $...$ / \(...\) / \[...\] を拾う
    text = (mp.get("text") or "")
    for pat in [r"\$(.+?)\$", r"\\\((.+?)\\\)", r"\\\[(.+?)\\\]"]:
        for m in re.finditer(pat, text, flags=re.S):
            exprs.append(m.group(1))
    # 重複削除・短すぎるもの除去
    uniq = []
    for e in exprs:
        e = e.strip()
        if len(e) < 2: continue
        if e not in uniq:
            uniq.append(e)
    return uniq

def latex_or_text_to_sympy(s: str) -> Tuple[Optional[sp.Expr], str]:
    """ LaTeX優先でSympy式へ。失敗時は正規化テキストで再挑戦。戻り値：(式 or None, 表示用テキスト) """
    disp = s
    if HAS_PARSE_LATEX:
        try:
            e = parse_latex(s)
            return e, disp
        except Exception:
            pass
    # LaTeXが無理ならテキストとして解釈
    norm = normalize_expr(s)
    try:
        e = sym_parse(norm)
        return e, norm.replace("**", "^")
    except Exception:
        return None, disp

def try_solve_expr(e: sp.Expr) -> Tuple[str, Optional[sp.Expr], Optional[sp.Expr]]:
    """
    与式 e について、数値評価 or 方程式解を試す。
    戻り：("eval"/"solve"/"skip", 表示用式, 結果)
    """
    try:
        # 方程式っぽい？
        if isinstance(e, sp.Equality):
            # e.lhs = e.rhs の解
            syms = sorted(list(e.free_symbols), key=lambda s: s.name)
            if not syms:
                return "eval", e, sp.simplify(e.lhs - e.rhs)
            sol = sp.solve(e, *syms, dict=True)
            return "solve", e, sp.ImmutableDenseNDimArray(sol)
        # 方程式でない式：数値 or 代数簡約
        numeric = sp.N(e)
        return "eval", e, numeric
    except Exception:
        # どうしても解けない：そのまま返す
        return "skip", e, None

def build_answer_block(kind: str, expr: sp.Expr, result: Optional[sp.Expr]) -> str:
    if kind == "solve":
        return f"[方程式] {sp.srepr(expr)}\n解: {result}"
    if kind == "eval":
        return f"[式] {sp.srepr(expr)}\n結果: {result}"
    return f"[式] {sp.srepr(expr)}\n結果: <解析不可>"

def maybe_keyseq(expr: sp.Expr) -> str:
    # “大人向け”は既定でキー列を非表示にしていたが、要望で常に表示に変更
    shown = str(sp.simplify(expr)).replace("**","^")
    return "fx-CG50 操作ガイド\n" + cg50_keyseq(shown)

# ====== ルーティング ======
@app.get("/")
async def root():
    return {"ok": True, "sympy": SYM_AVAILABLE, "latex_parser": HAS_PARSE_LATEX}

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
            # ===== テキスト =====
            if msg_type == "text":
                text = (m.get("text") or "").strip()

                # 角度モード切替
                if text.lower().startswith("mode:"):
                    v = text.split(":",1)[1].strip().lower()
                    if v in ("deg","degree","degrees"):
                        ANGLE_MODE["mode"]="deg"
                        await reply_message(reply_token,[{"type":"text","text":"角度モード: Deg"}]); continue
                    if v in ("rad","radian","radians"):
                        ANGLE_MODE["mode"]="rad"
                        await reply_message(reply_token,[{"type":"text","text":"角度モード: Rad"}]); continue
                    await reply_message(reply_token,[{"type":"text","text":"mode:deg / mode:rad"}]); continue

                # calc:
                if text.lower().startswith("calc:"):
                    if not SYM_AVAILABLE:
                        await reply_long_text(reply_token,"Sympy 未導入のため計算不可。requirements.txt に sympy を追加してください。"); continue
                    raw = text[5:].strip()
                    if not raw:
                        await reply_message(reply_token,[{"type":"text","text":"式が空です。例: calc: sin30° + 3^2"}]); continue
                    norm = normalize_expr(raw)
                    try:
                        e = sym_parse(norm)
                        kind, expr, res = try_solve_expr(e)
                        block = build_answer_block(kind, expr, res)
                        guide = maybe_keyseq(expr)
                        await reply_long_text(reply_token, f"{block}\n\n{guide}")
                    except Exception as ex:
                        await reply_long_text(reply_token, f"解析失敗: {ex}\n入力: {raw}")
                    continue

                # solve:, diff:, int:, factor:, expand:, matrix: は省略（必要なら前の版を流用）

                # それ以外
                await reply_message(reply_token,[{"type":"text","text":"画像を送れば式→解→操作手順まで自動で返します。テキスト計算は calc: から。"}])
                continue

            # ===== 画像 =====
            if msg_type == "image":
                # 画像取得
                cp = m.get("contentProvider") or {}
                if cp.get("type") == "external" and cp.get("originalContentUrl"):
                    async with httpx.AsyncClient(timeout=30) as ac:
                        r = await ac.get(cp["originalContentUrl"]); r.raise_for_status()
                        img = r.content
                else:
                    img = await get_line_image_bytes(m.get("id"))
                if not SYM_AVAILABLE:
                    await reply_long_text(reply_token, "Sympy 未導入のため解答を生成できません。requirements.txt に sympy を追加してください。"); continue

                # OCR
                try:
                    mp = await ocr_mathpix(img)
                except Exception as ex:
                    await reply_long_text(reply_token, f"OCR失敗: {ex}\n（MathpixのAPP_ID/KEY、画像の解像度をご確認ください）")
                    continue

                exprs_latex = extract_expressions(mp)
                if not exprs_latex:
                    await reply_long_text(reply_token, "数式を検出できませんでした。影/傾き/ピンぼけを避け、用紙全体が入るよう撮影してください。")
                    continue

                answers: List[str] = []
                for idx, exs in enumerate(exprs_latex, 1):
                    e_sym, shown = latex_or_text_to_sympy(exs)
                    if e_sym is None:
                        answers.append(f"#{idx}\n抽出: {shown}\n→ 解析不可")
                        continue
                    kind, expr, res = try_solve_expr(e_sym)
                    block = build_answer_block(kind, expr, res)
                    guide = maybe_keyseq(expr)
                    answers.append(f"#{idx}\n抽出: {shown}\n{block}\n\n{guide}")

                await reply_long_text(reply_token, "\n\n".join(answers))
                continue

            # ===== 未対応 =====
            await reply_message(reply_token,[{"type":"text","text":f"未対応メッセージタイプ: {msg_type}"}])

        except httpx.HTTPStatusError as he:
            await reply_message(reply_token,[{"type":"text","text":f"HTTPエラー: {he.response.status_code}"}])
            logging.exception("HTTPStatusError")
        except Exception:
            await reply_message(reply_token,[{"type":"text","text":"内部エラーが発生しました。"}])
            logging.exception("Unhandled error")

    return JSONResponse({"status":"ok"})

