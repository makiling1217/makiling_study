# main.py — LINE Bot: OCR(optional) + Sympy CAS + 電卓手順（堅牢化版）
import os, hmac, hashlib, base64, json, re, logging, traceback
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse, Response
import httpx

# ====== Sympy (CAS) ======
SYM_AVAILABLE = True
try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations,
        implicit_multiplication_application, convert_xor, function_exponentiation
    )
    try:
        from sympy.parsing.latex import parse_latex
        HAS_PARSE_LATEX = True
    except Exception:
        HAS_PARSE_LATEX = False
except Exception:
    SYM_AVAILABLE = False
    HAS_PARSE_LATEX = False

# ====== 画像前処理 ======
import numpy as np
import cv2

# ====== RapidOCR（遅延初期化） ======
RAPID_IMPORTED = False
rapid_ocr = None
def get_rapid():
    global RAPID_IMPORTED, rapid_ocr
    if rapid_ocr is not None:
        return rapid_ocr
    if not RAPID_IMPORTED:
        try:
            from rapidocr_onnxruntime import RapidOCR
            rapid_ocr = RapidOCR()
            RAPID_IMPORTED = True
            logging.info("RapidOCR initialized")
        except Exception as e:
            RAPID_IMPORTED = True
            rapid_ocr = None
            logging.error(f"RapidOCR init failed: {e}")
    return rapid_ocr

# ====== FastAPI ======
app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
MATHPIX_APP_ID = os.environ.get("MATHPIX_APP_ID", "")
MATHPIX_APP_KEY = os.environ.get("MATHPIX_APP_KEY", "")

LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
LINE_CONTENT_URL = "https://api-data.line.me/v2/bot/message/{messageId}/content"

# 角度モード（asin/acos/atan の返りを度/ラジアン切替）
ANGLE_MODE = {"mode": "deg"}  # "deg" or "rad"

# 近似表示の桁数
PREC = {"digits": int(os.environ.get("PREC_DIGITS", "6"))}

# 直近エラー保存（診断用）
LAST_ERROR = {"msg": None, "trace": None}

# ====== ユーティリティ ======
def verify_signature(secret: str, body: bytes, signature: str) -> bool:
    mac = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
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
    out = []
    while txt:
        out.append(txt[:limit]); txt = txt[limit:]
    return out

async def reply_long_text(reply_token: str, txt: str) -> None:
    chunks = chunk_text(txt)
    await reply_message(reply_token, [{"type":"text","text":c} for c in chunks])

async def get_line_image_bytes(message_id: str) -> bytes:
    url = LINE_CONTENT_URL.format(messageId=message_id)
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    async with httpx.AsyncClient(timeout=30) as ac:
        r = await ac.get(url, headers=headers)
        logging.info(f'GET {url} "{r.status_code}"')
        r.raise_for_status()
        return r.content

# ====== 全角→半角 ======
_ZK = "０１２３４５６７８９（）＊＋－／＾，．　ｉｊｘ"
_HK = "0123456789()*+-/^,. ijx"
assert len(_ZK)==len(_HK), "maketrans length mismatch"
TRANS = str.maketrans(_ZK, _HK)

# --- 式の正規化（度→ラジアン、暗黙の掛け算を強化）---
def normalize_expr(s: str) -> str:
    # 全角→半角・記号の統一
    s = s.translate(TRANS)
    s = (s.replace("×","*").replace("·","*").replace("∙","*").replace("・","*")
           .replace("÷","/").replace("−","-").replace("–","-")
           .replace("π","pi").replace("º","°"))
    # √n → sqrt(n
    s = re.sub(r"√\s*([0-9a-zA-Z_\(])", r"sqrt(\1", s)

    # 余分な空白を1個に
    s = re.sub(r"\s+", " ", s)

    # 1) f(… )° → f(rad(…))  （sin(30)°）
    s = re.sub(r"\b(sin|cos|tan|sinh|cosh|tanh)\s*\(\s*([^()]+?)\s*\)\s*°",
               r"\1(rad(\2))", s, flags=re.I)

    # 2) f 数字° → f(rad(数字))  （sin30°）
    s = re.sub(r"\b(sin|cos|tan|sinh|cosh|tanh)\s*([0-9]+(?:\.[0-9]+)?)\s*°",
               r"\1(rad(\2))", s, flags=re.I)

    # 3) 数字の度記号 30° / 30deg → rad(30)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*(?:°|deg|degree|degrees)\b",
               r"rad(\1)", s, flags=re.I)

    # 4) 関数の括弧省略（sin30 → sin(30) など）
    s = re.sub(r"\b(sin|cos|tan|sinh|cosh|tanh|asin|acos|atan)\s*([0-9]+(?:\.[0-9]+)?)\b",
               r"\1(\2)", s, flags=re.I)

    # 5) 暗黙の掛け算を明示化
    s = re.sub(r"(?<=[0-9\)])\s*(?=(?:sin|cos|tan|sinh|cosh|tanh|asin|acos|atan|sqrt|rad)\s*\()",
               "*", s, flags=re.I)
    s = re.sub(r"(?<=\d)\s*(?=[A-Za-z])", "*", s)          # 2x, 3pi
    s = re.sub(r"(?<=\))\s*(?=[0-9A-Za-z])", "*", s)       # )( くっつき

    # 6) べき
    s = s.replace("^", "**")

    # 7) 虚数 i/j（数の直後のみ）→ I
    s = re.sub(r"(\d)\s*[ij]\b", r"\1*I", s, flags=re.I)
    s = re.sub(r"\b[ij]\b", "I", s, flags=re.I)

    # 8) 最終整形
    s = s.replace(" ", "").replace("°","")  # ここまでで rad 化済みの想定
    return s

# ====== Sympy 準備 ======
if SYM_AVAILABLE:
    def _rad2deg(x): return x * 180 / sp.pi
    def nCr(n,r): return sp.binomial(n,r)
    def nPr(n,r): return sp.factorial(n)/sp.factorial(n-r)

    def asin_mode(x): 
        y = sp.asin(x)
        return y if ANGLE_MODE["mode"]=="rad" else _rad2deg(y)
    def acos_mode(x):
        y = sp.acos(x)
        return y if ANGLE_MODE["mode"]=="rad" else _rad2deg(y)
    def atan_mode(x):
        y = sp.atan(x)
        return y if ANGLE_MODE["mode"]=="rad" else _rad2deg(y)

    SYM_LOCALS = {
        "pi": sp.pi, "e": sp.E, "I": sp.I,
        "abs": sp.Abs, "sqrt": sp.sqrt, "exp": sp.exp,
        "log": sp.log, "log10": lambda x: sp.log(x,10),
        "floor": sp.floor, "ceil": sp.ceiling,
        "nCr": nCr, "C": nCr, "comb": nCr,
        "nPr": nPr, "P": nPr, "perm": nPr,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "asin": asin_mode, "acos": acos_mode, "atan": atan_mode,
        "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
        "Matrix": sp.Matrix,
        "rad": (lambda x: x*sp.pi/180),
    }

    TRANSFORMS = (standard_transformations
                  + (implicit_multiplication_application,)
                  + (convert_xor,)
                  + (function_exponentiation,))

    def sym_parse(expr: str):
        return parse_expr(expr, local_dict=SYM_LOCALS, transformations=TRANSFORMS, evaluate=True)

# ====== 電卓キー列（fx-CG50 風） ======
def cg50_keyseq(expr_show: str) -> str:
    s = expr_show.replace("**","^").replace("*","×").replace("/","÷")
    s = (s.replace("asin","[SHIFT][SIN]^-1")
           .replace("acos","[SHIFT][COS]^-1")
           .replace("atan","[SHIFT][TAN]^-1")
           .replace("sin","[SIN]").replace("cos","[COS]").replace("tan","[TAN]")
           .replace("sqrt","[√]").replace("log10","[LOG]10,").replace("log","[LN]"))
    return "角度:" + ("Deg" if ANGLE_MODE["mode"]=="deg" else "Rad") + " → 入力: " + s + " → [EXE]"

# ====== 画像前処理 ======
def preprocess(img_bytes: bytes):
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im is None: raise ValueError("decode error")
    h, w = im.shape[:2]
    short = min(h, w)
    scale = 1280.0 / short if short < 1280 else 1.5
    im_big = cv2.resize(im, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(im_big, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray < 250))
    angle = 0.0
    if coords.size > 0:
        rect = cv2.minAreaRect(coords); angle = rect[-1]
        angle = -(90 + angle) if angle < -45 else -angle
    M = cv2.getRotationMatrix2D((gray.shape[1]//2, gray.shape[0]//2), angle, 1.0)
    gray_rot = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enh = clahe.apply(gray_rot)
    bw = cv2.adaptiveThreshold(enh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15)
    return bw, gray_rot

# ====== RapidOCR（テキストOCR） ======
def rapid_ocr_text(img_gray) -> str:
    ocr = get_rapid()
    if ocr is None:
        return "[RapidOCR 未使用] 初期化に失敗したため、問題文OCRは省略されました。"
    rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    res, _ = ocr(rgb)
    lines = []
    for item in (res or []):
        txt = (item[1][0] or "").strip()
        if txt: lines.append(txt)
    text = "\n".join(lines)
    text = re.sub(r"[^\S\r\n]+", " ", text)
    return text.strip() or "(検出なし)"

# ====== Mathpix（キーがある時のみ使用） ======
async def ocr_mathpix(image_bytes: bytes) -> Dict[str, Any]:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {"src": f"data:image/jpeg;base64,{b64}",
               "formats": ["text","data","latex_simplified"],
               "rm_spaces": True,
               "math_inline_delimiters": ["$","$"],
               "math_block_delimiters": ["$$","$$"]}
    headers = {"app_id": MATHPIX_APP_ID, "app_key": MATHPIX_APP_KEY, "Content-Type":"application/json"}
    async with httpx.AsyncClient(timeout=45) as ac:
        r = await ac.post("https://api.mathpix.com/v3/text", headers=headers, json=payload)
        r.raise_for_status()
        return r.json()

def extract_latex(mp: Dict[str, Any]) -> List[str]:
    exprs: List[str] = []
    items = (mp.get("data") or {}).get("items") or []
    for it in items:
        if it.get("type") in ("latex","asciimath","mathml") and it.get("value"):
            exprs.append(it["value"])
    if isinstance(mp.get("latex_simplified"), str) and mp["latex_simplified"].strip():
        exprs.append(mp["latex_simplified"].strip())
    text = (mp.get("text") or "")
    for pat in [r"\$(.+?)\$", r"\\\((.+?)\\\)", r"\\\[(.+?))\\\]"]:
        for m in re.finditer(pat, text, flags=re.S):
            exprs.append(m.group(1))
    uniq = []
    for e in exprs:
        e = e.strip()
        if len(e) < 2: continue
        if e not in uniq: uniq.append(e)
    return uniq

def latex_or_text_to_sympy(s: str) -> Tuple[Optional['sp.Expr'], str]:
    if not SYM_AVAILABLE:
        return None, s
    disp = s
    if HAS_PARSE_LATEX:
        try:
            e = parse_latex(s); return e, disp
        except Exception:
            pass
    norm = normalize_expr(s)
    try:
        e = sym_parse(norm); return e, norm.replace("**","^")
    except Exception:
        return None, disp

# ====== ルート & 健康確認 ======
@app.get("/")
async def root():
    info = {"ok": True,
            "sympy": SYM_AVAILABLE,
            "latex_parser": HAS_PARSE_LATEX,
            "rapid_imported": RAPID_IMPORTED,
            "mathpix_keys": bool(MATHPIX_APP_ID and MATHPIX_APP_KEY),
            "prec_digits": PREC["digits"]}
    logging.info(f"Startup info: {info}")
    return info

@app.get("/calc_test")
async def calc_test(expr: str):
    n = normalize_expr(expr)
    return {"raw": expr, "norm": n}

@app.get("/botinfo")
async def botinfo():
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    async with httpx.AsyncClient(timeout=10) as ac:
        r = await ac.get("https://api.line.me/v2/bot/info", headers=headers)
    return Response(r.text, media_type="application/json", status_code=r.status_code)

@app.get("/envcheck")
async def envcheck():
    tok = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN","")
    sec = os.environ.get("LINE_CHANNEL_SECRET","")
    def mask(s): return f"{len(s)} chars : {s[:6]}...{s[-6:]}" if s else "(empty)"
    return {"access_token": mask(tok), "channel_secret": mask(sec)}

@app.get("/last_error")
async def last_error():
    return LAST_ERROR

# ====== Webhook ======
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

                # 角度モード
                if text.lower().startswith("mode:"):
                    v = text.split(":",1)[1].strip().lower()
                    if v in ("deg","degree","degrees"):
                        ANGLE_MODE["mode"]="deg"
                        await reply_message(reply_token,[{"type":"text","text":"角度モード: Deg"}]); continue
                    if v in ("rad","radian","radians"):
                        ANGLE_MODE["mode"]="rad"
                        await reply_message(reply_token,[{"type":"text","text":"角度モード: Rad"}]); continue
                    await reply_message(reply_token,[{"type":"text","text":"mode:deg / mode:rad"}]); continue

                # 近似桁数
                if text.lower().startswith("prec:"):
                    try:
                        n = int(text.split(":",1)[1].strip())
                        n = max(1, min(50, n))
                        PREC["digits"] = n
                        await reply_message(reply_token,[{"type":"text","text":f"近似表示桁数: {n} 桁"}]); continue
                    except Exception:
                        await reply_message(reply_token,[{"type":"text","text":"prec: の後に 1〜50 の整数を指定してください。"}]); continue

                # 計算
                if text.lower().startswith("calc:"):
                    if not SYM_AVAILABLE:
                        await reply_long_text(reply_token,"Sympy 未導入のため計算不可。requirements.txt に sympy を追加してください。"); continue
                    raw = text[5:].strip()
                    if not raw:
                        await reply_message(reply_token,[{"type":"text","text":"式が空です。例: calc: sin30° + 3^2"}]); continue

                    try:
                        norm = normalize_expr(raw)
                        expr = sym_parse(norm)
                        exact_str = str(sp.simplify(expr)).replace("**","^")
                        approx_str = str(sp.N(expr, PREC["digits"]))
                        shown = str(sp.srepr(expr))
                        guide = cg50_keyseq(exact_str)
                        msg = f"[式]\n{shown}\n厳密: {exact_str}\n近似({PREC['digits']}桁): {approx_str}\n\nfx-CG50 操作ガイド\n{guide}"
                        await reply_long_text(reply_token, msg)
                    except Exception as ex_calc:
                        LAST_ERROR["msg"] = f"{type(ex_calc).__name__}: {ex_calc}"
                        LAST_ERROR["trace"] = traceback.format_exc(limit=5)
                        logging.exception("calc error")
                        await reply_long_text(reply_token, f"解析失敗: {ex_calc}\n入力: {raw}\n正規化: {normalize_expr(raw)}")
                    continue

                await reply_message(reply_token,[{"type":"text","text":"画像を送れば、問題文OCR＋式抽出＋厳密解＋電卓操作まで返します。"}])
                continue

            # ===== 画像 =====
            if msg_type == "image":
                cp = m.get("contentProvider") or {}
                if cp.get("type") == "external" and cp.get("originalContentUrl"):
                    async with httpx.AsyncClient(timeout=30) as ac:
                        r = await ac.get(cp["originalContentUrl"]); r.raise_for_status()
                        img_raw = r.content
                else:
                    img_raw = await get_line_image_bytes(m.get("id"))

                try:
                    bw, gray = preprocess(img_raw)
                except Exception as ex:
                    LAST_ERROR["msg"] = f"preprocess: {ex}"
                    LAST_ERROR["trace"] = traceback.format_exc(limit=5)
                    await reply_long_text(reply_token, f"画像前処理に失敗: {ex}")
                    continue

                text_ocr = rapid_ocr_text(gray)

                expr_blocks: List[str] = []
                if SYM_AVAILABLE:
                    latex_list: List[str] = []
                    if MATHPIX_APP_ID and MATHPIX_APP_KEY:
                        try:
                            jpg = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY),95])[1].tobytes()
                            mp = await ocr_mathpix(jpg)
                            latex_list = extract_latex(mp)
                        except Exception as ex:
                            expr_blocks.append(f"【数式OCR】Mathpix失敗: {ex}")
                    else:
                        expr_blocks.append("【数式OCR】Mathpix未設定のためスキップしました。")

                    answers: List[str] = []
                    for idx, latex in enumerate(latex_list, 1):
                        e_sym, shown = latex_or_text_to_sympy(latex)
                        if e_sym is None:
                            answers.append(f"#{idx}\n抽出: {shown}\n→ 解析不可")
                            continue
                        exact_str = str(sp.simplify(e_sym)).replace("**","^")
                        approx_str = str(sp.N(e_sym, PREC["digits"]))
                        guide = cg50_keyseq(exact_str)
                        answers.append(f"#{idx}\n抽出: {shown}\n厳密: {exact_str}\n近似({PREC['digits']}桁): {approx_str}\n\nfx-CG50 操作ガイド\n{guide}")
                    if answers:
                        expr_blocks.append("\n\n".join(answers))
                else:
                    expr_blocks.append("【計算】Sympy 未導入のため、数式解答は生成できません。")

                head = "📄 問題文（OCR）\n" + (text_ocr if text_ocr else "(検出なし)")
                tail = "\n\n" + ("".join(expr_blocks) if expr_blocks else "（数式抽出なし）")
                await reply_long_text(reply_token, head + tail)
                continue

            await reply_message(reply_token,[{"type":"text","text":f"未対応メッセージタイプ: {msg_type}"}])

        except httpx.HTTPStatusError as he:
            LAST_ERROR["msg"] = f"HTTPStatusError: {he.response.status_code}"
            LAST_ERROR["trace"] = traceback.format_exc(limit=5)
            logging.exception("HTTPStatusError")
            try:
                await reply_message(reply_token,[{"type":"text","text":f"HTTPエラー: {he.response.status_code}"}])
            except Exception:
                pass
        except Exception as ex:
            LAST_ERROR["msg"] = f"{type(ex).__name__}: {ex}"
            LAST_ERROR["trace"] = traceback.format_exc(limit=5)
            logging.exception("Unhandled error")
            try:
                await reply_message(reply_token,[{"type":"text","text":f"内部エラー: {ex}"}])
            except Exception:
                pass

    return JSONResponse({"status":"ok"})
