# main.py  — fx-CG50 対応 LINE Bot（画像→式抽出、二次/二項の解法、番号付き手順）
# - 即時「解析中…」返信（無反応対策）
# - 画像は LINE 保存/外部URL の両対応
# - 画像は可能なら自動拡大・コントラスト強調（Pillow が無い環境でも動作）
# - テキストコマンド:
#     「操作方法」→ キー操作の総合ガイド
#     「二次 1,-3,2」/「二次1 -3 2」/「二次 a=1 b=-3 c=2」→ 解と手順
# - 環境変数: LINE_CHANNEL_ACCESS_TOKEN, OPENAI_API_KEY, （任意）OCR_UPSCALE=2..4
# ------------------------------------------------------------------------------

import os
import io
import re
import json
import math
import base64
import traceback
from typing import List, Dict, Any, Tuple

import httpx
from fastapi import FastAPI, Request

# ------------------------- 環境変数 -------------------------
LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OCR_UPSCALE = int(os.getenv("OCR_UPSCALE", "2"))  # 2~4 を推奨

# ------------------------- FastAPI -------------------------
app = FastAPI()

@app.get("/")
async def root():
    return {"status": "ok"}

# ------------------------- LINE 便利関数 -------------------------
def line_headers(json_type: bool = True) -> Dict[str, str]:
    h = {"Authorization": f"Bearer {LINE_TOKEN}"}
    if json_type:
        h["Content-Type"] = "application/json"}
    return h

async def line_reply(reply_token: str, messages: List[Dict[str, Any]]):
    url = "https://api.line.me/v2/bot/message/reply"
    body = {"replyToken": reply_token, "messages": messages[:5]}
    async with httpx.AsyncClient(timeout=30) as ac:
        r = await ac.post(url, headers=line_headers(), json=body)
        r.raise_for_status()

async def line_push(user_id: str, messages: List[Dict[str, Any]]):
    url = "https://api.line.me/v2/bot/message/push"
    body = {"to": user_id, "messages": messages[:5]}
    async with httpx.AsyncClient(timeout=30) as ac:
        r = await ac.post(url, headers=line_headers(), json=body)
        r.raise_for_status()

# ------------------------- 画像取得（LINE保存/外部URL両対応） -------------------------
async def get_line_image_bytes_from_event(message: dict) -> bytes:
    import httpx
    cprov = message.get("contentProvider", {}) or {}
    if cprov.get("type", "line") == "line":
        # LINE サーバ内
        url = f"https://api-data.line.me/v2/bot/message/{message['id']}/content"
        async with httpx.AsyncClient(timeout=30) as ac:
            r = await ac.get(url, headers=line_headers(json_type=False))
            r.raise_for_status()
            return r.content
    else:
        # 外部URL（転送など）
        url = cprov.get("originalContentUrl")
        if not url:
            raise RuntimeError("画像URLが取得できませんでした")
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as ac:
            r = await ac.get(url)
            r.raise_for_status()
            return r.content

# ------------------------- 画像の軽い前処理（任意/Pillow無しでもOK） -------------------------
def preprocess_for_ocr(img_bytes: bytes) -> bytes:
    try:
        from PIL import Image, ImageOps, ImageFilter
    except Exception:
        # Pillow が無ければそのまま返す
        return img_bytes

    try:
        with Image.open(io.BytesIO(img_bytes)) as im:
            # グレースケール → 自動コントラスト → 少しシャープ
            g = ImageOps.grayscale(im)
            g = ImageOps.autocontrast(g)
            # 可能なら拡大（2〜4倍）
            scale = max(1, min(4, OCR_UPSCALE))
            if scale > 1:
                g = g.resize((g.width * scale, g.height * scale))
            g = g.filter(ImageFilter.SHARPEN)
            out = io.BytesIO()
            g.save(out, format="JPEG", quality=92)
            return out.getvalue()
    except Exception:
        return img_bytes

# ------------------------- Vision: 画像→問題抽出(JSON) -------------------------
async def vision_extract_problems(img_bytes: bytes) -> List[Dict[str, Any]]:
    """
    OpenAI Vision に画像を渡して、最大2題の問題を JSON で抽出させる。
    出力例:
      [{"classification":"quadratic","expression":"x^2-3x+2=0", "quadratic":{"a":1,"b":-3,"c":2}},
       {"classification":"binomial","expression":"5 trials p=1/3, P(X=2)", "binomial":{"n":5,"k":2,"p":0.333333}}]
    """
    if not OPENAI_API_KEY:
        return []

    b64 = base64.b64encode(img_bytes).decode()
    data_url = f"data:image/jpeg;base64,{b64}"

    system = (
        "You are an OCR+math parser. Read the photo (Japanese worksheets possible). "
        "Find up to TWO independent math questions on the page. For each, decide a classification "
        "from: 'quadratic' (ax^2+bx+c=0 type), 'binomial' (n trials, prob p, probability that X=k), or 'other'. "
        "If quadratic, extract numeric a,b,c (decimals OK). If binomial, extract n,k,p (decimal p). "
        "Return ONLY a JSON object with key 'problems' (array)."
    )
    user_text = (
        "Return strict JSON. "
        "For quadratic, if equation is like 'a=1,b=-3,c=2', set a=1,b=-3,c=2. "
        "For binomial like '(5 trials) p=1/3, find P(X=2)', set n=5,k=2,p=0.3333333333. "
        "Do not include explanations."
    )

    payload = {
        "model": "gpt-4o-mini",
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]},
        ],
        "temperature": 0,
    }

    try:
        async with httpx.AsyncClient(timeout=60) as ac:
            r = await ac.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                         "Content-Type": "application/json"},
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            obj = json.loads(text)
            probs = obj.get("problems", [])
            if not isinstance(probs, list):
                return []
            # 数は多くても2問に制限
            return probs[:2]
    except Exception as e:
        print("VISION ERROR:", repr(e))
        return []

# ------------------------- 数学ユーティリティ -------------------------
def solve_quadratic(a: float, b: float, c: float) -> Tuple[float, str, Tuple[complex, complex], str]:
    D = b*b - 4*a*c
    if D > 0:
        kind = "異なる2実数解"
    elif D == 0:
        kind = "重解"
    else:
        kind = "虚数解"
    x1 = (-b + complex(D) ** 0.5) / (2*a)
    x2 = (-b - complex(D) ** 0.5) / (2*a)
    expr = f"{a}x^2 + {b}x + {c} = 0"
    return D, kind, (x1, x2), expr

def format_roots(roots: Tuple[complex, complex]) -> str:
    x1, x2 = roots
    def fmt(z: complex) -> str:
        if abs(z.imag) < 1e-12:
            return f"{z.real:.10g}"
        return f"{z.real:.10g} + {z.imag:.10g}i"
    return f"解: x₁={fmt(x1)}, x₂={fmt(x2)}"

# ------------------------- fx-CG50: 番号付き手順 -------------------------
def steps_quadratic_fx(a: float, b: float, c: float) -> str:
    return (
        "【fx-CG50 二次方程式（aX²+bX+c=0）解法の手順】\n"
        "1. [MENU] → 「EQUA（Equation）」を選ぶ\n"
        "2. [F2] Polynomial（多項式）を選ぶ\n"
        "3. 次の画面で次数を『2』にして [EXE]\n"
        "4. a の欄に  a = {a}  と入力 → [EXE]\n"
        "5. b の欄に  b = {b}  と入力 → [EXE]\n"
        "6. c の欄に  c = {c}  と入力 → [EXE]\n"
        "7. 係数の入力がそろったら [EXE] で解を表示（x1, x2）\n"
        "―― 補足 ――\n"
        "・負の数は白キーの [(-)] で入力（例: −3 は [3]→[(-)] ではなく [(-)]→[3]）\n"
        "・[F1]～[F6] は画面下のラベルに対応（別画面では役割が変わります）\n"
    ).format(a=a, b=b, c=c)

def steps_binom_fx(n: int, k: int, p: float) -> str:
    return (
        "【fx-CG50 二項分布 P(X=k) の手順（例）】\n"
        "1. [MENU] → 「RUN-MAT」を選ぶ\n"
        "2. nCk を入力：  [OPTN] → [PROB] → nCr を選択 →  n = {n}, k = {k}\n"
        "3. その結果に  × p^k × (1−p)^(n−k) を続けて入力\n"
        "   → p = {p} を代入して [EXE]\n"
        "（補足）p=1/3 なら [1] [÷] [3] を使うと正確です\n"
    ).format(n=n, k=k, p=p)

def steps_all_keys_guide() -> str:
    return (
        "【fx-CG50 キー操作の総合ガイド】\n"
        "1. [MENU]：アプリアイコンの一覧を開く（EQUA, RUN-MAT など）\n"
        "2. [EQUA]：方程式メニュー。多項式（Polynomial）は [F2]、次数を数字で指定\n"
        "3. [RUN-MAT]：通常計算。nCr, nPr, !（階乗）などは [OPTN] → [PROB]\n"
        "4. [(-)]：負号。−3 は [(-)]→[3] の順で入力\n"
        "5. [EXE]：確定／計算実行。係数入力の各欄や最終確定で必ず押す\n"
        "6. [DEL]/[AC]：入力修正／リセット（[AC] は実行中止に相当）\n"
        "7. [F1]〜[F6]：画面下のラベルに対応（EQUA では [F1]戻る / [F2]Polynomial / など）\n"
    )

# ------------------------- テキスト解析（「二次 ...」） -------------------------
_quad_cmd = re.compile(r"^二次\s*(.+)$")
def parse_quadratic_text(text: str):
    m = _quad_cmd.match(text.strip())
    if not m:
        return None
    tail = m.group(1).strip()

    # a=1 b=-3 c=2 パターン
    m2 = re.findall(r"[abcＡＢＣ]\s*=\s*[-+]?[\d\.]+", tail, flags=re.IGNORECASE)
    if m2 and len(m2) >= 3:
        vals = {}
        for tok in m2:
            k, v = tok.split("=")
            k = k.strip().lower().replace("Ａ", "a").replace("Ｂ", "b").replace("Ｃ", "c")
            vals[k] = float(v)
        if all(k in vals for k in ("a", "b", "c")):
            return vals["a"], vals["b"], vals["c"]

    # 1,-3,2 / 1 -3 2 / 1．-3．2（誤打の全角ピリオドも許容）
    tail = tail.replace("，", ",").replace("．", ".").replace("　", " ")
    parts = re.split(r"[\s,]+", tail)
    nums = []
    for p in parts:
        if p == "":
            continue
        try:
            nums.append(float(p))
        except:
            pass
    if len(nums) >= 3:
        return nums[0], nums[1], nums[2]

    return None

# ------------------------- 画像ハンドラ -------------------------
async def handle_image(user_id: str, reply_token: str, message: dict):
    # 1) 即返信（無反応対策）
    try:
        await line_reply(reply_token, [{"type": "text", "text": "解析中…（数秒お待ちください）"}])
    except Exception as e:
        print("REPLY ERROR:", e)

    try:
        raw = await get_line_image_bytes_from_event(message)
        pre = preprocess_for_ocr(raw)
        problems = await vision_extract_problems(pre)

        if not problems:
            raise RuntimeError("式を特定できませんでした")

        out_msgs = []
        count = 0
        for p in problems:
            if count >= 2:
                break
            cls = p.get("classification", "other")
            expr = p.get("expression", "")

            if cls == "quadratic" and p.get("quadratic"):
                try:
                    a = float(p["quadratic"]["a"])
                    b = float(p["quadratic"]["b"])
                    c = float(p["quadratic"]["c"])
                except Exception:
                    continue
                D, kind, roots, expr2 = solve_quadratic(a, b, c)
                out_msgs += [
                    {"type": "text",
                     "text": f"【抽出(二次)】{expr or expr2}\n a={a}, b={b}, c={c}\n判別式 D={D:.10g} → {kind}\n{format_roots(roots)}"},
                    {"type": "text", "text": steps_quadratic_fx(a, b, c)}
                ]
                count += 1

            elif cls == "binomial" and p.get("binomial"):
                try:
                    n = int(p["binomial"]["n"])
                    k = int(p["binomial"]["k"])
                    p_ = float(p["binomial"]["p"])
                except Exception:
                    continue
                val = math.comb(n, k) * (p_ ** k) * ((1 - p_) ** (n - k))
                out_msgs += [
                    {"type": "text",
                     "text": f"【抽出(二項)】{expr}\nP(X={k}) = C({n},{k})·p^{k}(1-p)^{n-k} = {val:.10g}"},
                    {"type": "text", "text": steps_binom_fx(n, k, p_)}
                ]
                count += 1

        if not out_msgs:
            raise RuntimeError("対応分類が見つかりませんでした")

        try:
            await line_push(user_id, out_msgs[:5])
        except Exception as e:
            print("PUSH ERROR -> fallback reply:", e)
            await line_reply(reply_token, [out_msgs[0]])

    except Exception as e:
        print("IMAGE FLOW ERROR:", e)
        advice = (
            "式を特定できませんでした。\n"
            "できれば：正面／影を避ける／余白少なめ／濃いモード。\n"
            "今すぐ試せる入力：\n"
            "・二次 1,-3,2  や  二次 a=1 b=-3 c=2\n"
            "・キー操作一覧： 操作方法"
        )
        try:
            await line_reply(reply_token, [{"type": "text", "text": advice}])
        except:
            await line_push(user_id, [{"type": "text", "text": advice}])

# ------------------------- テキストハンドラ -------------------------
def usage_text() -> str:
    return (
        "使い方：\n"
        "1) 問題の写真を送る → 解析後『式＋答え＋番号付き手順』（最大2問）\n"
        "2) 二次： 例）二次 a=1 b=-3 c=2 ／ 例）二次 1,-3,2 ／ 例）二次1 -3 2\n"
        "3) キー操作の一覧： 『操作方法』\n"
    )

async def handle_text(user_id: str, reply_token: str, text: str):
    t = text.strip()

    # 総合ガイド
    if t in ("操作方法", "ヘルプ", "help"):
        await line_reply(reply_token, [{"type": "text", "text": steps_all_keys_guide()}])
        return

    # 二次コマンド
    quad = parse_quadratic_text(t)
    if quad:
        a, b, c = quad
        try:
            a = float(a); b = float(b); c = float(c)
        except Exception:
            await line_reply(reply_token, [{"type": "text", "text": "係数が読めませんでした。例）二次 1,-3,2"}])
            return
        D, kind, roots, expr = solve_quadratic(a, b, c)
        msgs = [
            {"type": "text",
             "text": f"式: {expr}\n判別式 D={D:.10g} → {kind}\n{format_roots(roots)}"},
            {"type": "text", "text": steps_quadratic_fx(a, b, c)}
        ]
        await line_reply(reply_token, msgs)
        return

    # キーワード
    if t in ("使い方", "つかいかた", "howto"):
        await line_reply(reply_token, [{"type": "text", "text": usage_text()}])
        return

    # 既定
    await line_reply(reply_token, [{"type": "text", "text": usage_text()}])

# ------------------------- Webhook -------------------------
@app.post("/webhook")
async def webhook(request: Request):
    body = await request.json()
    try:
        events = body.get("events", [])
        for ev in events:
            typ = ev.get("type")
            src = ev.get("source", {})
            user_id = src.get("userId")
            msg = ev.get("message", {}) or {}
            reply_token = ev.get("replyToken")

            if typ == "message":
                mtype = msg.get("type")
                if mtype == "text":
                    await handle_text(user_id, reply_token, msg.get("text", ""))
                elif mtype == "image":
                    await handle_image(user_id, reply_token, msg)  # message 全体を渡す
                else:
                    await line_reply(reply_token, [{"type": "text", "text": usage_text()}])
    except Exception as e:
        print("WEBHOOK ERROR:", e)
        print(traceback.format_exc())
    return {"ok": True}

# ------------------------- ローカル起動用 -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
