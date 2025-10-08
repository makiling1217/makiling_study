# main.py  ←まるごと差し替え

import os, io, re, base64, asyncio
from typing import List, Optional
from fastapi import FastAPI, Request
import httpx
from PIL import Image, ImageOps, ImageEnhance
from openai import OpenAI

app = FastAPI()

LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

client_ai = OpenAI(api_key=OPENAI_KEY)

LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
LINE_PUSH_URL  = "https://api.line.me/v2/bot/message/push"
LINE_CONTENT   = "https://api-data.line.me/v2/bot/message/{mid}/content"  # ← api-data が正解

# ---------- LINE 送信ユーティリティ ----------
async def line_reply(reply_token: str, messages: List[dict]):
    async with httpx.AsyncClient(timeout=10) as cli:
        r = await cli.post(
            LINE_REPLY_URL,
            headers={"Authorization": f"Bearer {LINE_TOKEN}"},
            json={"replyToken": reply_token, "messages": messages},
        )
        print("LINE reply status:", r.status_code, await _safe_text(r))

async def line_push(user_id: str, messages: List[dict]):
    async with httpx.AsyncClient(timeout=10) as cli:
        r = await cli.post(
            LINE_PUSH_URL,
            headers={"Authorization": f"Bearer {LINE_TOKEN}"},
            json={"to": user_id, "messages": messages},
        )
        print("LINE push status:", r.status_code, await _safe_text(r))

async def _safe_text(r: httpx.Response) -> str:
    try:
        return r.text[:300]
    except Exception:
        return "<no body>"

# ---------- 共通メッセージ ----------
GUIDE_OPERATIONS = (
    "【fx-CG50 操作キーの総合ガイド】\n"
    "1) 負号: 青い「(ー)」キー（白い「-」は引き算）\n"
    "2) 小数: 「.」\n"
    "3) 決定: 「EXE」\n"
    "4) カーソル: 十字キー（↑↓←→）\n"
    "5) 取消: 「DEL」/ 全消去「AC/ON」\n"
    "6) EQUATION: アイコンメニューで『EQUATION』を選択\n"
    "7) EQUATION内: F1 一次, F2 二次, F3 三元一次, F4 四次, F5 多項式, F6 戻る\n"
    "8) 二次 ax²+bx+c=0: a を入力→EXE→b→EXE→c→EXE\n"
)

# ---------- 文章コマンド（テキスト） ----------
def parse_quadratic(text: str) -> Optional[tuple]:
    txt = text.strip().replace("，", ",").replace("．", ".").replace("、", ",").replace("　", " ")
    if not txt.startswith("二次"):
        return None

    # パターン a=1 b=-3 c=2
    m = re.search(r"二次.*?a\s*=\s*([+-]?\d+(?:\.\d+)?)\s*b\s*=\s*([+-]?\d+(?:\.\d+)?)\s*c\s*=\s*([+-]?\d+(?:\.\d+)?)", txt)
    if m:
        a, b, c = map(float, m.groups())
        return a, b, c

    # 区切り文字（カンマ/空白/中点/ピリオド）を全部許容
    payload = re.sub(r"^二次", "", txt).strip()
    tokens = re.split(r"[,\s・]+", payload)
    nums = []
    for t in tokens:
        if t == "":
            continue
        # 「1.-3.2」など dot 区切りに対応（小数と衝突しにくい判定）
        parts = re.split(r"(?<!\d)\.(?!\d)", t)  # 数字に挟まれないピリオドだけを区切りとして扱う
        for p in parts:
            p = p.strip()
            if p == "":
                continue
            try:
                nums.append(float(p))
            except ValueError:
                pass
    if len(nums) >= 3:
        return nums[0], nums[1], nums[2]
    return None

def explain_quadratic(a: float, b: float, c: float) -> str:
    D = b*b - 4*a*c
    if D > 0:
        kind = "実数解（異なる2解）"
    elif D == 0:
        kind = "実数解（重解）"
    else:
        kind = "複素数解（共役）"

    lines = []
    lines.append(f"【式】{a}x² + {b}x + {c} = 0")
    lines.append(f"【判別式】D = b²-4ac = {b}² - 4×{a}×{c} = {D} → {kind}")
    lines.append("【番号付き手順（電卓）】")
    lines += [
        "1) アイコンメニューで『EQUATION』→F2（二次）",
        "2) 形式『ax²+bx+c=0』を確認",
        f"3) aに「{a}」→EXE",
        f"4) bに「{b}」→EXE",
        f"5) cに「{c}」→EXE",
        "6) 解が表示される（x1, x2）/ 確定は EXE",
    ]
    # 解も計算（複素数も文字で）
    import cmath
    x1 = (-b + cmath.sqrt(D)) / (2*a)
    x2 = (-b - cmath.sqrt(D)) / (2*a)
    def fmt(z):
        if abs(z.imag) < 1e-10:
            return f"{z.real:.10g}"
        return f"{z.real:.10g} + {z.imag:.10g}i"
    lines.insert(1, f"【解】x1 = {fmt(x1)} , x2 = {fmt(x2)}")
    return "\n".join(lines)

# ---------- 画像取得 & 前処理 ----------
async def fetch_line_image_bytes(message_id: str) -> bytes:
    url = LINE_CONTENT.format(mid=message_id)  # ← api-data
    headers = {"Authorization": f"Bearer {LINE_TOKEN}"}
    async with httpx.AsyncClient(timeout=30) as cli:
        r = await cli.get(url, headers=headers)
        r.raise_for_status()
        return r.content

def preprocess_for_ocr(data: bytes) -> str:
    """PILで拡大・グレースケール・コントラスト強調して base64 data URL にして返す"""
    img = Image.open(io.BytesIO(data))
    img = img.convert("L")  # グレースケール
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Sharpness(img).enhance(1.6)
    # 画像が小さければ強めに拡大（A4の写真対策）
    w, h = img.size
    scale = 3 if max(w, h) < 1400 else 2
    img = img.resize((w*scale, h*scale), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"

async def vision_solve_math(image_data_url: str) -> str:
    """
    画像から最大2問の式を特定して、
    『式』『解』『番号付き手順（電卓）』の順で返すように指示。
    """
    prompt = (
        "あなたは CASIO fx-CG50 の操作に詳しい家庭教師です。"
        "画像から数学の問題文を最大2問だけ読み取り、各問について必ず次の順に日本語で返答してください。\n"
        "【式】… / 【解】… / 【番号付き手順（電卓）】1) … 2) … と番号付き。"
        "二次方程式なら fx-CG50 での解き方（EQUATION→F2→a,b,c入力→EXE）を説明。"
        "線形/連立/三角/指数/文章題などは、fx-CG50で現実的に入力する方法を簡潔に案内してください。"
        "式が読み取れない場合は『式を特定できませんでした』と書き、撮影改善のコツを1行で。"
    )
    rsp = client_ai.responses.create(
        model="gpt-4o-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": image_data_url},
            ],
        }],
        max_output_tokens=800,
    )
    return rsp.output_text

# ---------- Webhook ----------
@app.post("/webhook")
async def webhook(request: Request):
    body = await request.json()
    print("WEBHOOK:", body)
    events = body.get("events", [])
    if not events:
        return {"ok": True}
    ev = events[0]
    msg = ev.get("message", {})
    typ = msg.get("type")
    reply_token = ev.get("replyToken")
    user_id = (ev.get("source") or {}).get("userId", "")

    # テキストコマンド
    if typ == "text":
        text = msg.get("text", "").strip()
        # 1) 操作方法
        if text == "操作方法":
            await line_reply(reply_token, [{"type": "text", "text": GUIDE_OPERATIONS}])
            return {"ok": True}
        # 2) 二次…
        quad = parse_quadratic(text)
        if quad:
            a, b, c = quad
            await line_reply(reply_token, [{"type": "text", "text": "解析中… 少し待ってね。"}])
            out = explain_quadratic(a, b, c)
            await line_push(user_id, [{"type": "text", "text": out}])
            return {"ok": True}

        # それ以外はヘルプ
        help_text = (
            "使い方：\n"
            "1) 問題の写真を送る → 画像解析して『式＋解＋番号付き手順』（最大2問）\n"
            "2) 係数で直接： 例）二次 a=1 b=-3 c=2 / 例）二次 1,-3,2 / 例）二次 1 -3 2\n"
            "3) キー操作の一覧：『操作方法』"
        )
        await line_reply(reply_token, [{"type": "text", "text": help_text}])
        return {"ok": True}

    # 画像メッセージ
    if typ == "image":
        mid = msg.get("id")
        # まず即時返信（タイムアウト防止）
        await line_reply(reply_token, [{"type": "text", "text": "解析中… 少し待ってね。"}])

        async def _do():
            try:
                raw = await fetch_line_image_bytes(mid)  # ← 404対策済み
                data_url = preprocess_for_ocr(raw)      # ← 拡大・強調
                text = await vision_solve_math(data_url)
                # 長いと分割して送る
                chunks = _split_text(text, 480)
                for ch in chunks:
                    await line_push(user_id, [{"type": "text", "text": ch}])
            except httpx.HTTPStatusError as e:
                msg = (
                    "画像を取得できませんでした（LINE画像URL 404/401）。もう一度撮影して送ってください。"
                )
                print("image flow error:", e)
                await line_push(user_id, [{"type": "text", "text": msg}])
            except Exception as e:
                print("vision error:", repr(e))
                fallback = (
                    "式を特定できませんでした。A4は『正面・影なし・濃いめ』で撮り直し。"
                    "どうしても難しい場合は『二次 a=1 b=-3 c=2』のように式を文字で送ってください。"
                )
                await line_push(user_id, [{"type": "text", "text": fallback}])

        asyncio.create_task(_do())
        return {"ok": True}

    # その他のタイプは無視
    await line_reply(reply_token, [{"type": "text", "text": "テキストか画像で送ってください。"}])
    return {"ok": True}

def _split_text(s: str, n: int) -> List[str]:
    out = []
    while s:
        out.append(s[:n])
        s = s[n:]
    return out

@app.get("/")
async def root():
    return {"status": "ok"}
