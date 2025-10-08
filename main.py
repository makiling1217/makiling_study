import os, io, re, json, math, base64, asyncio
from typing import List, Tuple, Optional

import httpx
from fastapi import FastAPI, Request
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

# ---------- 環境変数 ----------
LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")  # gpt-4o でもOK

# ---------- OpenAI ----------
from openai import OpenAI
oa_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- FastAPI ----------
app = FastAPI()

# ---------- LINE 送受信用 ----------
LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
LINE_PUSH_URL  = "https://api.line.me/v2/bot/message/push"
LINE_CONTENT_URL = "https://api-data.line.me/v2/bot/message/{message_id}/content"

def line_headers(json_type=True):
    h = {"Authorization": f"Bearer {LINE_TOKEN}"}
    if json_type:
        h["Content-Type"] = "application/json"
    return h

async def line_reply(reply_token: str, messages: List[dict]):
    payload = {"replyToken": reply_token, "messages": messages[:5]}
    async with httpx.AsyncClient(timeout=20) as ac:
        r = await ac.post(LINE_REPLY_URL, headers=line_headers(), json=payload)
        r.raise_for_status()

async def line_push(user_id: str, messages: List[dict]):
    payload = {"to": user_id, "messages": messages[:5]}
    async with httpx.AsyncClient(timeout=20) as ac:
        r = await ac.post(LINE_PUSH_URL, headers=line_headers(), json=payload)
        r.raise_for_status()

async def get_line_image_bytes(message_id: str) -> bytes:
    # 404 の原因は api.line.me ではなく api-data.line.me を使う必要があるため
    url = LINE_CONTENT_URL.format(message_id=message_id)
    async with httpx.AsyncClient(timeout=30) as ac:
        r = await ac.get(url, headers=line_headers(json_type=False))
        r.raise_for_status()
        return r.content

# ---------- 画像前処理（A4でも読めるよう強化） ----------
def preprocess_for_ocr(img_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(img_bytes))
    img = ImageOps.exif_transpose(img)  # 回転補正

    # 200% 以上に拡大（短辺が 1600px 未満なら 2200px まで）
    w, h = img.size
    target_short = 2200
    scale = max(1.0, target_short / min(w, h))
    if scale > 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # グレースケール → 自動コントラスト → 強コントラスト → シャープ
    g = img.convert("L")
    g = ImageOps.autocontrast(g)
    g = ImageEnhance.Contrast(g).enhance(1.8)
    g = ImageEnhance.Sharpness(g).enhance(1.3)

    # ほんのり二値化（弱め）で文字を立たせる
    g = g.point(lambda x: 0 if x < 170 else 255)

    out = io.BytesIO()
    g.save(out, format="JPEG", quality=95)
    return out.getvalue()

# ---------- fx-CG50: 汎用キーガイド（番号付き / EXE必須） ----------
def general_key_guide() -> str:
    return (
        "【fx-CG50 キー操作の総合ガイド】\n"
        "1) 電源ON：右上 [AC/ON]\n"
        "2) メニュー： [MENU]\n"
        "3) 実行： [EXE]\n"
        "4) 取り消し： [DEL] / 1つ戻る： [EXIT]\n"
        "5) 小数⇔分数： [S⇔D]\n"
        "6) べき乗： [^]（例 3^2）\n"
        "7) サブメニュー（画面上のFキー）：[F1]～[F6]\n"
        "   例）EQUATION→POLYNOMIAL→DEGREE 2 などで [F1]～[F6] を使います\n"
        "8) 負の数： [(-)]（マイナスはこのキー）\n"
        "9) 分数： [a□/b□] テンプレート or `(`/`)` と `÷`\n"
    )

# ---------- 二次方程式：解析 & 手順 ----------
def parse_quadratic(text: str) -> Optional[Tuple[float, float, float]]:
    t = text.strip()
    # 全角を半角へ
    trans = str.maketrans({
        "，": ",", "．": ".", "　": " ",
        "−": "-", "ー": "-", "―": "-",
        "＋": "+", "／": "/", "＊": "*"
    })
    t = t.translate(trans)

    if "二次" not in t:
        return None

    # パターン1: 二次 1,-3,2 / 二次1,-3,2 / 二次 1 -3 2
    m = re.search(
        r"二次\s*([+\-]?\d+(?:\.\d+)?)\s*(?:,|\s)\s*([+\-]?\d+(?:\.\d+)?)\s*(?:,|\s)\s*([+\-]?\d+(?:\.\d+)?)",
        t
    )
    if m:
        a, b, c = map(float, m.groups())
        return a, b, c

    # パターン2: 二次 a=1 b=-3 c=2（順不同対応）
    named = dict(re.findall(r"\b([abc])\s*=\s*([+\-]?\d+(?:\.\d+)?)", t))
    if all(k in named for k in ("a", "b", "c")):
        return float(named["a"]), float(named["b"]), float(named["c"])

    return None

def solve_quadratic(a: float, b: float, c: float):
    D = b*b - 4*a*c
    steps = []
    steps.append(f"判別式 D = b^2 - 4ac = {b}^2 - 4×{a}×{c} = {D}")
    if D > 0:
        r1 = (-b + math.sqrt(D)) / (2*a)
        r2 = (-b - math.sqrt(D)) / (2*a)
        kind = "2つの実数解"
        roots = (r1, r2)
    elif D == 0:
        r = (-b) / (2*a)
        kind = "重解"
        roots = (r, r)
    else:
        kind = "虚数解（実数解なし）"
        r1 = complex(-b,  math.sqrt(-D)) / (2*a)
        r2 = complex(-b, -math.sqrt(-D)) / (2*a)
        roots = (r1, r2)
    return D, kind, roots, steps

def steps_quadratic_fx(a: float, b: float, c: float) -> str:
    # EQUATION 経由の手順（EXE明記）
    return (
        "【fx-CG50：二次方程式の解（EQUATION）】\n"
        "1) [MENU] → [F1]EQUATION を選び [EXE]\n"
        "2) [F1]POLYNOMIAL（多項式）を押して [EXE]\n"
        "3) DEGREE（次数）で [F2] 2 を選び [EXE]\n"
        f"4) 係数 a に「{a}」を入力して [EXE]\n"
        f"5) 係数 b に「{b}」を入力して [EXE]\n"
        f"6) 係数 c に「{c}」を入力して [EXE]\n"
        "7) 解 x1, x2 が表示されます（[S⇔D]で分数/小数切替）\n"
        "\n【RUN•MAT 直打ち（早い）】\n"
        "1) [MENU] → RUN•MAT → \n"
        "   例）`10×(1/3)^3×(2/3)^2` のように式を入力して [EXE]\n"
    )

def format_roots(roots):
    def f(x):
        if isinstance(x, complex):
            return str(x)
        # 分数表示は S⇔D に任せるので小数は丸め
        return f"{x:.10g}"
    return f"x₁ = {f(roots[0])},  x₂ = {f(roots[1])}"

# ---------- 二項分布：手順（本問の(8)(9) 用） ----------
def steps_binom_fx(n: int, k: int, p: float) -> str:
    return (
        "【fx-CG50：二項分布 Bpd（ちょうど k 個）】\n"
        "1) [MENU] → [STAT] → [F5]DIST → [F5]BINM → [F1]Bpd を [EXE]\n"
        "2) [F2]Variable を選んで [EXE]\n"
        f"3) n={n}, p={p}, x={k} を入力して [EXE]\n"
        "4) P(X=x) が表示されます（[S⇔D]で分数/小数切替）\n"
    )

# ---------- Vision 解析 ----------
async def vision_extract_problems(img_bytes: bytes) -> Optional[list]:
    b64 = base64.b64encode(img_bytes).decode()
    prompt = (
        "次の写真から **最大2問** の数学問題を抽出して、以下のJSONだけを返してください。"
        "出力は日本語でも英語でも構いません。コメントは禁止。"
        "構造:\n"
        "{"
        "\"problems\":["
        "  {"
        "    \"classification\":\"quadratic|binomial|other\","
        "    \"expression\":\"<問題文の要点/式>\","
        "    \"quadratic\": {\"a\":?,\"b\":?,\"c\":?} (二次なら),"
        "    \"binomial\": {\"n\":?,\"k\":?,\"p\":?} (二項なら)"
        "  }"
        "]}"
    )
    try:
        resp = oa_client.responses.create(
            model=VISION_MODEL,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}
                ]
            }],
            max_output_tokens=700
        )
        txt = resp.output_text
        data = json.loads(txt)
        return data.get("problems", None)
    except Exception as e:
        print("Vision error:", e)
        return None

# ---------- テキスト指令の分岐 ----------
async def handle_text(user_id: str, reply_token: str, text: str):
    t = text.strip()

    # 1) 操作方法（総合ガイド）
    if "操作方法" in t:
        await line_reply(reply_token, [{"type":"text","text": general_key_guide()}])
        return

    # 2) 二次 方程式（多様な書式を許容）
    abc = parse_quadratic(t)
    if abc:
        a,b,c = abc
        D, kind, roots, detail = solve_quadratic(a,b,c)
        msg = []
        msg.append({"type":"text","text":
            f"【二次方程式】a={a}, b={b}, c={c}\n判別式D={D} → {kind}\n{format_roots(roots)}"})
        msg.append({"type":"text","text": steps_quadratic_fx(a,b,c)})
        await line_reply(reply_token, msg[:1])  # まず1通
        if len(msg)>1:
            await line_push(user_id, msg[1:])
        return

    # 3) 使い方ガイド
    howto = (
        "使い方：\n"
        "1) 問題の写真を送る → 解析後「式＋答え＋番号付き手順」を返します（最大2問）。\n"
        "2) 係数で直接：例）「二次 1,-3,2」「二次1 -3 2」「二次 a=1 b=-3 c=2」\n"
        "3) キー操作の一覧：『操作方法』と送信\n"
    )
    await line_reply(reply_token, [{"type":"text","text": howto}])

# ---------- 画像メッセージ ----------
async def handle_image(user_id: str, reply_token: str, message_id: str):
    # まず即時返信（無返信防止）
    await line_reply(reply_token, [{"type":"text","text":"解析中…（数秒お待ちください）"}])

    try:
        raw = await get_line_image_bytes(message_id)
        pre = preprocess_for_ocr(raw)
        problems = await vision_extract_problems(pre)

        if not problems:
            raise RuntimeError("式を特定できませんでした")

        out_msgs = []
        count = 0
        for p in problems:
            if count >= 2:
                break
            cls = p.get("classification","other")
            expr = p.get("expression","")
            if cls == "quadratic" and p.get("quadratic"):
                a = float(p["quadratic"]["a"])
                b = float(p["quadratic"]["b"])
                c = float(p["quadratic"]["c"])
                D, kind, roots, _ = solve_quadratic(a,b,c)
                out_msgs.append({"type":"text","text":
                    f"【抽出(二次)】{expr}\n a={a}, b={b}, c={c}\n判別式D={D} → {kind}\n{format_roots(roots)}"})
                out_msgs.append({"type":"text","text": steps_quadratic_fx(a,b,c)})
                count += 1
            elif cls == "binomial" and p.get("binomial"):
                n = int(p["binomial"]["n"]); k = int(p["binomial"]["k"]); p_ = float(p["binomial"]["p"])
                comb = math.comb(n,k)
                val = comb*(p_**k)*((1-p_)**(n-k))
                out_msgs.append({"type":"text","text":
                    f"【抽出(二項)】{expr}\n P(X={k})=C({n},{k})·p^{k}(1-p)^{n-k}={comb}×{p_}^{k}×{1-p_}^{n-k} = {val:.10g}"})
                out_msgs.append({"type":"text","text": steps_binom_fx(n,k,p_)})
                count += 1

        if not out_msgs:
            raise RuntimeError("対応分類が見つかりませんでした")

        # 解析結果を push（reply は上で済み）
        # 分割しても 5 通まで
        await line_push(user_id, out_msgs[:5])

    except Exception as e:
        print("IMAGE FLOW ERROR:", e)
        advice = (
            "式を特定できませんでした。\n"
            "・写真はA4全体でOK。こちらで自動拡大/強調しています。\n"
            "・それでも難しい場合は、係数で送ってください：\n"
            "  例）「二次 1,-3,2」 または 「二次 a=1 b=-3 c=2」\n"
            "・キー操作の一覧は『操作方法』と送信してください。"
        )
        await line_push(user_id, [{"type":"text","text": advice}])

# ---------- ルーティング ----------
@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/webhook")
async def webhook(request: Request):
    body = await request.json()
    events = body.get("events", [])
    for ev in events:
        t = ev.get("type")
        src = ev.get("source", {})
        user_id = src.get("userId")
        if t == "message":
            msg = ev.get("message", {})
            mtype = msg.get("type")
            reply_token = ev.get("replyToken")
            if mtype == "text":
                text = msg.get("text", "")
                await handle_text(user_id, reply_token, text)
            elif mtype == "image":
                message_id = msg.get("id")
                await handle_image(user_id, reply_token, message_id)
            else:
                await line_reply(reply_token, [{"type":"text","text":"テキストか画像で送ってください。"}])
    return {"ok": True}
