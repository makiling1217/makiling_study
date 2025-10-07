# main.py
# fx-CG50 向け：LINE 受信 → テキスト/画像を解析し、式・解・番号付き手順を返信
# 変更点：
# - 画像取得URLを api-data.line.me に修正（404対策）
# - Visionへは base64 の data URL で送信（"Expecting value"対策）
# - A4の小さい文字対策で簡易前処理（拡大+コントラスト）
# - 「二次 1,-3,2」「二次1,-3,2」「二次1 -3 2」「二次 a=… b=… c=…」等を全部パース
# - 画像からは最大2問の式をJSONで抽出→各々に「式／答え／番号付き手順」を返す
# - 「操作方法」で総合ガイド（F1〜F6含む）を返す

from fastapi import FastAPI, Request
import os, json, re, math, cmath, base64, io
import httpx
from PIL import Image, ImageEnhance
from typing import List, Dict, Any

# ==== 環境変数 ====
LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

app = FastAPI()


# ---------- 共通：LINE返信 ----------
async def line_reply(reply_token: str, texts: List[str]):
    if not LINE_TOKEN:
        return
    # 長文は分割
    messages = [{"type": "text", "text": t[:4900]} for t in texts]
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {"Authorization": f"Bearer {LINE_TOKEN}"}
    payload = {"replyToken": reply_token, "messages": messages}
    async with httpx.AsyncClient(timeout=20) as cli:
        r = await cli.post(url, headers=headers, json=payload)
        print("LINE reply status:", r.status_code, r.text[:200])


# ---------- 使い方 ----------
HELP_TEXT = (
    "使い方：\n"
    "1) 問題の写真を送る → 解析して『式＋答え＋番号付き手順』（最大2問）\n"
    "2) 文字で係数：\n"
    "   例) 二次 1 -3 2 / 二次 1,-3,2 / 二次1,-3,2 / 二次 a=1 b=-3 c=2\n"
    "3) キー操作の一覧：『操作方法』\n"
)

# ---------- fx-CG50：番号付き手順（タイプ別） ----------
def steps_quadratic(a: float, b: float, c: float) -> str:
    # 負号入力の注意やFキー凡例も含める
    return (
        "【fx-CG50 二次方程式の解き方（EQUATION）】\n"
        "1. [MENU] → 『EQUA (Equation)』を選択 → [EXE]\n"
        "2. 画面下の[F2]『POLY』(多項式) を選択 → 次へ\n"
        "3. 次の画面で次数『2』を選択（必要なら[F1]〜[F6]の表示に合わせて選ぶ）\n"
        f"4. 係数を入力： a={a} → [EXE] → b={b} → [EXE] → c={c} → [EXE]\n"
        "   ※負の数は［(−)］キー（x10^xの左）で入力。引き算の(−)とは別です。\n"
        "5. 解が表示されたら、[▼/▲] で x₁, x₂ を確認。\n"
        "6. 分数/小数の切替： [SHIFT]→[S⇔D] / 複素数を許可： [SHIFT]→[SETUP]→『Complex: a+bi』\n"
        "7. 画面下の機能キー（例）：[F1]戻る / [F2]POLY / [F3]SIML / [F4]DEG / [F5]COEF / [F6]SOLV\n"
        "   ※実際のラベルは画面下に表示される内容に従ってください。\n"
    )

def steps_linear(a: float, b: float) -> str:
    return (
        "【fx-CG50 一次方程式 ax+b=0 の解き方（EQUATION）】\n"
        "1. [MENU] → 『EQUA (Equation)』→ [EXE]\n"
        "2. [F2]『POLY』→ 次数『1』を選択\n"
        f"3. 係数を入力： a={a} → [EXE] → b={b} → [EXE]\n"
        "   ※負の数は［(−)］キーで入力\n"
        "4. 表示された解を確認（[S⇔D]で分数/小数切替、複素は[SETUP]で a+bi）\n"
        "5. 画面下の[F1]〜[F6]は画面表示のラベルに従って操作。\n"
    )

GENERAL_OPERATIONS = (
    "【fx-CG50 キー操作の総合ガイド】\n"
    "1. 負の数入力：‘−3’は［(−)］キーを使う（引き算の[−]とは別）\n"
    "2. 分数/小数の切替：[SHIFT]→[S⇔D]\n"
    "3. 角度設定：[SHIFT]→[SETUP]→『Angle: Deg/Rad』\n"
    "4. 複素数を許可：[SHIFT]→[SETUP]→『Complex: a+bi』\n"
    "5. EQUA内の種類：下段の[F1]〜[F6]（画面の表示ラベルに従う）\n"
    "6. クリア：[AC/ON]（式欄で長押しで消去/復帰）\n"
    "7. よくある[Fキー]の例：\n"
    "   F1:戻る / F2:POLY / F3:SIML / F4:DEG選択 / F5:COEF（係数編集）/ F6:SOLV（解を出す）\n"
)

# ---------- 数式 → 解 ----------
def solve_quadratic(a: float, b: float, c: float) -> Dict[str, Any]:
    if abs(a) < 1e-12:
        # 退避：一次として扱う
        return solve_linear(b, c) | {"as_linear": True}
    D = b*b - 4*a*c
    kind = "D>0（2実数解）" if D > 1e-12 else ("D=0（重解）" if abs(D) <= 1e-12 else "D<0（虚数解）")
    sqrtD = cmath.sqrt(D)
    x1 = (-b + sqrtD) / (2*a)
    x2 = (-b - sqrtD) / (2*a)
    def fmt(z):
        if abs(z.imag) < 1e-12:
            return f"{z.real:.10g}"
        return f"{z.real:.10g} + {z.imag:.10g}i"
    return {
        "equation": f"{a}x^2 + {b}x + {c} = 0",
        "discriminant": D,
        "kind": kind,
        "roots": [fmt(x1), fmt(x2)],
    }

def solve_linear(a: float, b: float) -> Dict[str, Any]:
    if abs(a) < 1e-12:
        return {"equation": f"{a}x + {b} = 0", "kind": "解なし（a=0）", "roots": []}
    x = -b / a
    return {"equation": f"{a}x + {b} = 0", "kind": "一次方程式", "roots": [f"{x:.10g}"]}

# ---------- テキスト：『二次 …』 の強力パーサ ----------
NUM = r"[-+]?\d+(?:\.\d+)?"
def parse_quadratic_command(text: str):
    # いろいろな表記を許容
    s = text.replace("，", ",").replace("　", " ").strip()
    if not s.startswith("二次"):
        return None
    # a=1 b=-3 c=2
    m = re.search(r"a\s*=\s*("+NUM+").*?b\s*=\s*("+NUM+").*?c\s*=\s*("+NUM+")", s)
    if m:
        return tuple(float(x) for x in m.groups())
    # 二次 1,-3,2 / 二次1,-3,2 / 二次 1 -3 2 / 二次1 -3 2
    m = re.search(r"二次[\s]*("+NUM+")[\s,]+("+NUM+")[\s,]+("+NUM+")", s)
    if m:
        return tuple(float(x) for x in m.groups())
    return None

# ---------- 画像取得（LINE） ----------
async def fetch_line_image(message_id: str) -> bytes:
    # 404対策：content は api-data ドメイン
    url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
    headers = {"Authorization": f"Bearer {LINE_TOKEN}"}
    async with httpx.AsyncClient(timeout=30) as cli:
        r = await cli.get(url, headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"content get failed: {r.status_code}")
        return r.content

# 画像前処理（A4の小さい文字を少し読みやすく）
def preprocess(img_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("L")  # グレースケール
    # 長辺を最大 2048px に拡大（小さい文字対策）
    w, h = img.size
    scale = 2048 / max(w, h)
    if scale > 1.0:
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    # コントラスト上げる
    img = ImageEnhance.Contrast(img).enhance(1.6)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()

# ---------- OpenAI Vision（最大2問の式をJSON抽出） ----------
async def vision_extract_equations(img_bytes: bytes) -> List[Dict[str, Any]]:
    if not OPENAI_API_KEY:
        return []
    b64 = base64.b64encode(img_bytes).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"

    # Chat Completions（gpt-4o-mini）で JSON 強制
    import httpx as _httpx
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    prompt = (
        "あなたはOCR+数式抽出の役割です。画像から最大2問の『一次 or 二次方程式』を見つけ、"
        "各問題について JSON で返すこと。係数は整数/小数を許可。二次は ax^2+bx+c=0 形式に正規化。"
        "出力は必ず次のJSONのみ："
        "{ \"problems\": [ {\"type\":\"quadratic\",\"a\":...,\"b\":...,\"c\":...,\"text\":\"...\"}, "
        "{\"type\":\"linear\",\"a\":...,\"b\":...,\"text\":\"...\"} ] }"
    )
    payload = {
        "model": "gpt-4o-mini",
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are a precise OCR and math extractor."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ],
        "max_tokens": 800,
        "temperature": 0.2,
    }
    async with _httpx.AsyncClient(timeout=60) as cli:
        r = await cli.post("https://api.openai.com/v1/chat/completions",
                           headers=headers, json=payload)
        if r.status_code != 200:
            print("OpenAI status:", r.status_code, r.text[:200])
            return []
        res = r.json()
        try:
            content = res["choices"][0]["message"]["content"]
            obj = json.loads(content)
            probs = obj.get("problems", [])[:2]
            out = []
            for p in probs:
                t = (p.get("type") or "").lower()
                if t == "quadratic" and all(k in p for k in ("a","b","c")):
                    out.append({"type":"quadratic","a":float(p["a"]), "b":float(p["b"]), "c":float(p["c"]), "text":p.get("text","")})
                elif t == "linear" and all(k in p for k in ("a","b")):
                    out.append({"type":"linear","a":float(p["a"]), "b":float(p["b"]), "text":p.get("text","")})
            return out
        except Exception as e:
            print("parse vision json error:", e, res)
            return []

# ---------- ルート ----------
@app.get("/")
def root():
    return {"ok": True}

@app.post("/webhook")
async def webhook(req: Request):
    # Verify対策：空/非JSONでも必ず200
    try:
        body = await req.json()
    except Exception:
        body = {}
    print("WEBHOOK:", body)
    events = body.get("events", [])
    for ev in events:
        etype = ev.get("type")
        reply_token = ev.get("replyToken")
        if etype == "message":
            m = ev.get("message", {})
            mtype = m.get("type")
            # ---- 画像 ----
            if mtype == "image":
                try:
                    await line_reply(reply_token, ["解析中…（最大2問まで抽出）"])
                    raw = await fetch_line_image(m.get("id"))
                    proc = preprocess(raw)
                    probs = await vision_extract_equations(proc)
                    if not probs:
                        await line_reply(reply_token, [
                            "画像から式を特定できませんでした。画面いっぱいに撮る／ピントを合わせる／"
                            "影や傾きを減らす などを試してください。"
                        ])
                        continue
                    # 各問題を解いて返信
                    out_msgs = []
                    for i, p in enumerate(probs, 1):
                        if p["type"] == "quadratic":
                            sol = solve_quadratic(p["a"], p["b"], p["c"])
                            head = f"【問題{i}/{len(probs)}】（推定）{sol['equation']}"
                            body = f"種別: {sol['kind']}\n解: {', '.join(sol['roots'])}"
                            steps = steps_quadratic(p["a"], p["b"], p["c"])
                            out_msgs += [head, body, steps]
                        elif p["type"] == "linear":
                            sol = solve_linear(p["a"], p["b"])
                            head = f"【問題{i}/{len(probs)}】（推定）{sol['equation']}"
                            body = f"種別: {sol['kind']}\n解: {', '.join(sol['roots']) if sol['roots'] else 'なし'}"
                            steps = steps_linear(p["a"], p["b"])
                            out_msgs += [head, body, steps]
                    await line_reply(reply_token, out_msgs)
                except Exception as e:
                    print("image flow error:", repr(e))
                    await line_reply(reply_token, [
                        "画像解析で内部エラーが発生しました。もう一度、紙面を大きく・ピントを合わせて撮影してみてください。"
                    ])
            # ---- テキスト ----
            elif mtype == "text":
                text = (m.get("text") or "").strip()
                # 総合ガイド
                if text in ("操作方法", "ヘルプ", "help"):
                    await line_reply(reply_token, [GENERAL_OPERATIONS])
                    continue
                # 係数指定の二次
                abc = parse_quadratic_command(text)
                if abc:
                    a, b, c = abc
                    sol = solve_quadratic(a, b, c)
                    msg1 = f"【式】{sol['equation']}\n【種別】{sol['kind']}\n【解】" + (", ".join(sol["roots"]) or "なし")
                    msg2 = steps_quadratic(a, b, c)
                    await line_reply(reply_token, [msg1, msg2])
                    continue
                # その他：使い方
                await line_reply(reply_token, [HELP_TEXT])
        # それ以外のイベントは無視
    return {"ok": True}
