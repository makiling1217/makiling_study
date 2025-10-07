# main.py
from fastapi import FastAPI, Request
import os, json, math, asyncio, base64, io, re
import httpx
from PIL import Image, ImageOps, ImageFilter

# ====== 環境変数 ======
LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
OPENAI_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # 任意で上書き可

app = FastAPI()

# ====== ユーティリティ ======
def j(obj):  # 見やすいログ
    try:
        print(json.dumps(obj, ensure_ascii=False))
    except Exception:
        print(obj)

def normalize_text(s: str) -> str:
    # 全角→半角、全角記号対応、区切りのゆれを吸収
    try:
        import unicodedata
        s = unicodedata.normalize("NFKC", s)
    except Exception:
        pass
    s = s.replace("，", ",").replace("．", ".").replace("・", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_msgs(big: str, limit: int = 900) -> list[str]:
    # LINEは長文で切れるので分割
    parts, cur = [], []
    cur_len = 0
    for line in big.splitlines():
        if cur_len + len(line) + 1 > limit:
            parts.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(line)
        cur_len += len(line) + 1
    if cur:
        parts.append("\n".join(cur))
    return parts or ["(empty)"]

# ====== LINE API ======
LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
LINE_PUSH_URL  = "https://api.line.me/v2/bot/message/push"
LINE_CONTENT   = "https://api-data.line.me/v2/bot/message/{messageId}/content"  # ← 404対策：api-data

async def line_reply(reply_token: str, texts: list[str]):
    if not reply_token: return
    messages = [{"type": "text", "text": t} for t in texts]
    async with httpx.AsyncClient(timeout=15.0) as cli:
        r = await cli.post(
            LINE_REPLY_URL,
            headers={"Authorization": f"Bearer {LINE_TOKEN}"},
            json={"replyToken": reply_token, "messages": messages},
        )
        print("LINE reply status:", r.status_code, r.text)

async def line_push(user_id: str, texts: list[str]):
    if not user_id: return
    messages = [{"type": "text", "text": t} for t in texts]
    async with httpx.AsyncClient(timeout=15.0) as cli:
        r = await cli.post(
            LINE_PUSH_URL,
            headers={"Authorization": f"Bearer {LINE_TOKEN}"},
            json={"to": user_id, "messages": messages},
        )
        print("LINE push status:", r.status_code, r.text)

# ====== 画像前処理（A4対策：拡大＋コントラスト） ======
def preprocess_image(img_bytes: bytes) -> list[bytes]:
    """
    A4の小さめ文字でも読めるように：
      1) 2.0〜2.5倍拡大
      2) グレイスケール＋自動コントラスト
      3) ほんの少しシャープ
      4) 上下分割（最大2問）も試す
    返り値：OpenAIに渡す候補画像（フル・上半分・下半分）
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    w, h = img.size
    scale = 2.2 if max(w, h) < 2200 else 1.6   # だいたいA4の画素数に合わせて
    img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))

    # JPEGに再エンコード
    def to_bytes(pil: Image.Image) -> bytes:
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=92)
        return buf.getvalue()

    out = [to_bytes(img)]

    # 上下2分割（問題2つ想定）
    upper = img.crop((0, 0, img.width, img.height//2))
    lower = img.crop((0, img.height//2, img.width, img.height))
    out.append(to_bytes(upper))
    out.append(to_bytes(lower))
    return out

def b64_data_url(jpeg_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")

# ====== 数学ユーティリティ ======
def solve_quadratic(a: float, b: float, c: float):
    D = b*b - 4*a*c
    if a == 0:
        # 一次へフォールバック
        if b == 0:
            return {"kind":"degenerate","text":"解なし（a=b=0）"}
        return {"kind":"linear","x": -c/b}
    if D > 0:
        r1 = (-b + math.sqrt(D)) / (2*a)
        r2 = (-b - math.sqrt(D)) / (2*a)
        kind = "2 real roots"
    elif D == 0:
        r1 = r2 = (-b) / (2*a)
        kind = "double root"
    else:
        # 複素数
        real = (-b) / (2*a)
        imag = math.sqrt(-D) / (2*a)
        return {"kind":"complex", "x1": f"{real}+{imag}i", "x2": f"{real}-{imag}i", "D": D}
    return {"kind":kind, "x1": r1, "x2": r2, "D": D}

def steps_quadratic_fxcg50(a,b,c) -> str:
    return "\n".join([
        "【fx-CG50：二次方程式のキー操作】",
        "1) [MENU] → アイコン『EQUATION/EQUA』を選んで [EXE]",
        "2) 画面下の[F2] Polynomial を選択",
        "3) Degree（次数）で「2」を押す（2次）",
        "4) 係数を順に入力：",
        "   ・a= を入力 → [EXE]",
        "   ・b= を入力 → [EXE]",
        "   ・c= を入力 → [EXE]",
        "   ※負号は [(-)]（テンキーの右下）を使う。ハイフン[-]は不可。",
        "5) 解 x1, x2 が表示される",
        "6) 小数/分数の切替：[S⇔D] キー",
        "7) 角度や表示など設定：[SHIFT]→[MENU](SETUP) で調整／戻るときは [EXIT]",
        "（Fキーの表示は画面下のラベルどおり：F1 Simultaneous, F2 Polynomial, F3…/F6 EXIT 等）",
        f"入力例：a={a}, b={b}, c={c}"
    ])

def format_solution(a,b,c,sol) -> str:
    eq = f"{a}·x^2 + {b}·x + {c} = 0"
    if sol.get("kind") == "complex":
        body = f"D={sol['D']}\n解：x1={sol['x1']}, x2={sol['x2']}"
    elif sol.get("kind") == "linear":
        body = f"(a=0 のため一次)  解：x={sol['x']}"
    elif sol.get("kind") == "degenerate":
        body = sol["text"]
    else:
        body = f"D={sol['D']}\n解：x1={sol['x1']},  x2={sol['x2']}"
    return f"式：{eq}\n{body}"

# ====== テキスト解析（コマンド） ======
def parse_quadratic_command(text: str):
    """
    受理する形：
      ・「二次 a=1 b=-3 c=2」
      ・「二次 1 -3 2」
      ・「二次 1,-3,2」 など区切り自由
    """
    t = normalize_text(text)
    if not t.startswith("二次"):
        return None
    t = t[2:].strip()  # "二次" を取り除く
    # a=,b=,c= のとき
    m = re.findall(r"[abcABC]\s*=\s*[-+]?\d+(?:\.\d+)?", t)
    if m and len(m) >= 3:
        vals = {}
        for kv in m:
            k,v = kv.split("=")
            vals[k.strip().lower()] = float(v)
        a = vals.get("a"); b = vals.get("b"); c = vals.get("c")
        if a is not None and b is not None and c is not None:
            return (a,b,c)

    # 数字3つ（区切りは空白/カンマ/スラッシュ）
    nums = re.split(r"[,\s/]+", t)
    nums = [n for n in nums if n]
    if len(nums) >= 3:
        try:
            a = float(nums[0]); b = float(nums[1]); c = float(nums[2])
            return (a,b,c)
        except Exception:
            return None
    return None

HELP_TEXT = "\n".join([
    "使い方：",
    "1) 問題の写真を送る → 自動解析して『式＋答え＋番号付き手順』（最大2問）",
    "2) 係数指定：例）二次 a=1 b=-3 c=2   /   二次 1 -3 2   /   二次 1,-3,2",
    "3) キー操作の一覧：『操作方法』と送信",
])

OPS_GUIDE = "\n".join([
    "【fx-CG50 キー操作の総合ガイド】",
    "1. メニュー：[MENU] → 各アイコンを [EXE] で決定",
    "2. EQUATION（EQUA）：方程式の解法メニュー",
    "   F1: Simultaneous（連立） / F2: Polynomial（多項式） / F6: EXIT（戻る）",
    "3. 入力のコツ：",
    "   ・負号は [(-)] を使用（ハイフン[-]は不可）",
    "   ・小数⇔分数切替は [S⇔D]",
    "   ・数式入力後は [EXE] で確定",
    "4. 二次方程式は『F2→Degree 2→a,b,c 入力→EXE』",
    "5. 設定変更：[SHIFT]→[MENU](SETUP)",
])

# ====== OpenAI 呼び出し（画像→JSON抽出） ======
async def ask_openai_equations(image_bytes_list: list[bytes]) -> list[dict]:
    """
    画像（フル/上/下）を渡し、最大2問の方程式を JSON で返すよう指示。
    サポート：一次/二次。認識できない場合は空配列。
    """
    if not OPENAI_KEY:
        return []

    # 画像3パターンを並べて与える（どれかで読めばOK）
    images = [{"type": "input_image", "image_url": b64_data_url(b)} for b in image_bytes_list]

    system = (
        "You are a math OCR and solver. Extract up to TWO equations from the images. "
        "Return strict JSON with this schema:\n"
        "{ \"items\": [ {\"type\": \"quadratic\", \"latex\": \"ax^2+bx+c=0\", \"a\": number, \"b\": number, \"c\": number}, "
        "              {\"type\": \"linear\",    \"latex\": \"ax+b=0\",     \"a\": number, \"b\": number} ] }\n"
        "If nothing confident, return {\"items\":[]}. No explanation."
    )

    # Responses API（失敗時はフォールバック）
    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "input_text", "text": "Find up to two equations."},
                *images
            ]},
        ],
        "temperature": 0
    }

    try:
        import openai
        from openai import OpenAI
        cli = OpenAI(api_key=OPENAI_KEY)
        resp = await asyncio.to_thread(cli.responses.create, **payload)
        txt = resp.output_text  # SDK>=1.40系
    except Exception as e:
        print("OpenAI responses error:", e)
        # 旧ChatCompletionsフォールバック
        try:
            import openai
            openai.api_key = OPENAI_KEY
            comp = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-4o",
                messages=[
                    {"role":"system", "content": system},
                    {"role":"user", "content": [{"type":"text","text":"Find up to two equations."}]}
                ],
                temperature=0
            )
            txt = comp.choices[0].message["content"]
        except Exception as e2:
            print("OpenAI fallback error:", e2)
            return []

    try:
        data = json.loads(txt)
        items = data.get("items", [])
        # 型・係数が揃っているものだけ通す
        cleaned = []
        for it in items[:2]:
            t = it.get("type")
            if t == "quadratic":
                a = float(it.get("a")); b = float(it.get("b")); c = float(it.get("c"))
                cleaned.append({"type":"quadratic","a":a,"b":b,"c":c,"latex":it.get("latex","")})
            elif t == "linear":
                a = float(it.get("a")); b = float(it.get("b"))
                cleaned.append({"type":"linear","a":a,"b":b,"latex":it.get("latex","")})
        return cleaned
    except Exception as e:
        print("JSON parse error:", e, "raw:", txt)
        return []

# ====== 画像→解析→PUSH（非同期ジョブ） ======
async def analyze_image_and_push(user_id: str, img_bytes: bytes):
    try:
        imgs = preprocess_image(img_bytes)  # 拡大・強調・上下
        items = await ask_openai_equations(imgs)

        if not items:
            await line_push(user_id, [
                "式を特定できませんでした。A4の写真でもOKです。可能なら：",
                "・紙面をできるだけ正面から（台形にならないように）",
                "・影/反射を避ける、文字が淡ければ少し濃いモードで再撮影",
                "・それでも難しい場合は『二次 a=1 b=-3 c=2』の形式で送ってください"
            ])
            return

        out_msgs = []
        for idx, it in enumerate(items, start=1):
            if it["type"] == "quadratic":
                a,b,c = it["a"], it["b"], it["c"]
                sol = solve_quadratic(a,b,c)
                head = f"【問題{idx}】二次方程式"
                out_msgs.append(head + "\n" + format_solution(a,b,c,sol))
                out_msgs.append(steps_quadratic_fxcg50(a,b,c))
            elif it["type"] == "linear":
                a,b = it["a"], it["b"]
                x = None if a==0 else (-b/a)
                head = f"【問題{idx}】一次方程式"
                eq = f"{a}·x + {b} = 0"
                body = "解なし（a=0）" if a==0 else f"解：x = {x}"
                out_msgs.append(head + "\n式：" + eq + "\n" + body)

        # 分割してプッシュ
        for m in out_msgs:
            for part in split_msgs(m):
                await line_push(user_id, [part])
                await asyncio.sleep(0.2)

    except Exception as e:
        print("analyze_image_and_push error:", e)
        await line_push(user_id, ["内部エラーが発生しました。もう一度お試しください。"])

# ====== ルーティング ======
@app.get("/")
def root():
    return {"ok": True}

@app.post("/webhook")
async def webhook(req: Request):
    # Verify対策：空でも200
    try:
        body = await req.json()
    except Exception:
        body = {}
    print("WEBHOOK:", body)

    events = body.get("events", [])
    if not events:
        return {"ok": True}

    for ev in events:
        et = ev.get("type")
        src = ev.get("source", {})
        user_id = src.get("userId")
        reply_token = ev.get("replyToken", "")

        if et == "message":
            m = ev.get("message", {})
            mtype = m.get("type")

            # ---- テキスト ----
            if mtype == "text":
                text = m.get("text", "")
                tnorm = normalize_text(text)

                # 1) 操作方法
                if "操作方法" in tnorm or "キー操作" in tnorm:
                    await line_reply(reply_token, split_msgs(OPS_GUIDE))
                    continue

                # 2) ヘルプ
                if tnorm in ("help","？","?","使い方"):
                    await line_reply(reply_token, split_msgs(HELP_TEXT))
                    continue

                # 3) 二次 a,b,c
                abc = parse_quadratic_command(tnorm)
                if abc:
                    a,b,c = abc
                    sol = solve_quadratic(a,b,c)
                    msgs = []
                    msgs.append(format_solution(a,b,c,sol))
                    msgs.append(steps_quadratic_fxcg50(a,b,c))
                    # 判別式の種別も添える
                    D = b*b - 4*a*c
                    kind = "実数解2つ" if D>0 else ("重解" if D==0 else "虚数解")
                    msgs.insert(0, f"判別式D={D} → {kind}")
                    # 返信
                    await line_reply(reply_token, split_msgs("\n".join(msgs)))
                    continue

                # 4) それ以外はヘルプ
                await line_reply(reply_token, split_msgs("すみません。形式を認識できませんでした。\n" + HELP_TEXT))
                continue

            # ---- 画像 ----
            if mtype == "image":
                # まず即時応答（固まって見えないように）
                await line_reply(reply_token, ["解析中… 少し待ってね。"])

                # コンテンツ取得（api-data.* に変更済）
                mid = m.get("id")
                img_bytes = None
                if mid:
                    fetch_url = LINE_CONTENT.format(messageId=mid)
                    async with httpx.AsyncClient(timeout=20.0) as cli:
                        # LINE側がストレージ反映に時間かかる場合があるので数回リトライ
                        for _ in range(4):
                            rr = await cli.get(
                                fetch_url,
                                headers={"Authorization": f"Bearer {LINE_TOKEN}"},
                            )
                            if rr.status_code == 200:
                                img_bytes = rr.content
                                break
                            await asyncio.sleep(0.6)

                if not img_bytes:
                    await line_push(user_id, ["画像を取得できませんでした。もう一度送ってください。"])
                else:
                    # 非同期で解析→PUSH
                    asyncio.create_task(analyze_image_and_push(user_id, img_bytes))

                continue

    return {"ok": True}
