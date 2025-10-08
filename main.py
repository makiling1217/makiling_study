# main.py  — V2 (A4強化OCR/タイル分割/連立対応/堅牢化)
from fastapi import FastAPI, Request
import os, json, math, asyncio, base64, io, re
import httpx

# 画像前処理
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2  # opencv-python-headless

# 数式の計算
import sympy as sp

app = FastAPI()

# ========= 環境変数 =========
LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")   # 画像→テキスト/JSON
TEXT_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")          # テキスト→JSON整形など

# ========= 小ユーティリティ =========
def logj(x):
    try:
        print(json.dumps(x, ensure_ascii=False))
    except Exception:
        print(x)

def normalize(s: str) -> str:
    try:
        import unicodedata
        s = unicodedata.normalize("NFKC", s)
    except Exception:
        pass
    s = s.replace("，", ",").replace("．", ".")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_long(text: str, limit=900):
    out, cur, n = [], [], 0
    for line in text.splitlines():
        if n + len(line) + 1 > limit:
            out.append("\n".join(cur)); cur=[]; n=0
        cur.append(line); n += len(line)+1
    if cur: out.append("\n".join(cur))
    return out or ["(empty)"]

# ========= LINE API =========
LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
LINE_PUSH_URL  = "https://api.line.me/v2/bot/message/push"
LINE_CONTENT   = "https://api-data.line.me/v2/bot/message/{messageId}/content"  # ← 404対策：api-data

async def line_reply(reply_token: str, texts):
    if not reply_token: return
    messages = [{"type":"text","text":t} for t in texts]
    async with httpx.AsyncClient(timeout=20.0) as cli:
        r = await cli.post(LINE_REPLY_URL,
            headers={"Authorization": f"Bearer {LINE_TOKEN}"},
            json={"replyToken": reply_token, "messages": messages})
        print("LINE reply:", r.status_code, r.text[:200])

async def line_push(user_id: str, texts):
    if not user_id: return
    messages = [{"type":"text","text":t} for t in texts]
    async with httpx.AsyncClient(timeout=20.0) as cli:
        r = await cli.post(LINE_PUSH_URL,
            headers={"Authorization": f"Bearer {LINE_TOKEN}"},
            json={"to": user_id, "messages": messages})
        print("LINE push:", r.status_code, r.text[:200])

# ========= A4対応：画像強化 & 自動分割 =========
def to_bytes(pil: Image.Image, q=92) -> bytes:
    buf = io.BytesIO(); pil.save(buf, format="JPEG", quality=q); return buf.getvalue()

def deskew_cv(pil: Image.Image) -> Image.Image:
    """軽い傾き補正（Hough線から角度推定）。失敗時はそのまま返す。"""
    try:
        img = np.array(pil.convert("L"))
        edges = cv2.Canny(img, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        if lines is None: return pil
        angles = []
        for rho, theta in lines[:,0,:]:
            ang = (theta*180/np.pi) - 90
            if -30 <= ang <= 30:
                angles.append(ang)
        if not angles: return pil
        angle = float(np.median(angles))
        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rot = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(rot)
    except Exception:
        return pil

def enhance_and_tile(img_bytes: bytes) -> list[bytes]:
    """
    A4の小さめ文字向け：
      1) deskew → 2.0〜2.5倍拡大 → autocontrast → ほんの少しシャープ
      2) フル + 上下 + 4分割（最大6タイル）を返す（順に試す）
    """
    pil = Image.open(io.BytesIO(img_bytes)).convert("L")
    pil = deskew_cv(pil)

    w, h = pil.size
    scale = 2.5 if max(w,h) < 2200 else 1.8
    pil = pil.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    pil = ImageOps.autocontrast(pil)
    pil = pil.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))

    tiles = []
    W, H = pil.size
    # 1) フル
    tiles.append(to_bytes(pil))
    # 2) 上/下
    tiles.append(to_bytes(pil.crop((0,0,W,H//2))))
    tiles.append(to_bytes(pil.crop((0,H//2,W,H))))
    # 3) 四分割
    tiles.append(to_bytes(pil.crop((0,0,W//2,H//2))))
    tiles.append(to_bytes(pil.crop((W//2,0,W,H//2))))
    tiles.append(to_bytes(pil.crop((0,H//2,W//2,H))))
    # （6個で打ち止め）
    return tiles

def as_data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")

# ========= 数学処理（一次・二次・連立2元） =========
def fx_steps_quadratic(a,b,c):
    return "\n".join([
        "【fx-CG50：二次方程式】",
        "1. [MENU] → 『EQUATION/EQUA』 → [EXE]",
        "2. [F2] Polynomial",
        "3. Degree に『2』を選択",
        "4. a= を入力 → [EXE]",
        "5. b= を入力 → [EXE]",
        "6. c= を入力 → [EXE]",
        "※ 負号はテンキー右下の [(-)] を使う（ハイフン[-]は不可）",
        "7. 解が表示。小数/分数の切替は [S⇔D]、戻るは [EXIT]",
        f"(入力例 a={a}, b={b}, c={c})"
    ])

def fx_steps_linear(a,b):
    return "\n".join([
        "【fx-CG50：一次方程式 ax+b=0】",
        "1. [MENU] → 『EQUATION/EQUA』 → [EXE]",
        "2. [F2] Polynomial → Degree『1』",
        "3. a=, b= を入力 → [EXE]",
        "4. 解 x が表示（[S⇔D] で表記切替）",
        "※ 負号は [(-)] を使用"
    ])

def fx_steps_simul2():
    return "\n".join([
        "【fx-CG50：連立2元一次】",
        "1. [MENU] → 『EQUATION/EQUA』 → [EXE]",
        "2. [F1] Simultaneous",
        "3. Unknowns（未知数）で『2』を選択 → [EXE]",
        "4. 1本目の式の係数を順に入力（xの係数, yの係数, 定数）→ [EXE]",
        "5. 2本目も同様に入力 → [EXE]",
        "6. 解 x, y が表示（[S⇔D] で切替）",
        "※ 負号は [(-)] を使用"
    ])

def solve_linear(a,b):
    if a == 0: return {"kind":"none" if b!=0 else "indef"}
    return {"kind":"one", "x": float(-b/a)}

def solve_quadratic(a,b,c):
    a,b,c = float(a),float(b),float(c)
    if a == 0:  # 一次へ
        s = solve_linear(b,c)
        s["fallback"]="linear"
        return s
    D = b*b - 4*a*c
    if D > 0:
        r1 = (-b + math.sqrt(D))/(2*a)
        r2 = (-b - math.sqrt(D))/(2*a)
        return {"kind":"real2","D":D,"x1":r1,"x2":r2}
    elif D == 0:
        r = (-b)/(2*a)
        return {"kind":"double","D":D,"x":r}
    else:
        real = (-b)/(2*a); imag = math.sqrt(-D)/(2*a)
        return {"kind":"complex","D":D,"x1":f"{real}+{imag}i","x2":f"{real}-{imag}i"}

def solve_simul_2(eq1, eq2, vars=("x","y")):
    x,y = sp.symbols(vars)
    try:
        s = sp.solve([sp.Eq(*eq1), sp.Eq(*eq2)], (x,y), dict=True)
        if not s: return {"kind":"none"}
        sol = s[0]
        return {"kind":"ok","x":sp.N(sol[x]),"y":sp.N(sol[y])}
    except Exception:
        return {"kind":"err"}

def fmt_quadratic(a,b,c,sol):
    eq = f"{a}·x^2 + {b}·x + {c} = 0"
    if sol.get("fallback")=="linear":
        lin = f"{b}·x + {c} = 0"
        if sol["kind"]=="one": return f"式：{lin}\n解：x = {sol['x']}"
        if sol["kind"]=="none": return f"式：{lin}\n解なし（矛盾）"
        return f"式：{lin}\n不定（無数の解）"
    k = sol["kind"]
    if k=="real2":
        return f"式：{eq}\nD={sol['D']}\n解：x1={sol['x1']} , x2={sol['x2']}"
    if k=="double":
        return f"式：{eq}\nD=0\n解：x={sol['x']}"
    if k=="complex":
        return f"式：{eq}\nD={sol['D']}\n解：x1={sol['x1']} , x2={sol['x2']}"
    return f"式：{eq}\n解の判定不可"

# ========= コマンド（手入力）解析 =========
HELP_TEXT = "\n".join([
    "使い方：",
    "1) A4の問題写真を送る → 『式＋解＋番号付き手順』（最大2問）で返信",
    "2) 係数指定：二次 a=1 b=-3 c=2 / 二次 1 -3 2 / 二次 1,-3,2",
    "3) 一次：一次 2 3  （2x+3=0）",
    "4) 連立：連立 2,1,5; 1,-1,1  （2x+y=5 と x−y=1）",
    "5) キー一覧：『操作方法』"
])

OPS_GUIDE = "\n".join([
    "【fx-CG50 キー操作総合】",
    "・[MENU]→アイコン決定[EXE] / 戻る[EXIT] / 小数⇔分数[S⇔D]",
    "・負号は [(-)] を使う（ハイフン[-]不可）",
    "・EQUATION(EQUA)：F1 連立 / F2 多項式 / F6 EXIT",
    "・二次：F2→Degree 2→a,b,c 入力→[EXE]",
    "・一次：F2→Degree 1→a,b 入力→[EXE]",
    "・2元連立：F1→Unknowns 2→係数入力→[EXE]"
])

def parse_quadratic_cmd(t: str):
    t = normalize(t)
    if not t.startswith("二次"): return None
    rest = t[2:].strip()
    # a= b= c=
    m = re.findall(r"[abc]\s*=\s*[-+]?\d+(?:\.\d+)?", rest, flags=re.I)
    if len(m) >= 3:
        vals = {}
        for kv in m[:3]:
            k,v = kv.split("="); vals[k.strip().lower()] = float(v)
        return vals.get("a"), vals.get("b"), vals.get("c")
    # 数字3つ
    nums = re.split(r"[,\s/]+", rest)
    nums = [x for x in nums if x]
    if len(nums) >= 3:
        try:
            return float(nums[0]), float(nums[1]), float(nums[2])
        except: return None
    return None

def parse_linear_cmd(t:str):
    t = normalize(t)
    if not t.startswith("一次"): return None
    nums = re.split(r"[,\s/]+", t[2:].strip())
    nums = [x for x in nums if x]
    if len(nums)>=2:
        try: return float(nums[0]), float(nums[1])
        except: return None
    return None

def parse_simul_cmd(t:str):
    t = normalize(t)
    if not t.startswith("連立"): return None
    # 例: 連立 2,1,5; 1,-1,1  → 2x+1y=5 / 1x-1y=1
    body = t[2:].strip()
    parts = [p.strip() for p in re.split(r"[;；]", body) if p.strip()]
    if len(parts) != 2: return None
    def parse_triplet(s):
        ss = [u for u in re.split(r"[,\s/]+", s) if u]
        if len(ss) < 3: return None
        return float(ss[0]), float(ss[1]), float(ss[2])
    t1 = parse_triplet(parts[0]); t2 = parse_triplet(parts[1])
    if not t1 or not t2: return None
    return t1, t2   # (a1,b1,c1), (a2,b2,c2)

# ========= OpenAI 呼び出し =========
def data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")

async def openai_extract_from_tiles(tiles: list[bytes]) -> list[dict]:
    """
    複数タイルを順に投げ、最大2問を抽出して返す。
    抽出スキーマ：
      {"items":[
         {"type":"quadratic","a":..,"b":..,"c":..,"latex":"..."},
         {"type":"linear","a":..,"b":..},
         {"type":"simul2","a1":..,"b1":..,"c1":..,"a2":..,"b2":..,"c2":..}
      ]}
    """
    if not OPENAI_KEY: return []

    system = (
        "You are a rigorous math OCR/understander. "
        "Return STRICT JSON only: {\"items\":[...]} with up to TWO problems detected "
        "(Japanese/English allowed). Recognize linear/quadratic and 2x2 simultaneous equations. "
        "Numbers only (no fractions symbols; convert to decimals if needed)."
    )

    # まずフル→上下→四分割の順で、見つかったら追加していく
    found = []
    try:
        from openai import OpenAI
        cli = OpenAI(api_key=OPENAI_KEY)
    except Exception:
        return []

    for img in tiles:
        if len(found) >= 2: break
        payload = {
            "model": VISION_MODEL,
            "input": [
                {"role":"system","content":system},
                {"role":"user","content":[
                    {"type":"input_text","text":"Extract up to one problem from this image tile as JSON."},
                    {"type":"input_image","image_url": data_url(img)}
                ]}
            ],
            "temperature": 0
        }
        try:
            resp = await asyncio.to_thread(cli.responses.create, **payload)
            txt = resp.output_text
            data = json.loads(txt)
            items = data.get("items", [])
            for it in items:
                t = it.get("type")
                if t == "quadratic":
                    a=float(it["a"]); b=float(it["b"]); c=float(it["c"])
                    found.append({"type":"quadratic","a":a,"b":b,"c":c,"latex":it.get("latex","")})
                elif t == "linear":
                    a=float(it["a"]); b=float(it["b"])
                    found.append({"type":"linear","a":a,"b":b})
                elif t == "simul2":
                    a1=float(it["a1"]); b1=float(it["b1"]); c1=float(it["c1"])
                    a2=float(it["a2"]); b2=float(it["b2"]); c2=float(it["c2"])
                    found.append({"type":"simul2","a1":a1,"b1":b1,"c1":c1,"a2":a2,"b2":b2,"c2":c2})
                if len(found) >= 2: break
        except Exception as e:
            print("Vision tile error:", e)
            continue

    return found[:2]

# ========= 非同期：画像解析→PUSH =========
async def analyze_and_push(user_id: str, img_bytes: bytes):
    try:
        tiles = enhance_and_tile(img_bytes)   # A4強化 + 分割
        items = await openai_extract_from_tiles(tiles)

        if not items:
            await line_push(user_id, [
                "式を特定できませんでした。",
                "・A4のままでOK（正面・影少なめ）",
                "・どうしても読めない場合：『二次 a=1 b=-3 c=2』『連立 2,1,5; 1,-1,1』など係数指定でも解答できます。"
            ])
            return

        out = []
        for i,it in enumerate(items, start=1):
            header = f"【問題{i}】"
            if it["type"]=="quadratic":
                a,b,c = it["a"],it["b"],it["c"]
                sol = solve_quadratic(a,b,c)
                out.append(header + "\n" + fmt_quadratic(a,b,c,sol))
                out.append(fx_steps_quadratic(a,b,c))
            elif it["type"]=="linear":
                a,b = it["a"],it["b"]
                s = solve_linear(a,b)
                if s["kind"]=="one":
                    out.append(header + f"\n式：{a}·x + {b} = 0\n解：x = {s['x']}")
                elif s["kind"]=="none":
                    out.append(header + f"\n式：{a}·x + {b} = 0\n解なし（矛盾）")
                else:
                    out.append(header + f"\n式：{a}·x + {b} = 0\n不定（無数の解）")
                out.append(fx_steps_linear(a,b))
            elif it["type"]=="simul2":
                a1,b1,c1 = it["a1"],it["b1"],it["c1"]
                a2,b2,c2 = it["a2"],it["b2"],it["c2"]
                x,y = sp.symbols("x y")
                res = sp.solve([sp.Eq(a1*x+b1*y,c1), sp.Eq(a2*x+b2*y,c2)], (x,y), dict=True)
                if res:
                    sol = res[0]
                    out.append(header + f"\n式：{a1}x+{b1}y={c1} , {a2}x+{b2}y={c2}\n解：x={sp.N(sol[x])}, y={sp.N(sol[y])}")
                else:
                    out.append(header + f"\n式：{a1}x+{b1}y={c1} , {a2}x+{b2}y={c2}\n解を特定できませんでした。")
                out.append(fx_steps_simul2())

        # 分割して順にPUSH
        for m in out:
            for part in split_long(m):
                await line_push(user_id, [part])
                await asyncio.sleep(0.2)

    except Exception as e:
        print("analyze_and_push error:", e)
        await line_push(user_id, ["内部エラーが発生しました。もう一度お試しください。"])

# ========= ルート =========
@app.get("/")
def root():
    return {"ok": True}

@app.post("/webhook")
async def webhook(req: Request):
    try:
        body = await req.json()
    except Exception:
        body = {}
    logj({"WEBHOOK": body})

    events = body.get("events", [])
    if not events: return {"ok": True}

    for ev in events:
        et = ev.get("type")
        src = ev.get("source", {})
        user_id = src.get("userId", "")
        rep = ev.get("replyToken","")

        if et == "message":
            m = ev.get("message", {})
            mt = m.get("type")

            # 文字メッセージ
            if mt == "text":
                t = normalize(m.get("text",""))

                if "操作方法" in t or "キー操作" in t:
                    await line_reply(rep, split_long(OPS_GUIDE)); continue
                if t in ("help","使い方","?","？"):
                    await line_reply(rep, split_long(HELP_TEXT)); continue

                q = parse_quadratic_cmd(t)
                if q:
                    a,b,c = q
                    sol = solve_quadratic(a,b,c)
                    msg = fmt_quadratic(a,b,c,sol)
                    out = ["判別式・タイプは実数/重解/虚数で自動判定", msg, fx_steps_quadratic(a,b,c)]
                    await line_reply(rep, split_long("\n".join(out))); continue

                lin = parse_linear_cmd(t)
                if lin:
                    a,b = lin
                    s = solve_linear(a,b)
                    if s["kind"]=="one":
                        out = [f"式：{a}·x + {b} = 0\n解：x = {s['x']}", fx_steps_linear(a,b)]
                    elif s["kind"]=="none":
                        out = [f"式：{a}·x + {b} = 0\n解なし（矛盾）", fx_steps_linear(a,b)]
                    else:
                        out = [f"式：{a}·x + {b} = 0\n不定（無数の解）", fx_steps_linear(a,b)]
                    await line_reply(rep, split_long("\n".join(out))); continue

                sim = parse_simul_cmd(t)
                if sim:
                    (a1,b1,c1),(a2,b2,c2) = sim
                    x,y = sp.symbols("x y")
                    res = sp.solve([sp.Eq(a1*x+b1*y,c1), sp.Eq(a2*x+b2*y,c2)], (x,y), dict=True)
                    if res:
                        sol = res[0]
                        msg = f"式：{a1}x+{b1}y={c1} , {a2}x+{b2}y={c2}\n解：x={sp.N(sol[x])}, y={sp.N(sol[y])}"
                    else:
                        msg = f"式：{a1}x+{b1}y={c1} , {a2}x+{b2}y={c2}\n解を特定できませんでした。"
                    await line_reply(rep, split_long(msg + "\n\n" + fx_steps_simul2())); continue

                # それ以外
                await line_reply(rep, split_long("形式を認識できませんでした。\n" + HELP_TEXT))
                continue

            # 画像メッセージ
            if mt == "image":
                await line_reply(rep, ["解析中… 少し待ってね。"])
                mid = m.get("id")
                img = None
                if mid:
                    url = LINE_CONTENT.format(messageId=mid)
                    async with httpx.AsyncClient(timeout=25.0) as cli:
                        for _ in range(5):
                            r = await cli.get(url, headers={"Authorization": f"Bearer {LINE_TOKEN}"})
                            if r.status_code == 200:
                                img = r.content; break
                            await asyncio.sleep(0.6)
                if not img:
                    await line_push(user_id, ["画像を取得できませんでした。もう一度送ってください。"])
                else:
                    asyncio.create_task(analyze_and_push(user_id, img))
                continue

    return {"ok": True}
