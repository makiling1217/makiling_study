# main.py
# FastAPI + LINE Bot + OpenAI Vision
# 画像(A4想定)→前処理→AIで最大2問抽出→式/解/手順を返信
# テキスト:
#   ・「操作方法」：総合ガイド(番号付き)
#   ・「二次 a=1 b=-3 c=2」 / 「二次 1,-3,2」 / 「二次 1 -3 2」：解＋手順

from fastapi import FastAPI, Request
import os, httpx, json, base64, math
from PIL import Image, ImageOps, ImageEnhance
from io import BytesIO
from typing import List, Dict

# ==== 環境変数 ====
LINE_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
VISION_MODEL = os.environ.get("VISION_MODEL", "gpt-4o-mini")  # 好きなら gpt-4o に

app = FastAPI()

# ========= ユーティリティ =========
async def line_reply(reply_token: str, messages: List[Dict]):
    if not LINE_TOKEN:
        print("WARN: no LINE_CHANNEL_ACCESS_TOKEN; skip reply")
        return
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {"Authorization": f"Bearer {LINE_TOKEN}"}
    payload = {"replyToken": reply_token, "messages": messages[:5]}
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(url, headers=headers, json=payload)
        print("LINE reply status:", r.status_code, r.text)

async def line_push(user_id: str, messages: List[Dict]):
    if not LINE_TOKEN:
        print("WARN: no LINE_CHANNEL_ACCESS_TOKEN; skip push")
        return
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Authorization": f"Bearer {LINE_TOKEN}"}
    payload = {"to": user_id, "messages": messages[:5]}
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(url, headers=headers, json=payload)
        print("LINE push status:", r.status_code, r.text)

def split_long_text(txt: str, chunk=1800) -> List[str]:
    out, cur = [], ""
    for line in txt.splitlines():
        if len(cur) + len(line) + 1 > chunk:
            out.append(cur)
            cur = ""
        cur += (line + "\n")
    if cur:
        out.append(cur)
    return out[:5]  # LINEは一度に最大5件

# ==== 操作ガイド ====
GUIDE = """\
【fx-CG50 操作方法（総合）】
1) MENU → GRAPH → EXE（FUNC 画面に入る）※右下に DRAW が見えること
2) Y1 行へカーソル → F1(SELECT) で左端の「＝」を濃く（ON）
3) Y1 を空にする → DEL
4) 数式の入力は次のキー順で。EXE を押すタイミングは必ず指示どおりに！
   例) y = −x² + 4ax + b
   [(-)] → [X,θ,T] → [x²] → [+] → 4 → [×] → [ALPHA][A] → [×] → [X,θ,T] → [+] → [ALPHA][B] → [EXE]
   ※ [ALPHA] はピンクの “A/B/C…”。[X,θ,T] は X キー。x² は二乗キー。
5) 描画：F6(DRAW)。解読支援：SHIFT+F5(G-Solv) から ROOT/INTERSECT/MAX/MIN など
6) 方程式を解くなら：MENU → EQUA → EXE → [F2](Polynomial) → degree を選ぶ → 係数 a,b,c を入れて EXE
7) RUN-MAT で A=0.5 のように代入する：
   [AC/ON] → 0.5 → [SHIFT][RCL](STO▶) → [ALPHA][A] → EXE
   B=4 なら 4 → [SHIFT][RCL] → [ALPHA][B] → EXE
   ※ [SHIFT][RCL] は “STO▶” です（x² の右、EXIT の左にあるキーが [RCL]）
"""

# ==== 数学処理（最小限）====
def solve_quadratic(a: float, b: float, c: float):
    D = b*b - 4*a*c
    if D > 0:
        r1 = (-b - math.sqrt(D)) / (2*a)
        r2 = (-b + math.sqrt(D)) / (2*a)
        kind = "2実数解"
        roots = sorted([r1, r2])
    elif D == 0:
        r = -b / (2*a)
        kind = "重解"
        roots = [r, r]
    else:
        kind = "虚数解"
        roots = []
    return kind, D, roots

def steps_quadratic(a, b, c):
    # Equation（EQUA）での丁寧な番号手順
    s = []
    s.append("1. [MENU] → EQUA → EXE（方程式）")
    s.append("2. [F2](Polynomial) → degree=2 → EXE")
    s.append(f"3. 係数を入力 → a={a} → EXE → b={b} → EXE → c={c} → EXE")
    s.append("4. [F6](Solve) → 解が表示 → [EXE] で確定")
    s.append("5. 戻って確認したい場合：MENU → GRAPH → Y1 を “ax²+bx+c” の形で入力 → F6(DRAW) → SHIFT+F5(G-Solv)")
    return s

def format_numbered(items: List[str]) -> str:
    return "\n".join([f"{i+1}. {t}" for i, t in enumerate(items)])

# ==== テキストコマンド処理 ====
def try_parse_quadratic(text: str):
    t = text.replace("，", ",").replace("．", ".").replace("　", " ").strip()
    if not t.startswith("二次"):
        return None
    raw = t.replace("二次", "").strip()
    # 形式1: a=1 b=-3 c=2
    if "a=" in raw or "b=" in raw or "c=" in raw:
        vals = {}
        for part in raw.replace("＝", "=").split():
            if "=" in part:
                k, v = part.split("=", 1)
                try:
                    vals[k.strip()] = float(v.replace(",", ""))
                except:
                    pass
        if all(k in vals for k in ("a","b","c")):
            return vals["a"], vals["b"], vals["c"]
    # 形式2: 1,-3,2 / 1 -3 2 / 1.-3.2（区切りなんでも）
    raw2 = raw.replace(",", " ").replace("/", " ").replace("・"," ").replace(".."," ").replace("."," ").replace("　"," ")
    nums = []
    for p in raw2.split():
        try:
            nums.append(float(p))
        except:
            pass
    if len(nums) >= 3:
        return nums[0], nums[1], nums[2]
    return None

def build_operating_steps_for_quadratic(a, b, c):
    # “EQUA 法”と“GRAPH 法”の2通りを番号付きで返す（EXE込み）
    equa = steps_quadratic(a,b,c)
    graph = [
        "1. [MENU] → GRAPH → EXE（FUNC 画面へ。右下に DRAW が見えること）",
        "2. Y1 行へカーソル → F1(SELECT) で左の「＝」を濃く（ON）→ DEL で空に",
        "3. 次をそのまま入力 → [EXE]\n   [ALPHA][A] → [X,θ,T] → [x²] → [+] → [ALPHA][B] → [×] → [X,θ,T] → [+] → [ALPHA][C]",
        "   例: a=1, b=-3, c=2 なら 1×x² + (−3)×x + 2",
        "4. F6(DRAW) で描画",
        "5. SHIFT+F5(G-Solv) → [F1](ROOT) で解を確認（複数ある場合は → / ← で移動）",
    ]
    return equa, graph

# ==== 画像処理 & OpenAI Vision ====
def pil_enhance_for_ocr(b: bytes) -> bytes:
    img = Image.open(BytesIO(b)).convert("L")
    # 余白トリミング（ざっくり）
    img = ImageOps.expand(img, border=2, fill=255)
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Contrast(img).enhance(1.5)
    # A4でも読みやすいよう 200% 拡大（上限 2600px）
    w, h = img.size
    scale = 2.0 if max(w,h) < 1800 else 1.5
    img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()

async def openai_extract_two_problems(img_bytes: bytes) -> Dict:
    if not OPENAI_API_KEY:
        return {"status":"no_api_key"}
    data_url = "data:image/png;base64," + base64.b64encode(img_bytes).decode()
    sys = (
        "You extract at most TWO math problems from the image. "
        "Return JSON with problems:[{id,title,latex,kind,params,answer,explain}]. "
        "Keep it short; if numeric coefficients are readable, use them."
    )
    user = [
        {"type":"text","text":"Read up to TWO problems (A4 photo). For each: id like (5) or (8), latex equation or short statement, kind (quadratic/binomial/other), params, numeric answer if solvable, and a brief explanation."},
        {"type":"image_url","image_url":{"url": data_url}}
    ]
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": VISION_MODEL,
                "temperature": 0.1,
                "messages": [
                    {"role":"system","content":sys},
                    {"role":"user","content":user}
                ],
                "response_format": {"type":"json_object"}
            }
        )
        if r.status_code != 200:
            return {"status":"api_error","detail":r.text}
        try:
            js = r.json()["choices"][0]["message"]["content"]
            return {"status":"ok","data":json.loads(js)}
        except Exception as e:
            return {"status":"parse_error","detail":str(e), "raw":r.text[:500]}

def build_fx_steps_from_ai(problem: Dict) -> List[str]:
    kind = (problem.get("kind") or "").lower()
    steps = []
    if kind.startswith("quadratic") and problem.get("params"):
        a = problem["params"].get("a"); b = problem["params"].get("b"); c = problem["params"].get("c")
        if isinstance(a,(int,float)) and isinstance(b,(int,float)) and isinstance(c,(int,float)):
            equa, graph = build_operating_steps_for_quadratic(a,b,c)
            steps = ["【EQUA（方程式）】"] + [*equa] + ["", "【GRAPH（確認用）】"] + [*graph]
        else:
            steps = [
                "【GRAPH（文字係数のまま）】",
                "1. MENU → GRAPH → EXE",
                "2. Y1 行を ON → DEL で空",
                "3. y = −x² + 4ax + b （問題の式）をそのままキーで入力 → EXE",
                "4. F6(DRAW) → SHIFT+F5(G-Solv) → 頂点/ROOT などで確認",
                "5. 文字 a,b に具体値があるなら RUN-MAT で A,B を代入（0.5 → SHIFT+RCL → ALPHA A → EXE など）"
            ]
    elif kind.startswith("binomial") and problem.get("params"):
        steps = [
            "【RUN-MAT（確率）】",
            "1. MENU → RUN-MAT → EXE",
            "2. nCr( n , k ) × p^k × (1−p)^(n−k) をそのまま入力 → EXE",
            "   例：C(5,3)×(1/3)^3×(2/3)^2"
        ]
    else:
        steps = [
            "【一般】",
            "1. MENU → GRAPH または RUN-MAT",
            "2. 問題の式にあわせて入力 → EXE",
            "3. 必要に応じて G-Solv や Solve を使う"
        ]
    return steps

# ========= ルート =========
@app.get("/")
def root():
    return {"ok": True}

# ========= Webhook =========
@app.post("/webhook")
async def webhook(req: Request):
    # Verify 対策：空/非JSONでも 200
    try:
        body = await req.json()
    except:
        body = {}
    print("Webhook:", body)

    events = body.get("events", [])
    if not events:
        return {"status":"ok","note":"no events"}

    for ev in events:
        etype = ev.get("type")
        reply_token = ev.get("replyToken")
        src = ev.get("source", {})
        user_id = src.get("userId")

        # ===== テキスト =====
        if etype == "message" and ev.get("message", {}).get("type") == "text":
            text = ev["message"]["text"].strip()

            # 総合ガイド
            if text == "操作方法":
                for chunk in split_long_text(GUIDE):
                    await line_reply(reply_token, [{"type":"text","text":chunk}])
                continue

            # 二次 a,b,c
            coeffs = try_parse_quadratic(text)
            if coeffs:
                a,b,c = coeffs
                kind, D, roots = solve_quadratic(a,b,c)
                header = f"【二次方程式】a={a}, b={b}, c={c}\n判別式 D={D} → {kind}"
                if roots:
                    header += f"\n解: x1={roots[0]}, x2={roots[1]}"
                msgs = [{"type":"text","text": header}]
                equa, graph = build_operating_steps_for_quadratic(a,b,c)
                msgs += [{"type":"text","text": "《EQUA》\n" + format_numbered(equa)}]
                msgs += [{"type":"text","text": "《GRAPH》\n" + format_numbered(graph)}]
                await line_reply(reply_token, msgs[:5])
                continue

            # その他：使い方ヒント
            hint = "使い方：\n・画像を送る → 最大2問の『式/解/手順』を返します\n・「操作方法」→ 総合ガイド\n・「二次 a=1 b=-3 c=2」や「二次 1,-3,2」→ 解＋手順"
            await line_reply(reply_token, [{"type":"text","text":hint}])
            continue

        # ===== 画像 =====
        if etype == "message" and ev.get("message", {}).get("type") in ("image","file"):
            mid = ev["message"]["id"]
            # 先に即時返信（1秒制限回避）
            await line_reply(reply_token, [{"type":"text","text":"解析中… 少し待ってね（最大2問まで抽出）"}])

            # 画像取得
            if not LINE_TOKEN:
                if user_id:
                    await line_push(user_id, [{"type":"text","text":"サーバの設定エラー：LINE トークン未設定"}])
                continue
            content_url = f"https://api-data.line.me/v2/bot/message/{mid}/content"
            headers = {"Authorization": f"Bearer {LINE_TOKEN}"}
            try:
                async with httpx.AsyncClient(timeout=30) as c:
                    r = await c.get(content_url, headers=headers)
                    r.raise_for_status()
                    raw = r.content
            except Exception as e:
                if user_id:
                    await line_push(user_id, [{"type":"text","text":f"画像の取得に失敗しました: {e}"}])
                continue

            # 前処理→OpenAI Vision
            try:
                enhanced = pil_enhance_for_ocr(raw)
                ai = await openai_extract_two_problems(enhanced)
            except Exception as e:
                if user_id:
                    await line_push(user_id, [{"type":"text","text":f"内部エラー（前処理）: {e}"}])
                continue

            if ai.get("status") != "ok":
                if user_id:
                    detail = ai.get("detail","")
                    await line_push(user_id, [{"type":"text","text":f"解析に失敗しました。コントラストを強めて再送してください。\n{detail[:200]}"}])
                continue

            probs = ai["data"].get("problems", [])[:2]
            if not probs:
                if user_id:
                    await line_push(user_id, [{"type":"text","text":"式を特定できませんでした（最大2問）。余白を少なく、正面から撮って再送してね。"}])
                continue

            # 各問題を整形して送信
            for p in probs:
                title = p.get("id") or "(問題)"
                latex = p.get("latex") or p.get("title") or ""
                ans = p.get("answer") or ""
                exp = p.get("explain") or ""
                head = f"{title}\n[式] {latex}\n[答え] {ans}".strip()
                await line_push(user_id, [{"type":"text","text": head}])

                # 種別に応じて fx-CG50 手順生成（AI抽出の kind/params をなるべく使う）
                steps = build_fx_steps_from_ai(p)
                for chunk in split_long_text("《電卓手順》\n" + format_numbered(steps)):
                    await line_push(user_id, [{"type":"text","text": chunk}])

                if exp:
                    for chunk in split_long_text("《考え方》\n" + exp):
                        await line_push(user_id, [{"type":"text","text": chunk}])

            continue

    return {"status":"ok"}

