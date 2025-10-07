from fastapi import FastAPI, Request, Response
import os, httpx, json, re, math, base64, unicodedata, traceback

app = FastAPI()

# ---------- small utils ----------
def nk(s: str) -> str:
    if not isinstance(s, str): return s
    s = unicodedata.normalize("NFKC", s)
    for k,v in {"，":",","、":",","．":".","：":":","；":";","＝":"=","–":"-","—":"-","―":"-","−":"-","〜":"~"}.items():
        s = s.replace(k,v)
    return s

def fmt_num(x):
    if isinstance(x, complex):
        r = fmt_num(x.real); i = fmt_num(x.imag)
        sign = "+" if x.imag >= 0 else "-"
        return f"{r}{sign}{i}i"
    s = f"{float(x):.12g}".rstrip("0").rstrip(".")
    return s if s else "0"

def chunk_text(text: str, limit: int = 1800):
    parts, cur = [], ""
    for para in text.split("\n\n"):
        add = para if not cur else cur + "\n\n" + para
        if len(add) <= limit:
            cur = add
        else:
            if cur: parts.append(cur); cur = ""
            while len(para) > limit:
                parts.append(para[:limit]); para = para[limit:]
            cur = para
    if cur: parts.append(cur)
    return parts[:10]

# ---------- LINE send helpers ----------
def reply_texts(reply_token: str, texts):
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN","")
    if not token or not reply_token: return
    headers = {"Authorization": f"Bearer {token}", "Content-Type":"application/json"}
    msgs = []
    for t in texts:
        for c in chunk_text(str(t)):
            msgs.append({"type":"text","text": c})
    while msgs:
        batch = msgs[:5]; msgs = msgs[5:]
        body = {"replyToken": reply_token, "messages": batch}
        r = httpx.post("https://api.line.me/v2/bot/message/reply", headers=headers,
                       content=json.dumps(body))
        print("LINE reply status:", r.status_code, r.text)

def push_texts(user_id: str, texts):
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN","")
    if not token or not user_id: return
    headers = {"Authorization": f"Bearer {token}", "Content-Type":"application/json"}
    msgs = []
    for t in texts:
        for c in chunk_text(str(t)):
            msgs.append({"type":"text","text": c})
    while msgs:
        batch = msgs[:5]; msgs = msgs[5:]
        r = httpx.post("https://api.line.me/v2/bot/message/push", headers=headers,
                       content=json.dumps({"to": user_id, "messages": batch}))
        print("LINE push status:", r.status_code, r.text)

# ---------- Guides ----------
KEY_GUIDE = (
    "【fx-CG50：キー操作の基本（番号付き）】\n"
    "1.[EXE] 決定／確定\n"
    "2.[EXIT] 1つ戻る／メニュー\n"
    "3.[▲][▼][◀][▶] カーソル／解の切替\n"
    "4.白い「(−)」=負号、灰色「−」=引き算\n"
    "5.[DEL] 1文字削除、[AC/ON] 全消去/電源\n"
    "6.[MENU]→ **EQUATION（方程式）**\n"
    "7.サブメニュー\n"
    "   [F1] Simultaneous（連立：2元/3元）\n"
    "   [F2] Polynomial（多項式：次数）\n"
    "   [F3] Quadratic/Solve（機種差あり）\n"
    "   [F4] Complex/設定（虚数/設定）\n"
    "   [F5] Option/Format（表示・角度）\n"
    "   [F6] →/MORE/EXIT（右スクロール/さらに/戻る）"
)
QUAD_STEPS = (
    "【二次 aX²+bX+c=0（番号付き）】\n"
    "1.[MENU]→EQUATION\n"
    "2.[F2] Polynomial（または[F3] Quadratic）\n"
    "3.次数=2 を選択（表示される場合）\n"
    "4.a 入力→[EXE]\n"
    "5.b 入力→[EXE]\n"
    "6.c 入力→[EXE]\n"
    "7.解表示→[▲][▼]で x₁/x₂ 切替\n"
    "※負号は白い「(−)」、戻るは[EXIT]"
)
SIM2_STEPS = (
    "【連立（2元） ax+by=c / dx+ey=f】\n"
    "1.[MENU]→EQUATION\n"
    "2.[F1] Simultaneous → [F1] 2 Unknowns\n"
    "3.1行目 a→[EXE], b→[EXE], c→[EXE]\n"
    "4.2行目 d→[EXE], e→[EXE], f→[EXE]\n"
    "5.解 x,y を確認→[EXE]"
)
SIM3_STEPS = (
    "【連立（3元） ax+by+cz=d など】\n"
    "1.[MENU]→EQUATION\n"
    "2.[F1] Simultaneous → [F2] 3 Unknowns\n"
    "3.3行分の係数と定数を入力→[EXE]\n"
    "4.解 x,y,z を確認→[EXE]"
)
POLY_STEPS = (
    "【多項式（1変数） Aₙxⁿ+…+A₀=0】\n"
    "1.[MENU]→EQUATION\n"
    "2.[F2] Polynomial\n"
    "3.次数（n）を選択\n"
    "4.Aₙ, Aₙ₋₁, …, A₀ を順に入力→[EXE]\n"
    "5.根は[▲][▼]で切替→[EXE]"
)

# ---------- regex parsers ----------
NUM = r"[+-]?\s*(?:\d+(?:\.\d+)?|\.\d+)"
re_abc = re.compile(r"二次.*?a\s*=\s*("+NUM+").*?b\s*=\s*("+NUM+").*?c\s*=\s*("+NUM+")", re.I)
re_csv = re.compile(r"二次\s+("+NUM+")\s*,\s*("+NUM+")\s*,\s*("+NUM+")", re.I)

def parse_coeffs_quadratic(text: str):
    t = nk(text)
    m = re_abc.search(t) or re_csv.search(t)
    if not m: return None
    vals = []
    for x in m.groups(): vals.append(float(nk(x).replace(" ","")))
    return tuple(vals)

def parse_equation_quadratic(eq: str):
    t = nk(eq).replace(" ", "").lower()
    m = re.match(r"([+-]?(?:\d+(?:\.\d+)?|\.\d+)?)(x\^?2)"
                 r"([+-](?:\d+(?:\.\d+)?|\.\d+)?)(x)"
                 r"([+-](?:\d+(?:\.\d+)?|\.\d+)?)=0$", t)
    if not m: return None
    def coef(s):
        if s in ("","+"): return 1.0
        if s == "-": return -1.0
        return float(s)
    a = coef(m.group(1))
    b = float(m.group(3).replace("+"," +").replace("-"," -").split()[-1])
    c = float(m.group(5).replace("+"," +").replace("-"," -").split()[-1])
    return a,b,c

# ---------- solvers ----------
def solve_quadratic(a,b,c):
    if abs(a) < 1e-12:
        if abs(b) < 1e-12: return "退化（a=b=0）", []
        return "一次(特例)", [("x", -c/b)]
    D = b*b - 4*a*c
    if D > 0:
        s = math.sqrt(D); x1 = (-b + s)/(2*a); x2 = (-b - s)/(2*a)
        return "異なる実数解", [("x1", x1), ("x2", x2)]
    if abs(D) <= 1e-12:
        return "重解", [("x", (-b)/(2*a))]
    s = math.sqrt(-D); real = (-b)/(2*a); imag = s/(2*a)
    return "複素数解", [("x1", complex(real, imag)), ("x2", complex(real, -imag))]

# ---------- OpenAI Vision ----------
def analyze_math_image(image_bytes: bytes):
    api_key = os.environ.get("OPENAI_API_KEY","")
    if not api_key: return None
    b64 = base64.b64encode(image_bytes).decode("ascii")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
    system_prompt = (
        "Return ONLY valid JSON matching this schema:\n"
        "{ \"problems\": [ PROBLEM, ... ] }\n"
        "PROBLEM ∈ {\n"
        "  {\"type\":\"quadratic\",\"equation\":string,\"a\":number,\"b\":number,\"c\":number},\n"
        "  {\"type\":\"sim2\",\"equation\":string,\"a\":number,\"b\":number,\"c\":number,\"d\":number,\"e\":number,\"f\":number},\n"
        "  {\"type\":\"sim3\",\"equation\":string,\"M\":[[number,number,number],[number,number,number],[number,number,number]],\"v\":[number,number,number]},\n"
        "  {\"type\":\"linear\",\"equation\":string,\"b\":number,\"c\":number},\n"
        "  {\"type\":\"poly\",\"equation\":string,\"degree\":integer,\"coeffs\":{string:number}}\n"
        "}\n"
        "Detect the true type carefully. Do NOT coerce every problem into quadratic.\n"
        "Max 2 problems. If unreadable, return {\"problems\":[]}."
    )
    payload = {
        "model": "gpt-4o-mini",
        "response_format": {"type":"json_object"},
        "messages": [
            {"role":"system","content":system_prompt},
            {"role":"user","content":[
                {"type":"text","text":"Extract math problems as JSON only."},
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
            ]}
        ],
        "temperature": 0
    }
    try:
        r = httpx.post("https://api.openai.com/v1/chat/completions",
                       headers=headers, content=json.dumps(payload), timeout=90)
        print("OpenAI status:", r.status_code)
        data = r.json()
        content = data.get("choices",[{}])[0].get("message",{}).get("content","{}")
        # ここでJSON以外が来ても安全に
        try:
            info = json.loads(content)
        except Exception:
            print("JSON content parse failed; content head:", content[:120])
            info = {"problems":[]}
        if "problems" not in info: info["problems"] = []
        return info
    except Exception as e:
        print("OpenAI Vision error:", e)
        return None

def fallback_extract_equations(image_bytes: bytes):
    # どうしてもJSON化できない時の式抽出
    api_key = os.environ.get("OPENAI_API_KEY","")
    if not api_key: return []
    b64 = base64.b64encode(image_bytes).decode("ascii")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "response_format": {"type":"json_object"},
        "messages": [
            {"role":"system","content":"Return JSON only: {\"equations\":[string,...]} (max 3)."},
            {"role":"user","content":[
                {"type":"text","text":"List each visible math equation as plain ASCII, no commentary."},
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
            ]},
        ],
        "temperature": 0
    }
    try:
        r = httpx.post("https://api.openai.com/v1/chat/completions",
                       headers=headers, content=json.dumps(payload), timeout=60)
        data = r.json()
        eqs = json.loads(data["choices"][0]["message"]["content"]).get("equations",[])
        return [nk(e) for e in eqs][:3]
    except Exception as e:
        print("fallback equations error:", e)
        return []

# ---------- FastAPI ----------
@app.get("/")
def root():
    return {"status":"ok"}

@app.post("/webhook")
async def webhook(req: Request):
    # とにかく 200 を返す。中で例外が出ても落とさない。
    try:
        try:
            body = await req.json()
        except Exception:
            raw = await req.body()
            body = json.loads(raw) if raw else {}
        print("WEBHOOK:", body)

        events = body.get("events",[])
        if not events: return {"ok": True}
        ev = events[0]
        if ev.get("type") != "message": return {"ok": True}

        reply_token = ev.get("replyToken","")
        msg = ev.get("message",{})
        mtype = msg.get("type")
        user_id = ev.get("source",{}).get("userId","")

        # 画像
        if mtype == "image":
            reply_texts(reply_token, ["画像を受け取りました。解析中です…（完了後に結果を送ります）"])
            token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN","")
            if not token:
                push_texts(user_id, ["⚠ 環境変数 LINE_CHANNEL_ACCESS_TOKEN 未設定"])
                return {"ok": True}

            # 画像取得
            mid = msg.get("id")
            try:
                img = httpx.get(f"https://api-data.line.me/v2/bot/message/{mid}/content",
                                headers={"Authorization": f"Bearer {token}"}, timeout=60).content
            except Exception as e:
                print("LINE image fetch error:", e)
                push_texts(user_id, ["画像の取得に失敗しました。もう一度お送りください。"])
                return {"ok": True}

            # 解析
            info = analyze_math_image(img)
            problems = info["problems"] if info and "problems" in info else []

            # フォールバック
            if not problems:
                eqs = fallback_extract_equations(img)
                for e in eqs:
                    q = parse_equation_quadratic(e)
                    if q:
                        a,b,c = q
                        problems.append({"type":"quadratic","equation":e,"a":a,"b":b,"c":c})
                        break

            if not problems:
                push_texts(user_id, [
                    "画像から式を特定できませんでした。\n"
                    "・影/ブレ/斜めを減らし明るく撮影\n"
                    "・主問題だけを大きく撮影\n"
                    "テキスト入力例：二次 a=1 b=-3 c=2 / 二次 1,-3,2\n"
                    "キー一覧は「操作方法」で表示できます。"
                ])
                return {"ok": True}

            # 結果作成（最大2問）
            out_pages = []
            for i,p in enumerate(problems[:2], start=1):
                t = p.get("type"); eq = p.get("equation","(不明)")
                head = f"【問題{i}】種類：{t}\n式：{eq}"
                if t == "quadratic":
                    a=float(p["a"]); b=float(p["b"]); c=float(p["c"])
                    kind, sols = solve_quadratic(a,b,c)
                    ans = "\n".join([f"{n} = {fmt_num(v)}" for n,v in sols]) or "解なし/条件付き"
                    out_pages += chunk_text(head + f"\n判別：{kind}\n{ans}\n\n" + QUAD_STEPS +
                                            f"\n\n【今回の係数】a={fmt_num(a)}, b={fmt_num(b)}, c={fmt_num(c)}")
                elif t == "sim2":
                    a,b_,c,d,e,f_ = map(float,(p["a"],p["b"],p["c"],p["d"],p["e"],p["f"]))
                    # ここでは手順のみ（厳密解は別実装でもOK）
                    out_pages += chunk_text(head + "\n" + SIM2_STEPS +
                                            f"\n\n【今回の係数】\n"
                                            f" 1行目: a={fmt_num(a)}, b={fmt_num(b_)}, 定数={fmt_num(c)}\n"
                                            f" 2行目: d={fmt_num(d)}, e={fmt_num(e)}, 定数={fmt_num(f_)}")
                elif t == "sim3":
                    out_pages += chunk_text(head + "\n" + SIM3_STEPS)
                elif t == "poly":
                    out_pages += chunk_text(head + "\n" + POLY_STEPS)
                else:
                    out_pages += chunk_text(head + "\nこの種類は未対応です。")

            push_texts(user_id, out_pages)
            return {"ok": True}

        # テキスト
        if mtype == "text":
            text = nk(msg.get("text") or "")

            if text in ("操作方法","キー操作","ヘルプ","キー一覧"):
                reply_texts(reply_token, [KEY_GUIDE, QUAD_STEPS, SIM2_STEPS, SIM3_STEPS, POLY_STEPS])
                return {"ok": True}

            co = parse_coeffs_quadratic(text)
            if co:
                a,b,c = co
                kind, sols = solve_quadratic(a,b,c)
                eq = f"{fmt_num(a)}x^2 + {fmt_num(b)}x + {fmt_num(c)} = 0"
                ans = "\n".join([f"{n} = {fmt_num(v)}" for n,v in sols]) or "解なし/条件付き"
                reply_texts(reply_token, chunk_text("【二次】\n"+f"式：{eq}\n判別：{kind}\n{ans}\n\n"+QUAD_STEPS))
                return {"ok": True}

            reply_texts(reply_token, [(
                "使い方：\n"
                "1) 問題の写真を送る → 種類判定して『式＋答え＋番号付き手順』（最大2問）\n"
                "2) 二次：例）二次 a=1 b=-3 c=2 / 例）二次 1,-3,2\n"
                "3) キー操作の一覧：『操作方法』"
            )])
            return {"ok": True}

        return {"ok": True}

    except Exception as e:
        # ここで握り潰して 200 を返す（落とさない）
        print("UNCAUGHT ERROR:", e)
        print(traceback.format_exc())
        try:
            # 可能ならユーザーへ通知
            body = await req.json()
            ev = body.get("events",[{}])[0]
            user_id = ev.get("source",{}).get("userId","")
            push_texts(user_id, ["⚠ 内部エラーが発生しました。もう一度お試しください。"])
        except Exception:
            pass
        return Response(content=json.dumps({"ok": False}), media_type="application/json")
