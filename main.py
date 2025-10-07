# main.py
from fastapi import FastAPI, Request
import os, httpx, json, re, math, base64, unicodedata

app = FastAPI()

# ----------------- Utilities -----------------
def nk(s: str) -> str:
    if not isinstance(s, str): return s
    s = unicodedata.normalize("NFKC", s)
    table = {"，":",","、":",","．":".","：":":","；":";","＝":"=","–":"-","—":"-","―":"-","−":"-","〜":"~"}
    for k,v in table.items(): s = s.replace(k,v)
    return s

def fmt_num(x):
    if isinstance(x, complex):
        r = fmt_num(x.real); i = fmt_num(x.imag)
        sign = "+" if x.imag >= 0 else "-"
        return f"{r}{sign}{i}i"
    s = f"{float(x):.12g}".rstrip("0").rstrip(".")
    return s if s else "0"

def reply_text(reply_token: str, text: str):
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN","")
    if not token:
        print("ERROR: LINE_CHANNEL_ACCESS_TOKEN 未設定"); return
    headers = {"Authorization": f"Bearer {token}","Content-Type":"application/json"}
    body = {"replyToken": reply_token, "messages":[{"type":"text","text": text}]}
    r = httpx.post("https://api.line.me/v2/bot/message/reply", headers=headers, content=json.dumps(body))
    print("LINE reply status:", r.status_code, r.text)

# ----------------- Guides -----------------
KEY_GUIDE = (
    "【fx-CG50：キー操作の基本（番号付き）】\n"
    "1.[EXE] 決定／確定\n"
    "2.[EXIT] 1つ戻る／メニューに戻る\n"
    "3.[▲][▼][◀][▶] カーソル移動／解の切替\n"
    "4.白い「(−)」=負号、灰色「−」=引き算\n"
    "5.[DEL] 1文字削除、[AC/ON] 全消去/電源\n"
    "6.[MENU]→ **EQUATION（方程式）**\n"
    "7.サブメニュー（機種差あり）\n"
    "   [F1] Simultaneous（連立：2元/3元）\n"
    "   [F2] Polynomial（多項式：次数を選択）\n"
    "   [F3] Solve / Quadratic（機種により直接解法/二次）\n"
    "   [F4] Complex/設定（虚数モード切替など：表示時のみ）\n"
    "   [F5] Option（表示形式/角度など：表示時のみ）\n"
    "   [F6] →/MORE/EXIT（右スクロール/さらに表示/戻る：画面右端）"
)

QUAD_STEPS = (
    "【二次 aX²+bX+c=0（番号付き）】\n"
    "1.[MENU]→EQUATION\n"
    "2.[F2] Polynomial（または [F3] Quadratic）\n"
    "3.次数=2 を選ぶ（表示される場合）\n"
    "4.a を入力→[EXE]\n"
    "5.b を入力→[EXE]\n"
    "6.c を入力→[EXE]\n"
    "7.解が表示→[▲][▼]で x₁/x₂ 切替→[EXE]\n"
    "※負号は白い「(−)」、戻るは[EXIT]"
)

SIM2_STEPS = (
    "【連立（2元） ax+by=c / dx+ey=f】\n"
    "1.[MENU]→EQUATION\n"
    "2.[F1] Simultaneous\n"
    "3.[F1] 2 Unknowns を選ぶ\n"
    "4.1行目 a→[EXE], b→[EXE], c→[EXE]\n"
    "5.2行目 d→[EXE], e→[EXE], f→[EXE]\n"
    "6.解 x,y を確認→[EXE]"
)

SIM3_STEPS = (
    "【連立（3元） ax+by+cz=d など】\n"
    "1.[MENU]→EQUATION\n"
    "2.[F1] Simultaneous\n"
    "3.[F2] 3 Unknowns を選ぶ\n"
    "4.各行で係数→[EXE]、定数→[EXE] を3行入力\n"
    "5.解 x,y,z を確認→[EXE]"
)

POLY_STEPS = (
    "【多項式（1変数） Aₙxⁿ+…+A₀=0】\n"
    "1.[MENU]→EQUATION\n"
    "2.[F2] Polynomial を選ぶ\n"
    "3.次数（n）を選ぶ\n"
    "4.Aₙ, Aₙ₋₁, …, A₀ を順に入力→[EXE]\n"
    "5.根が表示→[▲][▼] で切替→[EXE]"
)

# ----------------- Parsers -----------------
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

# fallback: parse “ax^2+bx+c=0”
def parse_equation_quadratic(eq: str):
    t = nk(eq).replace(" ", "").lower()
    # 係数省略(1/-1)対応
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

def parse_equation_sim2(eq: str):
    t = nk(eq).replace(" ", "").lower()
    m = re.match(r"([+-]?(?:\d+(?:\.\d+)?|\.\d+)?)(x)"
                 r"([+-](?:\d+(?:\.\d+)?|\.\d+)?)(y)=(?:"
                 r"([+-]?(?:\d+(?:\.\d+)?|\.\d+)?))$", t)
    if not m: return None
    def coef(s):
        if s in ("","+"): return 1.0
        if s == "-": return -1.0
        return float(s)
    a = coef(m.group(1))
    b = float(m.group(3).replace("+"," +").replace("-"," -").split()[-1])
    c = float(m.group(5))
    return a,b,c

# ----------------- Solvers -----------------
def solve_quadratic(a,b,c):
    if abs(a) < 1e-12:
        if abs(b) < 1e-12: return "退化（a=b=0）", []
        return "一次(特例)", [("x", -c/b)]
    D = b*b - 4*a*c
    if D > 0:
        s = math.sqrt(D); x1 = (-b + s)/(2*a); x2 = (-b - s)/(2*a)
        return "異なる実数解", [("x1", x1), ("x2", x2)]
    if abs(D) <= 1e-12:
        x = (-b)/(2*a); return "重解", [("x", x)]
    s = math.sqrt(-D); real = (-b)/(2*a); imag = s/(2*a)
    return "複素数解", [("x1", complex(real, imag)), ("x2", complex(real, -imag))]

def solve_sim2(a,b,c,d,e,f):
    det = a*e - b*d
    if abs(det) < 1e-12: return "行列式=0 → 一意解なし", []
    x = (c*e - b*f)/det
    y = (a*f - c*d)/det
    return "一意解", [("x", x), ("y", y)]

def solve_sim3(M, v):
    def det3(A):
        return (A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
              - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
              + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]))
    det = det3(M)
    if abs(det) < 1e-12: return "行列式=0 → 一意解なし", []
    Mx = [[v[0],M[0][1],M[0][2]],[v[1],M[1][1],M[1][2]],[v[2],M[2][1],M[2][2]]]
    My = [[M[0][0],v[0],M[0][2]],[M[1][0],v[1],M[1][2]],[M[2][0],v[2],M[2][2]]]
    Mz = [[M[0][0],M[0][1],v[0]],[M[1][0],M[1][1],v[1]],[M[2][0],M[2][1],v[2]]]
    def d(A): return det3(A)
    x = d(Mx)/det; y = d(My)/det; z = d(Mz)/det
    return "一意解", [("x", x), ("y", y), ("z", z)]

# ----------------- Step generators -----------------
def steps_quadratic(a,b,c):
    return QUAD_STEPS + f"\n\n【今回の係数】a={fmt_num(a)}, b={fmt_num(b)}, c={fmt_num(c)}"

def steps_sim2(a,b,c,d,e,f):
    return SIM2_STEPS + (
        f"\n\n【今回の係数】\n"
        f" 1行目: a={fmt_num(a)}, b={fmt_num(b)}, 定数={fmt_num(c)}\n"
        f" 2行目: d={fmt_num(d)}, e={fmt_num(e)}, 定数={fmt_num(f)}"
    )

def steps_sim3(M, v):
    s = (
        f"1行目: {fmt_num(M[0][0])}, {fmt_num(M[0][1])}, {fmt_num(M[0][2])}, 定数={fmt_num(v[0])}\n"
        f"2行目: {fmt_num(M[1][0])}, {fmt_num(M[1][1])}, {fmt_num(M[1][2])}, 定数={fmt_num(v[1])}\n"
        f"3行目: {fmt_num(M[2][0])}, {fmt_num(M[2][1])}, {fmt_num(M[2][2])}, 定数={fmt_num(v[2])}"
    )
    return SIM3_STEPS + "\n\n【今回の係数】\n" + s

def steps_poly(deg, coeffs_desc):
    lined = ", ".join([f"A{p}={fmt_num(v)}" for p,v in coeffs_desc])
    return POLY_STEPS + f"\n\n【今回の係数】{lined}"

# ----------------- OpenAI Vision (JSON mode) -----------------
def analyze_math_image(image_bytes: bytes):
    """
    画像→最大2問を抽出:
    { "problems":[
        {"type":"quadratic","equation":"x^2-3x+2=0","a":1,"b":-3,"c":2},
        {"type":"sim2","equation":"2x+y=5; x-y=1","a":2,"b":1,"c":5,"d":1,"e":-1,"f":1}
    ]}
    """
    api_key = os.environ.get("OPENAI_API_KEY","")
    if not api_key: return None
    b64 = base64.b64encode(image_bytes).decode("ascii")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
    system_prompt = (
        "You are a strict JSON generator. "
        "Return ONLY valid JSON that matches exactly this schema:\n"
        "{ \"problems\": [ PROBLEM, ... ] }\n"
        "PROBLEM ∈ {\n"
        "  {\"type\":\"quadratic\",\"equation\":string,\"a\":number,\"b\":number,\"c\":number},\n"
        "  {\"type\":\"linear\",\"equation\":string,\"b\":number,\"c\":number},\n"
        "  {\"type\":\"sim2\",\"equation\":string,\"a\":number,\"b\":number,\"c\":number,\"d\":number,\"e\":number,\"f\":number},\n"
        "  {\"type\":\"sim3\",\"equation\":string,\"M\":[[number,number,number],[number,number,number],[number,number,number]],\"v\":[number,number,number]},\n"
        "  {\"type\":\"poly\",\"equation\":string,\"degree\":integer,\"coeffs\":{string:number}}\n"
        "}\n"
        "Detect the true type carefully. Never coerce into quadratic. "
        "Include up to 2 most prominent problems. If unreadable, return {\"problems\":[]}. "
        "No markdown, no prose."
    )
    payload = {
        "model": "gpt-4o-mini",
        "response_format": {"type":"json_object"},
        "messages": [
            {"role":"system","content":system_prompt},
            {"role":"user","content":[
                {"type":"text","text":"Extract as JSON only."},
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
            ]}
        ],
        "temperature": 0
    }
    try:
        r = httpx.post("https://api.openai.com/v1/chat/completions",
                       headers=headers, content=json.dumps(payload), timeout=60)
        print("OpenAI status:", r.status_code)
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        info = json.loads(content)  # JSON modeなのでここで必ずJSON
        if "problems" not in info: return {"problems":[]}
        return info
    except Exception as e:
        print("OpenAI Vision error:", e)
        return None

# フォールバック：式の配列を抜いてから自前で解析
def fallback_extract_equations(image_bytes: bytes):
    api_key = os.environ.get("OPENAI_API_KEY","")
    if not api_key: return []
    b64 = base64.b64encode(image_bytes).decode("ascii")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "response_format": {"type":"json_object"},
        "messages": [
            {"role":"system","content":"Return JSON only: {\"equations\":[string,...]} (max 3). No prose."},
            {"role":"user","content":[
                {"type":"text","text":"List each visible math equation as plain ASCII, one per item."},
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
            ]}
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

# ----------------- FastAPI -----------------
@app.get("/")
def root(): return {"status":"ok"}

@app.post("/webhook")
async def webhook(req: Request):
    # Verify対策：空/非JSONでも200
    try:
        raw = await req.body()
        body = json.loads(raw) if raw else {}
    except Exception:
        body = {}
    print("WEBHOOK:", body)

    events = body.get("events",[])
    if not events: return {"ok": True}
    ev = events[0]
    if ev.get("type") != "message": return {"ok": True}

    reply_token = ev.get("replyToken","")
    msg = ev.get("message",{})
    mtype = msg.get("type")

    # ---------- image ----------
    if mtype == "image":
        token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN","")
        if not token:
            reply_text(reply_token, "⚠ 環境変数が未設定です（LINE_CHANNEL_ACCESS_TOKEN）"); return {"ok": True}
        mid = msg.get("id")
        try:
            img = httpx.get(f"https://api-data.line.me/v2/bot/message/{mid}/content",
                            headers={"Authorization": f"Bearer {token}"}, timeout=60).content
        except Exception as e:
            print("LINE image fetch error:", e)
            reply_text(reply_token, "画像の取得に失敗しました。もう一度お送りください。"); return {"ok": True}

        info = analyze_math_image(img)
        problems = info["problems"] if info and "problems" in info else []

        # フォールバック：式だけ取って自前解析
        if not problems:
            eqs = fallback_extract_equations(img)
            pack = []
            # 2元連立（2式）を優先して拾う
            sim_candidates = []
            for e in eqs:
                s2 = parse_equation_sim2(e)
                if s2: sim_candidates.append((e, s2))
            if len(sim_candidates) >= 2:
                (e1,(a,b,c)) = sim_candidates[0]
                (e2,(d,e,f)) = sim_candidates[1]
                problems.append({"type":"sim2","equation":f"{e1}; {e2}",
                                 "a":a,"b":b,"c":c,"d":d,"e":e,"f":f})
            else:
                # 二次を拾う
                for e in eqs:
                    q = parse_equation_quadratic(e)
                    if q:
                        a,b,c = q
                        problems.append({"type":"quadratic","equation":e,"a":a,"b":b,"c":c})
                        break

        if not problems:
            reply_text(
                reply_token,
                "画像から式を特定できませんでした。\n"
                "・影/ブレ/斜めを減らして明るく撮影\n"
                "・問題が複数ならメイン問題を拡大\n"
                "テキストでもOK：例）二次 a=1 b=-3 c=2 / 例）二次 1,-3,2\n"
                "一覧は「操作方法」で表示できます。"
            )
            return {"ok": True}

        out = []
        for i,p in enumerate(problems[:2], start=1):
            t = p.get("type"); eq = p.get("equation","(不明)")
            head = f"【問題{i}】種別：{t}\n式：{eq}"
            if t == "quadratic":
                a=float(p["a"]); b=float(p["b"]); c=float(p["c"])
                kind, sols = solve_quadratic(a,b,c)
                ans = "\n".join([f"{n} = {fmt_num(v)}" for n,v in sols]) or "解なし/条件付き"
                out.append(head + f"\n判別：{kind}\n{ans}\n\n" + steps_quadratic(a,b,c))
            elif t == "linear":
                b=float(p["b"]); c=float(p["c"])
                if abs(b) < 1e-12:
                    ans = "b=0 → c=0なら無数／c≠0なら解なし"
                else:
                    ans = f"x = {fmt_num(-c/b)}"
                steps = ("【一次（bx+c=0）】\n1.[MENU]→EQUATION\n2.Linear（機種差あり）\n3.b 入力→[EXE], c 入力→[EXE]\n4.解→[EXE]")
                out.append(head + f"\n解：{ans}\n\n" + steps)
            elif t == "sim2":
                a,b_,c,d,e,f = map(float, (p["a"],p["b"],p["c"],p["d"],p["e"],p["f"]))
                kind, sols = solve_sim2(a,b_,c,d,e,f)
                ans = "\n".join([f"{n} = {fmt_num(v)}" for n,v in sols]) or "解なし/条件付き"
                out.append(head + f"\n解の種別：{kind}\n{ans}\n\n" + steps_sim2(a,b_,c,d,e,f))
            elif t == "sim3":
                M = [[float(x) for x in row] for row in p["M"]]; v = [float(x) for x in p["v"]]
                kind, sols = solve_sim3(M, v)
                ans = "\n".join([f"{n} = {fmt_num(vv)}" for n,vv in sols]) or "解なし/条件付き"
                out.append(head + f"\n解の種別：{kind}\n{ans}\n\n" + steps_sim3(M, v))
            elif t == "poly":
                deg = int(p.get("degree",0))
                coeffs = {int(k): float(v) for k,v in p.get("coeffs",{}).items()}
                desc = sorted(coeffs.items(), key=lambda kv: -kv[0])
                out.append(head + "\n（数値解は電卓で確認）\n\n" + steps_poly(deg, desc))
            else:
                out.append(head + "\nこの種類は未対応です。")
        reply_text(reply_token, "\n\n".join(out))
        return {"ok": True}

    # ---------- text ----------
    if mtype == "text":
        text = nk(msg.get("text") or "")

        if text in ("操作方法","キー操作","ヘルプ","キー一覧"):
            reply_text(reply_token, KEY_GUIDE); return {"ok": True}

        co = parse_coeffs_quadratic(text)
        if co:
            a,b,c = co
            kind, sols = solve_quadratic(a,b,c)
            eq = f"{fmt_num(a)}x^2 + {fmt_num(b)}x + {fmt_num(c)} = 0"
            ans = "\n".join([f"{n} = {fmt_num(v)}" for n,v in sols]) or "解なし/条件付き"
            reply_text(reply_token, "【二次】\n"+f"式：{eq}\n判別：{kind}\n{ans}\n\n"+steps_quadratic(a,b,c))
            return {"ok": True}

        reply_text(
            reply_token,
            "使い方：\n"
            "1) 問題の写真を送る → 種類判定して「式＋答え＋番号付き手順」（最大2問）\n"
            "2) 二次：例）二次 a=1 b=-3 c=2 / 例）二次 1,-3,2\n"
            "3) キー操作の一覧：『操作方法』"
        )
        return {"ok": True}

    return {"ok": True}
