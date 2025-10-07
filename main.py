# main.py
from fastapi import FastAPI, Request
import os, httpx, json, re, math, base64, unicodedata

app = FastAPI()

# ================= ユーティリティ =================
def nk(s: str) -> str:
    """全角→半角などNFKC正規化 & よく出る記号をASCIIへ寄せる"""
    if not isinstance(s, str): return s
    s = unicodedata.normalize("NFKC", s)
    table = {
        "，": ",", "、": ",", "．": ".", "：": ":", "；": ";",
        "＝": "=", "–": "-", "—": "-", "―": "-", "−": "-", "〜": "~"
    }
    for k, v in table.items():
        s = s.replace(k, v)
    return s

def fmt_num(x) -> str:
    if isinstance(x, complex):
        r = fmt_num(x.real)
        i = fmt_num(x.imag)
        sign = "+" if x.imag >= 0 else "-"
        return f"{r}{sign}{i}i"
    s = f"{float(x):.12g}".rstrip("0").rstrip(".")
    return s if s else "0"

def reply_text(reply_token: str, text: str):
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
    if not token:
        print("ERROR: LINE_CHANNEL_ACCESS_TOKEN 未設定")
        return
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body = {"replyToken": reply_token, "messages": [{"type": "text", "text": text}]}
    r = httpx.post("https://api.line.me/v2/bot/message/reply",
                   headers=headers, content=json.dumps(body))
    print("LINE reply status:", r.status_code, r.text)

# ================= 定型テキスト =================
KEY_GUIDE = (
    "【fx-CG50：キー操作の基本（番号付き）】\n"
    "1.[EXE] 決定／確定\n"
    "2.[EXIT] 1つ戻る／メニューに戻る\n"
    "3.[▲][▼][◀][▶] カーソル移動／解の切替\n"
    "4.白い「(−)」=負号、灰色「−」=引き算\n"
    "5.[DEL] 1文字削除、[AC/ON] 全消去/電源\n"
    "6.[MENU]→ **EQUATION**（方程式）\n"
    "7.サブメニュー：[F1] Simultaneous / [F2] Quadratic / [F3] Polynomial\n"
)

QUAD_STEPS = (
    "【fx-CG50：二次方程式 aX²+bX+c=0（番号付き）】\n"
    "1.[MENU]→EQUATION\n"
    "2.[F2] Quadratic を選ぶ\n"
    "3.a を入力→[EXE]\n"
    "4.b を入力→[EXE]\n"
    "5.c を入力→[EXE]\n"
    "6.解が表示→[▲][▼]で x₁/x₂ 切替→[EXE]\n"
    "7.負号は白い「(−)」、戻るは[EXIT]"
)

SIM2_STEPS = (
    "【fx-CG50：連立（2元） ax+by=c / dx+ey=f】\n"
    "1.[MENU]→EQUATION\n"
    "2.[F1] Simultaneous\n"
    "3.[F1] 2 Unknowns を選ぶ\n"
    "4.1行目 a→[EXE], b→[EXE], c→[EXE]\n"
    "5.2行目 d→[EXE], e→[EXE], f→[EXE]\n"
    "6.解 x,y を確認→[EXE]"
)

SIM3_STEPS = (
    "【fx-CG50：連立（3元） ax+by+cz=d など】\n"
    "1.[MENU]→EQUATION\n"
    "2.[F1] Simultaneous\n"
    "3.[F2] 3 Unknowns を選ぶ\n"
    "4.各行で係数→[EXE]、定数→[EXE] を3行入力\n"
    "5.解 x,y,z を確認→[EXE]"
)

POLY_STEPS = (
    "【fx-CG50：多項式（1変数） Aₙxⁿ+…+A₀=0】\n"
    "1.[MENU]→EQUATION\n"
    "2.[F3] Polynomial を選ぶ\n"
    "3.次数（n）を選ぶ\n"
    "4.Aₙ, Aₙ₋₁, …, A₀ を順に入力→[EXE]\n"
    "5.根が表示→[▲][▼] で切替→[EXE]"
)

# ================= テキスト入力（二次）パース =================
NUM = r"[+-]?\s*(?:\d+(?:\.\d+)?|\.\d+)"
re_abc = re.compile(r"二次.*?a\s*=\s*("+NUM+").*?b\s*=\s*("+NUM+").*?c\s*=\s*("+NUM+")", re.I)
re_csv = re.compile(r"二次\s+("+NUM+")\s*[,]\s*("+NUM+")\s*[,]\s*("+NUM+")", re.I)

def parse_coeffs_quadratic(text: str):
    t = nk(text)
    m = re_abc.search(t) or re_csv.search(t)
    if not m: return None
    vals = []
    for x in m.groups():
        vals.append(float(nk(x).replace(" ", "")))
    return tuple(vals)  # (a,b,c)

# ================= 数値計算 =================
def solve_quadratic(a,b,c):
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return "退化（a=b=0）", []
        return "一次(特例)", [("x", -c/b)]
    D = b*b - 4*a*c
    if D > 0:
        s = math.sqrt(D); x1 = (-b + s)/(2*a); x2 = (-b - s)/(2*a)
        return "異なる実数解", [("x1", x1), ("x2", x2)]
    if abs(D) <= 1e-12:
        x = (-b)/(2*a)
        return "重解", [("x", x)]
    s = math.sqrt(-D); real = (-b)/(2*a); imag = s/(2*a)
    return "複素数解", [("x1", complex(real, imag)), ("x2", complex(real, -imag))]

def solve_sim2(a,b,c,d,e,f):
    det = a*e - b*d
    if abs(det) < 1e-12:
        return "行列式=0 → 一意解なし", []
    x = (c*e - b*f)/det
    y = (a*f - c*d)/det
    return "一意解", [("x", x), ("y", y)]

def solve_sim3(M, v):
    def det3(A):
        return (
            A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
            - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
            + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0])
        )
    det = det3(M)
    if abs(det) < 1e-12:
        return "行列式=0 → 一意解なし", []
    Mx = [[v[0],M[0][1],M[0][2]],[v[1],M[1][1],M[1][2]],[v[2],M[2][1],M[2][2]]]
    My = [[M[0][0],v[0],M[0][2]],[M[1][0],v[1],M[1][2]],[M[2][0],v[2],M[2][2]]]
    Mz = [[M[0][0],M[0][1],v[0]],[M[1][0],M[1][1],v[1]],[M[2][0],M[2][1],v[2]]]
    x = det3(Mx)/det; y = det3(My)/det; z = det3(Mz)/det
    return "一意解", [("x", x), ("y", y), ("z", z)]

# ================= ステップ生成 =================
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
    lined = ", ".join([f"A{p}={fmt_num(v)}" for p, v in coeffs_desc])
    return POLY_STEPS + f"\n\n【今回の係数】{lined}"

# ================= 画像→式抽出（OpenAI Vision） =================
def extract_json_block(text: str):
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if m: return m.group(0)
    return text  # そもそもJSONだけならそのまま

def analyze_math_image(image_bytes: bytes):
    """
    画像から複数の問題を抽出して返す:
    { "problems": [
        {"type":"quadratic","equation":"x^2-3x+2=0","a":1,"b":-3,"c":2},
        {"type":"sim2","equation":"{2x+y=5; x-y=1}","a":2,"b":1,"c":5,"d":1,"e":-1,"f":1}
    ]}
    失敗時 None
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    b64 = base64.b64encode(image_bytes).decode("ascii")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    system_prompt = (
        "You read math problems from ONE photo. "
        "Return STRICT JSON ONLY with this schema:\n"
        "{ \"problems\": [ PROBLEM, ... ] }\n"
        "PROBLEM is one of:\n"
        "- quadratic: {type:\"quadratic\", equation, a, b, c}  # a x^2 + b x + c = 0\n"
        "- linear:    {type:\"linear\",    equation, b, c}     # b x + c = 0\n"
        "- sim2:      {type:\"sim2\",      equation, a,b,c,d,e,f}\n"
        "- sim3:      {type:\"sim3\",      equation, M, v}     # M 3x3, v len 3\n"
        "- poly:      {type:\"poly\",      equation, degree, coeffs}  # coeffs is map power->value\n"
        "If multiple problems exist, include up to 2 most prominent. "
        "If unreadable, return {\"problems\":[]}. Numbers must be decimals."
    )
    user_text = "Extract problems as structured JSON. JSON only (no prose)."
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role":"system","content":system_prompt},
            {"role":"user","content":[
                {"type":"text","text":user_text},
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
        j = extract_json_block(content)
        info = json.loads(j)
        if not isinstance(info, dict): return None
        if "problems" not in info: return None
        return info
    except Exception as e:
        print("OpenAI Vision error:", e)
        return None

# ================= FastAPI ルート =================
@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/webhook")
async def webhook(req: Request):
    # Verify対策：空や非JSONでも200
    try:
        raw = await req.body()
        body = json.loads(raw) if raw else {}
    except Exception:
        body = {}
    print("WEBHOOK:", body)

    events = body.get("events", [])
    if not events:
        return {"ok": True}

    ev = events[0]
    if ev.get("type") != "message":
        return {"ok": True}

    reply_token = ev.get("replyToken", "")
    msg = ev.get("message", {})
    mtype = msg.get("type")

    # ---------- 画像 ----------
    if mtype == "image":
        token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
        if not token:
            reply_text(reply_token, "⚠ 環境変数が未設定です（LINE_CHANNEL_ACCESS_TOKEN）")
            return {"ok": True}
        mid = msg.get("id")
        try:
            img = httpx.get(
                f"https://api-data.line.me/v2/bot/message/{mid}/content",
                headers={"Authorization": f"Bearer {token}"}, timeout=60
            ).content
        except Exception as e:
            print("LINE image fetch error:", e)
            reply_text(reply_token, "画像の取得に失敗しました。もう一度お送りください。")
            return {"ok": True}

        info = analyze_math_image(img)
        if not info or not info.get("problems"):
            # 読めなかったら丁寧にフォールバック
            reply_text(
                reply_token,
                "画像から式を特定できませんでした。\n"
                "・影/ブレ/斜めを減らし明るく撮影\n"
                "・問題が複数ならメイン問題を拡大\n\n"
                "テキストでもOK：例）二次 a=1 b=-3 c=2 / 例）二次 1,-3,2\n"
                "一覧は「操作方法」で表示できます。"
            )
            return {"ok": True}

        # 最大2問に制限して返信をまとめる
        out_lines = []
        count = 0
        for p in info["problems"]:
            if count >= 2: break
            t = p.get("type")
            eq = p.get("equation", "(不明)")
            count += 1
            header = f"【問題{count}】種別：{t}\n式：{eq}"
            if t == "quadratic":
                a=float(p["a"]); b=float(p["b"]); c=float(p["c"])
                kind, sols = solve_quadratic(a,b,c)
                ans = "\n".join([f"{name} = {fmt_num(v)}" for name,v in sols]) or "解なし/条件付き"
                out_lines.append(header + f"\n判別：{kind}\n{ans}\n\n" + steps_quadratic(a,b,c))
            elif t == "linear":
                b=float(p["b"]); c=float(p["c"])
                if abs(b) < 1e-12:
                    ans = "b=0 → c=0なら無数の解／c≠0なら解なし"
                else:
                    ans = f"x = {fmt_num(-c/b)}"
                steps = (
                    "【一次（bx+c=0）の操作】\n"
                    "1.[MENU]→EQUATION\n"
                    "2.（機種により Simultaneous 内）Linear を選択\n"
                    "3.b 入力→[EXE]、c 入力→[EXE]\n"
                    "4.解を確認→[EXE]（負号は白い「(−)」）"
                )
                out_lines.append(header + f"\n解：{ans}\n\n" + steps)
            elif t == "sim2":
                a,b_,c,d,e,f = map(float, (p["a"],p["b"],p["c"],p["d"],p["e"],p["f"]))
                kind, sols = solve_sim2(a,b_,c,d,e,f)
                ans = "\n".join([f"{name} = {fmt_num(v)}" for name,v in sols]) or "解なし/条件付き"
                out_lines.append(header + f"\n解の種別：{kind}\n{ans}\n\n" + steps_sim2(a,b_,c,d,e,f))
            elif t == "sim3":
                M = [[float(x) for x in row] for row in p["M"]]
                v = [float(x) for x in p["v"]]
                kind, sols = solve_sim3(M, v)
                ans = "\n".join([f"{name} = {fmt_num(vv)}" for name,vv in sols]) or "解なし/条件付き"
                out_lines.append(header + f"\n解の種別：{kind}\n{ans}\n\n" + steps_sim3(M, v))
            elif t == "poly":
                deg = int(p.get("degree", 0))
                coeffs_map = {int(k): float(v) for k,v in p.get("coeffs", {}).items()}
                coeffs_desc = sorted(coeffs_map.items(), key=lambda kv: -kv[0])
                out_lines.append(header + "\n（数値解の列挙は電卓で確認）\n\n" + steps_poly(deg, coeffs_desc))
            else:
                out_lines.append(header + "\nこの種類は未対応です。")

        reply_text(reply_token, "\n\n".join(out_lines))
        return {"ok": True}

    # ---------- テキスト ----------
    if mtype == "text":
        text = nk(msg.get("text") or "")

        # 操作方法（総合ガイド）
        if text in ("操作方法","キー操作","ヘルプ","キー一覧"):
            reply_text(reply_token, KEY_GUIDE)
            return {"ok": True}

        # 「二次 ...」係数パース
        co = parse_coeffs_quadratic(text)
        if co:
            a,b,c = co
            kind, sols = solve_quadratic(a,b,c)
            eq = f"{fmt_num(a)}x^2 + {fmt_num(b)}x + {fmt_num(c)} = 0"
            ans = "\n".join([f"{name} = {fmt_num(v)}" for name,v in sols]) or "解なし/条件付き"
            reply_text(
                reply_token,
                "【二次】\n"
                f"式：{eq}\n"
                f"判別：{kind}\n{ans}\n\n" + steps_quadratic(a,b,c)
            )
            return {"ok": True}

        # 既定の案内
        reply_text(
            reply_token,
            "使い方：\n"
            "1) 問題の写真を送る → 種類を判定し、番号付きの電卓手順＋式＆答えを返信（最大2問）\n"
            "2) 二次をテキストで計算 → 例）二次 a=1 b=-3 c=2  または 例）二次 1,-3,2\n"
            "3) キー操作の一覧 → 「操作方法」"
        )
        return {"ok": True}

    return {"ok": True}
