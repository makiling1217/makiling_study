# main.py
from fastapi import FastAPI, Request
import os, httpx, json, re, math, base64

app = FastAPI()

# ========= 共通ユーティリティ =========
def fmt_num(x: float) -> str:
    s = f"{x:.12g}".rstrip("0").rstrip(".")
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

# ========= 定型メッセージ =========
KEY_GUIDE = (
    "【fx-CG50：キー操作の基本（番号付き）】\n"
    "1. [EXE] 決定／確定\n"
    "2. [EXIT] 1つ戻る／メニューへ戻る\n"
    "3. [▲][▼][◀][▶] カーソル移動／解の切替\n"
    "4. 白い「(−)」=負号、灰色「−」=引き算\n"
    "5. [DEL] 1文字削除、[AC/ON] 全消去/電源\n"
    "6. [MENU]→ **EQUATION**（方程式）\n"
    "7. サブメニュー： [F1] Simultaneous（連立）/ [F2] Quadratic（二次）/ [F3] Polynomial（多項式）\n"
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
    "【fx-CG50：連立（2元） ax+by=c / dx+ey=f（番号付き）】\n"
    "1.[MENU]→EQUATION\n"
    "2.[F1] Simultaneous\n"
    "3.[F1] 2 Unknowns を選ぶ\n"
    "4.1行目 a→[EXE], b→[EXE], c→[EXE]\n"
    "5.2行目 d→[EXE], e→[EXE], f→[EXE]\n"
    "6.解 x,y を確認→[EXE]"
)

SIM3_STEPS = (
    "【fx-CG50：連立（3元） ax+by+cz=d など（番号付き）】\n"
    "1.[MENU]→EQUATION\n"
    "2.[F1] Simultaneous\n"
    "3.[F2] 3 Unknowns を選ぶ\n"
    "4.各行で係数→[EXE]、定数→[EXE] を3行入力\n"
    "5.解 x,y,z を確認→[EXE]"
)

POLY_STEPS = (
    "【fx-CG50：多項式（1変数） Aₙxⁿ+…+A₀=0（番号付き）】\n"
    "1.[MENU]→EQUATION\n"
    "2.[F3] Polynomial を選ぶ\n"
    "3.次数（n）を選ぶ\n"
    "4.Aₙ, Aₙ₋₁, …, A₀ を順に入力→[EXE]\n"
    "5.根が表示→[▲][▼] で切替→[EXE]"
)

def send_key_guide(reply_token): reply_text(reply_token, KEY_GUIDE)
def send_quad_steps(reply_token): reply_text(reply_token, QUAD_STEPS)

# ========= 係数テキストの既存パース（二次） =========
NUM = r"[+-]?\s*(?:\d+(?:\.\d+)?|\.\d+)"
re_abc = re.compile(r"二次.*?a\s*=\s*("+NUM+").*?b\s*=\s*("+NUM+").*?c\s*=\s*("+NUM+")", re.I)
re_csv = re.compile(r"二次\s+("+NUM+")\s*[,，]\s*("+NUM+")\s*[,，]\s*("+NUM+")", re.I)

def parse_coeffs_quadratic(text: str):
    t = text.replace("　"," ")
    m = re_abc.search(t) or re_csv.search(t)
    if not m: return None
    return tuple(float(x.replace(" ", "")) for x in m.groups())

# ========= 数値計算 =========
def solve_quadratic(a,b,c):
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            info = "a=0,b=0 → c=0なら無数の解／c≠0なら解なし"
            return info, []
        x = -c/b
        return "一次（特例）", [("x", x)]
    D = b*b - 4*a*c
    if D > 0:
        s = math.sqrt(D); x1 = (-b + s)/(2*a); x2 = (-b - s)/(2*a)
        return "異なる実数解", [("x1", x1), ("x2", x2)]
    if abs(D) <= 1e-12:
        x = (-b)/(2*a)
        return "重解", [("x", x)]
    s = math.sqrt(-D); real = (-b)/(2*a); imag = s/(2*a)
    return "複素数解", [("x", complex(real, imag)), ("x", complex(real, -imag))]

def solve_sim2(a,b,c,d,e,f):
    det = a*e - b*d
    if abs(det) < 1e-12:
        return "行列式=0 → 解が一意に定まりません", []
    x = (c*e - b*f)/det
    y = (a*f - c*d)/det
    return "一意解", [("x", x), ("y", y)]

def solve_sim3(M, v):
    # M: 3x3、v: 3
    det = (
        M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
        - M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
        + M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0])
    )
    if abs(det) < 1e-12:
        return "行列式=0 → 解が一意に定まりません", []
    # クラメル
    def det3(A):
        return (
            A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
            - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
            + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0])
        )
    Mx = [[v[0],M[0][1],M[0][2]],[v[1],M[1][1],M[1][2]],[v[2],M[2][1],M[2][2]]]
    My = [[M[0][0],v[0],M[0][2]],[M[1][0],v[1],M[1][2]],[M[2][0],v[2],M[2][2]]]
    Mz = [[M[0][0],M[0][1],v[0]],[M[1][0],M[1][1],v[1]],[M[2][0],M[2][1],v[2]]]
    x = det3(Mx)/det; y = det3(My)/det; z = det3(Mz)/det
    return "一意解", [("x", x), ("y", y), ("z", z)]

# ========= 電卓の番号付きステップ生成 =========
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
    # coeffs_desc: [(power, value)] 例: [(3,A3), (2,A2), (1,A1), (0,A0)]
    lined = ", ".join([f"A{p}={fmt_num(v)}" for p, v in coeffs_desc])
    return POLY_STEPS + f"\n\n【今回の係数】{lined}"

# ========= OpenAI Vision で画像から式を抽出 =========
def analyze_math_image(image_bytes: bytes):
    """
    画像から問題タイプと係数を抽出。
    返り値の例:
    {"type":"quadratic", "equation":"x^2-3x+2=0", "a":1,"b":-3,"c":2}
    {"type":"linear", "equation":"2x+3=0", "b":2,"c":3}  # bx+c=0形式
    {"type":"sim2", "equation":"{ax+by=c; dx+ey=f}", "a":..., "b":..., "c":..., "d":..., "e":..., "f":...}
    {"type":"sim3", "M":[[a,b,c],[d,e,f],[g,h,i]], "v":[j,k,l], "equation":"..."}
    {"type":"poly", "degree":3, "coeffs": {"3":1,"2":-3,"1":0,"0":2}, "equation":"x^3-3x+2=0"}
    失敗時は None
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    b64 = base64.b64encode(image_bytes).decode("ascii")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    system_prompt = (
        "You extract math problems from a single photo. "
        "Return STRICT JSON only. Decide one of types: "
        "quadratic (a x^2 + b x + c = 0), "
        "linear (b x + c = 0), "
        "sim2 (two linear equations ax+by=c; dx+ey=f), "
        "sim3 (three linear equations), "
        "poly (univariate polynomial = 0 up to degree 6). "
        "Fields by type:\n"
        "- quadratic: {type,equation,a,b,c}\n"
        "- linear: {type,equation,b,c}\n"
        "- sim2: {type,equation,a,b,c,d,e,f}\n"
        "- sim3: {type,equation,M,v}  # M 3x3, v len 3\n"
        "- poly: {type,equation,degree,coeffs}  # coeffs is map power->value\n"
        "Numbers must be decimals (no fractions). If unreadable, answer {\"type\":\"unknown\"}."
    )
    user_text = "Extract structured coefficients for the main equation(s). JSON only."
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
        # JSONだけにしているが保険で抽出
        content = content.strip()
        # JSONでなければ落とす
        info = json.loads(content)
        if not isinstance(info, dict): return None
        return info
    except Exception as e:
        print("OpenAI Vision error:", e)
        return None

# ========= Webhook =========
@app.get("/")
def root(): return {"status":"ok"}

@app.post("/webhook")
async def webhook(req: Request):
    # Verify 対策
    try:
        raw = await req.body()
        body = json.loads(raw) if raw else {}
    except Exception:
        body = {}
    print("WEBHOOK:", body)

    events = body.get("events", [])
    if not events: return {"ok": True}
    ev = events[0]
    if ev.get("type") != "message": return {"ok": True}

    reply_token = ev.get("replyToken","")
    msg = ev.get("message", {}); mtype = msg.get("type")

    # ------- 画像：Vision → 解釈 → 手順＋式＋答え -------
    if mtype == "image":
        # LINE 画像を取得
        token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN","")
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
        if not info or info.get("type") in (None,"unknown"):
            # フォールバック：まずは二次の手順＋使い方
            reply_text(reply_token,
                QUAD_STEPS + "\n\n画像から式が読み取れませんでした。\n"
                "テキストでもOK：例）二次 a=1 b=-3 c=2 / 例）二次 1,-3,2\n"
                "一覧は「操作方法」で表示できます。")
            return {"ok": True}

        t = info.get("type")
        if t == "quadratic":
            a=float(info["a"]); b=float(info["b"]); c=float(info["c"])
            kind, sols = solve_quadratic(a,b,c)
            sol_txt = "\n".join([f"{name} = {fmt_num(val.real) if isinstance(val,complex) else fmt_num(val)}"
                                 + (f" ± {fmt_num(val.imag)}i" if isinstance(val,complex) and abs(val.imag)>1e-15 else "")
                                 for name,val in sols]) or "解なし/条件付き"
            text = (
                "【認識結果】二次方程式\n"
                f"式：{info.get('equation','(不明)')}\n"
                f"判別：{kind}\n{sol_txt}\n\n"
                + steps_quadratic(a,b,c)
            )
            reply_text(reply_token, text)
            return {"ok": True}

        if t == "linear":
            b=float(info["b"]); c=float(info["c"])
            if abs(b) < 1e-12:
                ans = "b=0 → c=0なら無数の解／c≠0なら解なし"
            else:
                x = -c/b; ans = f"x = {fmt_num(x)}"
            text = (
                "【認識結果】一次方程式（bx+c=0）\n"
                f"式：{info.get('equation','(不明)')}\n"
                f"解：{ans}\n\n"
                "＜電卓の操作（番号付き）＞\n"
                "1.[MENU]→EQUATION\n"
                "2.[F1] Linear（機種により Simultaneous 内に含まれる場合あり）\n"
                "3.b 入力→[EXE]\n"
                "4.c 入力→[EXE]\n"
                "5.解を確認→[EXE]\n"
                "※負号は白い「(−)」"
            )
            reply_text(reply_token, text)
            return {"ok": True}

        if t == "sim2":
            a,b,c,d,e,f = map(float, (info["a"],info["b"],info["c"],info["d"],info["e"],info["f"]))
            kind, sols = solve_sim2(a,b,c,d,e,f)
            sol_txt = "\n".join([f"{name} = {fmt_num(val)}" for name,val in sols]) or "解なし/条件付き"
            text = (
                "【認識結果】二元連立\n"
                f"式：{info.get('equation','(不明)')}\n"
                f"解の種別：{kind}\n{sol_txt}\n\n"
                + steps_sim2(a,b,c,d,e,f)
            )
            reply_text(reply_token, text)
            return {"ok": True}

        if t == "sim3":
            M = info["M"]; v = info["v"]
            M = [[float(x) for x in row] for row in M]
            v = [float(x) for x in v]
            kind, sols = solve_sim3(M, v)
            sol_txt = "\n".join([f"{name} = {fmt_num(val)}" for name,val in sols]) or "解なし/条件付き"
            text = (
                "【認識結果】三元連立\n"
                f"式：{info.get('equation','(不明)')}\n"
                f"解の種別：{kind}\n{sol_txt}\n\n"
                + steps_sim3(M, v)
            )
            reply_text(reply_token, text)
            return {"ok": True}

        if t == "poly":
            deg = int(info.get("degree", 0))
            coeffs_map = {int(k): float(v) for k,v in info.get("coeffs", {}).items()}
            # 並べ替え（高次→定数）
            coeffs_desc = sorted(coeffs_map.items(), key=lambda kv: -kv[0])
            # 簡易的に実根/複素根の列挙（解析はここでは省略し、電卓入力を重視）
            eq = info.get("equation","(不明)")
            text = (
                "【認識結果】多項式\n"
                f"式：{eq}\n"
                "（根の数値計算は電卓でご確認ください）\n\n" +
                steps_poly(deg, coeffs_desc)
            )
            reply_text(reply_token, text)
            return {"ok": True}

        # その他
        reply_text(reply_token, "画像を解析しましたがこの種類は未対応です。『操作方法』で一覧を表示できます。")
        return {"ok": True}

    # ------- テキスト：コマンド群 -------
    if mtype == "text":
        text = (msg.get("text") or "").strip()

        # 操作方法 → キー一覧
        if text in ("操作方法", "キー操作", "ヘルプ", "キー一覧"):
            send_key_guide(reply_token)
            return {"ok": True}

        # ガイド（種類別の手順だけ欲しい）
        if text in ("二次", "二次方程式", "手順"):
            send_quad_steps(reply_token); return {"ok": True}

        # 二次の係数（既存）
        co = parse_coeffs_quadratic(text)
        if co:
            a,b,c = co
            kind, sols = solve_quadratic(a,b,c)
            sol_txt = "\n".join([f"{name} = {fmt_num(val.real) if isinstance(val,complex) else fmt_num(val)}"
                                 + (f" ± {fmt_num(val.imag)}i" if isinstance(val,complex) and abs(val.imag)>1e-15 else "")
                                 for name,val in sols]) or "解なし/条件付き"
            reply_text(
                reply_token,
                f"【二次】a={fmt_num(a)}, b={fmt_num(b)}, c={fmt_num(c)}\n"
                f"判別：{kind}\n{sol_txt}\n\n" + steps_quadratic(a,b,c)
            )
            return {"ok": True}

        # 既定の案内
        reply_text(
            reply_token,
            "使い方：\n"
            "1) 問題の写真を送る → 種類を判定し、番号付きの電卓手順＋式＆答えを返信\n"
            "2) 二次をテキストで計算 → 例）二次 a=1 b=-3 c=2  または 例）二次 1,-3,2\n"
            "3) キー操作の一覧 → 「操作方法」"
        )
        return {"ok": True}

    return {"ok": True}
