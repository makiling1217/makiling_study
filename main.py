# main.py  ← まるごと置き換え

from fastapi import FastAPI, Request
import os, json, re, math, base64, asyncio
import httpx
from typing import List, Dict, Any

# ==== 設定 ====
LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

app = FastAPI()

# ---------- ユーティリティ ----------
def chunks(s: str, size: int = 900) -> List[str]:
    """LINEの1メッセージが長すぎると見づらいので小分け送信"""
    out = []
    buf = s.strip()
    while len(buf) > size:
        cut = buf.rfind("\n", 0, size)
        if cut <= 0:
            cut = size
        out.append(buf[:cut])
        buf = buf[cut:]
    if buf:
        out.append(buf)
    return out

def fmt_num(x: float) -> str:
    if math.isfinite(x) and abs(x - round(x)) < 1e-10:
        return str(int(round(x)))
    return f"{x:.6g}"

def steps_fx_cg50_quadratic(a: float, b: float, c: float) -> str:
    return (
        "【fx-CG50：二次方程式の解き方（番号付き）】\n"
        "1. 本体の電源を入れる。\n"
        "2. モード画面で「EQUATION（方程式）」を選ぶ（十字キーで移動→[EXE]）。\n"
        "3. 画面下のソフトキーで [F2] Polynomial（多項式）を選ぶ。\n"
        "4. 次に次数を選ぶ画面で「Degree 2」を選んで [EXE]。\n"
        f"5. 係数 a に「{fmt_num(a)}」を入力して [EXE]。\n"
        f"6. 係数 b に「{fmt_num(b)}」を入力して [EXE]。\n"
        f"7. 係数 c に「{fmt_num(c)}」を入力して [EXE]。\n"
        "   ※負の数は青い「(−)」キー（小さなマイナス）を使う。引き算は白い「−」キー。\n"
        "8. 画面下の [F6] Solve/Next（表示名は機種やOSで多少異なる）で解を表示。\n"
        "9. もう片方の解は [F6] Next（または [▶]）で確認。\n"
        "10. 終了は [EXIT]。"
    )

def steps_fx_cg50_linear(a: float, b: float) -> str:
    return (
        "【fx-CG50：一次方程式 ax+b=0 の解き方（番号付き）】\n"
        "1. モード画面で「EQUATION（方程式）」を選び [EXE]。\n"
        "2. [F2] Polynomial → 次数は「Degree 1」を選ぶ。\n"
        f"3. 係数 a に「{fmt_num(a)}」→ [EXE]。\n"
        f"4. 係数 b に「{fmt_num(b)}」→ [EXE]。\n"
        "5. [F6] Solve/Next で解を表示。\n"
        "6. 終了は [EXIT]。"
    )

HELP_KEYS = (
    "【fx-CG50 キー操作の総合ガイド】\n"
    "1. モード選択：十字キーでアプリを選び [EXE]。方程式は「EQUATION」。\n"
    "2. [F1]〜[F6]：画面下のソフトキー。場面で表示名が変わる（例：F2 Polynomial / F6 Solve / Next など）。\n"
    "3. 数字入力：テンキー。小数は [.]。分数は [a b/c] でも可（必要に応じて）。\n"
    "4. 負の数：青い「(−)」キー（小さなマイナス）。引き算は白い「−」。\n"
    "5. [DEL] 1字消去 / [AC] 全消去 / [EXIT] 一つ戻る。\n"
    "6. 方程式：EQUATION → [F2] Polynomial → 次数（Degree）を選ぶ → 係数を順に入力 → [F6] Solve/Next。\n"
    "7. その他のFキー例：\n"
    "   ・[F1] Simultaneous（連立）/場面別の切替\n"
    "   ・[F2] Polynomial（多項式）\n"
    "   ・[F3] Solver（数値解法）\n"
    "   ・[F4]/[F5] は項目切替・タイプ変更等に割り当ての場合あり\n"
    "   ・[F6] Solve / Next / EXE 相当（結果表示・次項目へ）\n"
)

def quick_usage() -> str:
    return (
        "使い方：\n"
        "1) 問題の写真を送る → 画像解析して『式＋答え＋番号付き手順』（最大2問）を返します。\n"
        "2) 係数を直接送る：\n"
        "   例）二次 1,-3,2 / 二次1,-3,2 / 二次1 -3 2\n"
        "3) 一覧ガイド：『操作方法』と送信するとキー操作の総合ガイドを返します。"
    )

# ---------- LINE 送信 ----------
async def line_reply(reply_token: str, text: str):
    if not LINE_TOKEN:
        print("WARN: no LINE token")
        return
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {"Authorization": f"Bearer {LINE_TOKEN}", "Content-Type": "application/json"}
    body = {"replyToken": reply_token, "messages": [{"type": "text", "text": text}]}
    async with httpx.AsyncClient(timeout=20) as cli:
        r = await cli.post(url, headers=headers, json=body)
    print("LINE reply status:", r.status_code, r.text)

async def line_push(to_user_id: str, texts: List[str]):
    if not LINE_TOKEN:
        return
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Authorization": f"Bearer {LINE_TOKEN}", "Content-Type": "application/json"}
    msgs = [{"type": "text", "text": t} for t in texts]
    body = {"to": to_user_id, "messages": msgs[:5]}  # LINEは一度に最大5件
    async with httpx.AsyncClient(timeout=20) as cli:
        r = await cli.post(url, headers=headers, json=body)
    print("LINE push status:", r.status_code, r.text)

async def fetch_line_image_content(message_id: str) -> bytes:
    """LINEの画像バイナリを取得"""
    url = f"https://api.line.me/v2/bot/message/{message_id}/content"
    headers = {"Authorization": f"Bearer {LINE_TOKEN}"}
    async with httpx.AsyncClient(timeout=60) as cli:
        r = await cli.get(url, headers=headers)
        r.raise_for_status()
        return r.content

# ---------- 数式パーサ & 計算 ----------
def parse_quadratic_from_text(t: str) -> Dict[str, float] | None:
    """
    入力： '二次 1,-3,2' / '二次1,-3,2' / '二次1 -3 2' など
    どれでもOKにする（数字3つを順番に抽出）
    """
    if "二次" not in t:
        return None
    nums = re.findall(r"[-+]?\d*\.?\d+", t.replace("，", ","))
    # '二次'の直後にある数を優先（保険で末尾からも拾う）
    vals = []
    for s in nums:
        try:
            vals.append(float(s))
        except:
            pass
    if len(vals) >= 3:
        a, b, c = vals[:3]
        return {"a": a, "b": b, "c": c}
    return None

def solve_quadratic(a: float, b: float, c: float) -> Dict[str, Any]:
    D = b*b - 4*a*c
    if D > 0:
        rootD = math.sqrt(D)
        x1 = (-b + rootD) / (2*a)
        x2 = (-b - rootD) / (2*a)
        kind = "実数解（異なる2解）"
        xs = [x1, x2]
    elif abs(D) < 1e-12:
        x = -b / (2*a)
        kind = "重解（重根）"
        xs = [x, x]
    else:
        rootD = math.sqrt(-D)
        re_part = -b / (2*a)
        im_part = rootD / (2*a)
        kind = "虚数解（複素数）"
        xs = [complex(re_part, im_part), complex(re_part, -im_part)]
    return {"D": D, "kind": kind, "roots": xs}

def solve_linear(a: float, b: float) -> float:
    return -b / a

# ---------- OpenAI（画像→数式抽出：最大2問をJSONで） ----------
async def ocr_math_from_image_b64(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    画像（A4の紙・教科書写真想定）から『最大2問』の方程式を抽出。
    現状サポート：二次方程式 ax^2+bx+c=0 / 一次方程式 ax+b=0
    返り値：[{kind:'quadratic'|'linear', 'equation':'...', 'a':..., 'b':..., 'c':...}, ...]
    """
    if not OPENAI_API_KEY:
        return []

    b64 = base64.b64encode(image_bytes).decode("ascii")
    image_data_url = f"data:image/jpeg;base64,{b64}"

    # Chat CompletionsでJSON限定回答を要求（堅牢化）
    system = (
        "あなたは数式OCRのアシスタントです。写真はA4紙・教科書が想定です。\n"
        "最大2問まで抽出し、一次: ax+b=0 / 二次: ax^2+bx+c=0 の係数a,b,(c)を数値で返してください。\n"
        "小数・分数は小数に直して。文字式・説明文は無視。回答は次のJSONのみ：\n"
        "{ \"items\": [ {\"kind\":\"quadratic\",\"equation\":\"a x^2 + b x + c = 0\",\"a\":1.0,\"b\":-3.0,\"c\":2.0},\n"
        "              {\"kind\":\"linear\",\"equation\":\"a x + b = 0\",\"a\":2.0,\"b\":-6.0} ] }\n"
        "必ず items を返し、0～2件。JSON以外の文字は一切出力しない。"
    )
    user_text = "この画像から方程式を最大2問、上記JSONだけで返してください。"

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_data_url, "detail": "high"}}
            ]},
        ],
        "temperature": 0.1,
    }

    async with httpx.AsyncClient(timeout=60) as cli:
        r = await cli.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json=payload,
        )
    if r.status_code != 200:
        print("OpenAI error:", r.status_code, r.text[:200])
        return []

    data = r.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except Exception:
        return []

    # JSON以外を混ぜないように上で強制しているが、一応ガード
    m = re.search(r"\{[\s\S]*\}\s*$", text.strip())
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
        items = obj.get("items", [])
        out = []
        for it in items[:2]:
            kind = (it.get("kind") or "").lower()
            if kind not in ("quadratic", "linear"):
                continue
            if kind == "quadratic":
                a = float(it.get("a"))
                b = float(it.get("b"))
                c = float(it.get("c"))
                out.append({"kind": "quadratic", "equation": it.get("equation", "ax^2+bx+c=0"),
                            "a": a, "b": b, "c": c})
            else:
                a = float(it.get("a"))
                b = float(it.get("b"))
                out.append({"kind": "linear", "equation": it.get("equation", "ax+b=0"),
                            "a": a, "b": b})
        return out
    except Exception as e:
        print("JSON parse fail:", e)
        return []

# ---------- 応答ビルダー ----------
def build_quadratic_message(a: float, b: float, c: float) -> List[str]:
    sol = solve_quadratic(a, b, c)
    eq = f"{fmt_num(a)}x² + {fmt_num(b)}x + {fmt_num(c)} = 0"
    D = sol["D"]
    kind = sol["kind"]
    roots = sol["roots"]
    if isinstance(roots[0], complex):
        x1 = f"{fmt_num(roots[0].real)} ± {fmt_num(abs(roots[0].imag))}i"
        xs = f"x = {x1}"
    else:
        xs = f"x₁ = {fmt_num(roots[0])},  x₂ = {fmt_num(roots[1])}"

    head = f"問題：{eq}\n判別式：D = {fmt_num(D)} → {kind}\n解：{xs}"
    st = steps_fx_cg50_quadratic(a, b, c)
    return chunks(head + "\n\n" + st)

def build_linear_message(a: float, b: float) -> List[str]:
    x = solve_linear(a, b)
    eq = f"{fmt_num(a)}x + {fmt_num(b)} = 0"
    head = f"問題：{eq}\n解：x = {fmt_num(x)}"
    st = steps_fx_cg50_linear(a, b)
    return chunks(head + "\n\n" + st)

# ---------- ルート ----------
@app.get("/")
def root():
    return {"ok": True}

@app.post("/webhook")
async def webhook(req: Request):
    """LINE Webhook（Verify対策：空/非JSONでも200）"""
    try:
        body = await req.json()
    except Exception:
        body = {}
    print("Webhook:", body)
    events = body.get("events", [])
    if not events:
        return {"ok": True}

    for ev in events:
        et = ev.get("type")
        src = ev.get("source", {})
        user_id = src.get("userId")
        reply_token = ev.get("replyToken")

        if et == "message":
            msg = ev.get("message", {})
            mtype = msg.get("type")

            # 画像：先に即レス→バックで解析→Pushで結果
            if mtype in ("image", "file"):
                if reply_token:
                    await line_reply(reply_token, "画像を受け取りました。解析中…（最大2問まで）")

                try:
                    img_bytes = await fetch_line_image_content(msg.get("id"))
                    items = await ocr_math_from_image_b64(img_bytes)
                    if not items:
                        await line_push(user_id, ["すみません、式を読み取れませんでした（一次/二次の方程式のみ対応）。写真をもう少し近め/明るめ/正対で撮ってみてください。"])
                    else:
                        out_msgs: List[str] = []
                        for i, it in enumerate(items, 1):
                            if it["kind"] == "quadratic":
                                out_msgs += [f"— 問題{i} —"] + build_quadratic_message(it["a"], it["b"], it["c"])
                            elif it["kind"] == "linear":
                                out_msgs += [f"— 問題{i} —"] + build_linear_message(it["a"], it["b"])
                        await line_push(user_id, out_msgs[:5])  # 5通まで
                except Exception as e:
                    print("image flow error:", e)
                    await line_push(user_id, ["内部エラー：画像解析に失敗しました。時間をおいて再送してください。"])

            # テキスト：コマンド分岐
            elif mtype == "text":
                text = (msg.get("text") or "").strip()

                # 総合ガイド
                if "操作方法" in text or text.lower() in ("help", "使い方"):
                    if reply_token:
                        await line_reply(reply_token, HELP_KEYS)
                    continue

                # 二次（どの書き方でもOK）
                quad = parse_quadratic_from_text(text)
                if quad:
                    a, b, c = quad["a"], quad["b"], quad["c"]
                    if abs(a) < 1e-15:
                        if reply_token:
                            await line_reply(reply_token, "a=0 です。一次方程式として解いてください。")
                    else:
                        msgs = build_quadratic_message(a, b, c)
                        # 判別式の種別も含めて返す
                        for i, part in enumerate(msgs):
                            if reply_token and i == 0:
                                await line_reply(reply_token, part)
                            else:
                                await line_push(user_id, [part])
                    continue

                    # 将来：一次/連立なども追加可能

                # どれにも当たらない → 使い方
                if reply_token:
                    await line_reply(reply_token, quick_usage())

        # その他のイベントは無視
    return {"ok": True}
