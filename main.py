# main.py  — Render/LINE(画像/テキスト)対応 & fx-CG50手順返信
# - 画像取得: api-data.line.me を使用（404対策）
# - 画像は即時に「解析中…」を返信 → バックグラウンドで解析し PUSH
# - テキスト:
#     * 「操作方法」→ キー操作の総合ガイド(番号つき、F1〜F6含む)
#     * 「二次 ...」→ 係数を robust に解釈（カンマ/空白/全角/ a= b= c= 形式）
#       解と fx-CG50 の番号つき手順を返信
# - どのケースでも例外時は必ず応答してハングを防止

import os
import re
import math
import base64
import asyncio
from typing import Dict, List, Any, Optional

import httpx
from fastapi import FastAPI, Request

# ---------------------- 環境変数 ----------------------
LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ---------------------- FastAPI ----------------------
app = FastAPI()


@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/healthz")
async def health():
    return {"ok": True}


# ---------------------- LINE 共通 ----------------------
def line_headers(json_type: bool = True) -> Dict[str, str]:
    h = {"Authorization": f"Bearer {LINE_TOKEN}"}
    if json_type:
        h["Content-Type"] = "application/json"
    return h


async def line_reply(reply_token: str, text: str):
    """返信（Reply）"""
    url = "https://api.line.me/v2/bot/message/reply"
    body = {"replyToken": reply_token, "messages": [{"type": "text", "text": text[:4900]}]}
    async with httpx.AsyncClient(timeout=30) as ac:
        r = await ac.post(url, headers=line_headers(True), json=body)
        r.raise_for_status()


async def line_push(user_id: str, texts: List[str]):
    """PUSH メッセージ（複数連投対応）"""
    url = "https://api.line.me/v2/bot/message/push"
    messages = [{"type": "text", "text": t[:4900]} for t in texts[:5]]
    body = {"to": user_id, "messages": messages}
    async with httpx.AsyncClient(timeout=30) as ac:
        r = await ac.post(url, headers=line_headers(True), json=body)
        r.raise_for_status()


# ---------------------- 画像取得（重要：api-data） ----------------------
async def fetch_line_image(message_id: str) -> Optional[bytes]:
    """
    画像のバイナリ取得。
    - 取得先は api-data.line.me
    - 1分で失効するので、受信後すぐに GET する
    """
    url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
    headers = {"Authorization": f"Bearer {LINE_TOKEN}"}
    try:
        async with httpx.AsyncClient(timeout=20) as ac:
            r = await ac.get(url, headers=headers)
            r.raise_for_status()
            return r.content
    except httpx.HTTPStatusError as e:
        print("image fetch error:", e.response.status_code, e.response.text)
    except Exception as e:
        print("image fetch error:", e)
    return None


# ---------------------- OpenAI (画像→数式抽出) ----------------------
async def openai_vision_solve(img_bytes: bytes) -> str:
    """
    OpenAI Vision で A4 の問題を最大2題まで解析し、
    『式 / 解 / fx-CG50 番号つき手順』のテキストを返す。
    失敗時は理由と再撮影ヒントを返す。
    """
    if not OPENAI_API_KEY:
        return "サーバの OPENAI_API_KEY が未設定のため、画像解析を実行できません。"

    # data URI で画像を渡す（Pillow等は未使用）
    b64 = base64.b64encode(img_bytes).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"

    prompt = (
        "You are a tutor specialized in reading math problems from A4 sheet photos and "
        "creating step-by-step CASIO fx-CG50 key instructions.\n"
        "Read the photo. If multiple (up to 2) problems exist, treat as ① and ②.\n"
        "For each problem, output in Japanese, strictly as:\n"
        "【式】<compact equation>\n"
        "【解】<final numeric solution or closed form>\n"
        "【電卓手順（fx-CG50）】番号付き 1. 2. 3. ... で、MENU→EQUA→Polynomial など実キー名を明示。\n"
        "Always include [EXE] where the user must press it.\n"
        "If problem is probability / stats etc., still give the best fx-CG50 path (e.g., STAT, RUN-MAT) "
        "and key steps; if ambiguous, say what to enter.\n"
        "If the text is too faint to be certain, say so and still attempt a best guess, then add "
        "『不確実: ～』 with your confidence reason.\n"
        "Be concise; no markdown, pure text."
    )

    # Chat Completions (gpt-4o-mini) で画像入力
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Answer in Japanese."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": 0.2,
    }

    try:
        async with httpx.AsyncClient(timeout=60) as ac:
            r = await ac.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"].strip()
            return text
    except Exception as e:
        print("OpenAI Vision error:", e)
        return (
            "式を特定できませんでした。写真は『画面いっぱい』『正面』『ピント』『影少なめ』で再送してください。"
            "（内部解析エラー）"
        )


# ---------------------- 二次方程式 解析/計算 ----------------------
_num = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"  # 1, -3, 2.5, .5 など

def parse_quadratic(text: str) -> Optional[Dict[str, float]]:
    t = text.replace("　", " ").replace("，", ",").strip()

    # パターン1: 「二次 a=1 b=-3 c=2」
    m = re.search(r"二次\s*a\s*=\s*(%s)\s*b\s*=\s*(%s)\s*c\s*=\s*(%s)" % (_num, _num, _num), t)
    if m:
        a, b, c = map(float, m.groups())
        return {"a": a, "b": b, "c": c}

    # パターン2: 「二次 1,-3,2」/ 全角カンマにも対応
    m = re.search(r"二次\s*(%s)\s*,\s*(%s)\s*,\s*(%s)" % (_num, _num, _num), t)
    if m:
        a, b, c = map(float, m.groups())
        return {"a": a, "b": b, "c": c}

    # パターン3: 「二次 1 -3 2」
    m = re.search(r"二次\s*(%s)\s+(%s)\s+(%s)" % (_num, _num, _num), t)
    if m:
        a, b, c = map(float, m.groups())
        return {"a": a, "b": b, "c": c}

    return None


def solve_quadratic(a: float, b: float, c: float) -> Dict[str, Any]:
    D = b * b - 4 * a * c
    if a == 0:
        # 退化: 一次
        if b == 0:
            return {"type": "invalid", "solution": "解なし（a=b=0）"}
        x = -c / b
        return {"type": "linear", "x": x, "equation": f"{b}x+{c}=0"}

    if D > 0:
        sqrtD = math.sqrt(D)
        x1 = (-b + sqrtD) / (2 * a)
        x2 = (-b - sqrtD) / (2 * a)
        t = "異なる実数解"
        return {"type": t, "x1": x1, "x2": x2, "D": D}
    elif D == 0:
        x = -b / (2 * a)
        t = "重解"
        return {"type": t, "x": x, "D": D}
    else:
        sqrtD = math.sqrt(-D)
        real = -b / (2 * a)
        imag = sqrtD / (2 * a)
        t = "虚数解"
        return {"type": t, "real": real, "imag": imag, "D": D}


def fxcg50_steps_quadratic(a: float, b: float, c: float) -> str:
    """
    fx-CG50 のキー操作（番号付き）。EXE を明示。F1〜F6の表記も含む。
    """
    return (
        "1. [MENU] → アイコン『Equation(方程式)』を選択 → [EXE]\n"
        "2. [F2] Polynomial（多項式）→ 次画面で次数 [2] を入力 → [EXE]\n"
        f"3. 係数 a に『{a}』と入力 → [EXE]\n"
        f"4. 係数 b に『{b}』と入力 → [EXE]\n"
        f"5. 係数 c に『{c}』と入力 → [EXE]\n"
        "6. 解一覧が表示 → [F6] Solve / [EXE] で次へ、解を確認\n"
        "7. 別解/再入力: [EXIT] → 係数行へ戻り、数値修正 → [EXE]\n"
        "（補足）Equation 画面の主なキー: [F1] Simultaneous / [F2] Polynomial / [F3] Solver /\n"
        "        [F4] Del / [F5] = / [F6] Solve  ※表示は機種/OSで多少異なります"
    )


def format_quadratic_answer(a: float, b: float, c: float) -> str:
    res = solve_quadratic(a, b, c)
    eq = f"{a}x^2 + {b}x + {c} = 0"
    if res["type"] == "異なる実数解":
        x1 = res["x1"]; x2 = res["x2"]
        head = f"【式】{eq}\n【判別式】D=b^2-4ac={res['D']}\n【解】x1={x1},  x2={x2}\n"
    elif res["type"] == "重解":
        x = res["x"]
        head = f"【式】{eq}\n【判別式】D=0\n【解】x={x}（重解）\n"
    elif res["type"] == "虚数解":
        head = (
            f"【式】{eq}\n【判別式】D={res['D']}<0\n"
            f"【解】x={res['real']}±{abs(res['imag'])}i\n"
        )
    elif res["type"] == "linear":
        head = f"【式】{res['equation']}\n【解】x={res['x']}\n"
    else:
        head = f"【式】{eq}\n【解】判定不可\n"

    steps = f"【電卓手順（fx-CG50）】\n{fxcg50_steps_quadratic(a, b, c)}"
    return head + steps


# ---------------------- キー操作 総合ガイド ----------------------
KEY_GUIDE = (
    "【fx-CG50 キー操作の総合ガイド】\n"
    "1. [MENU]：アプリ一覧へ。矢印キーで移動 → [EXE] で決定\n"
    "2. [EXIT]：1つ前の画面へ戻る / 入力確定前の取消\n"
    "3. [SHIFT]：黄色の二次機能、[ALPHA]：赤字入力\n"
    "4. 下段 [F1]〜[F6]：画面下に表示される機能キー\n"
    "   例（Equation）：[F1] Simul / [F2] Poly / [F3] Solver / [F4] Del / [F5] = / [F6] Solve\n"
    "5. 数字・小数点『.』・負号『(-)』で係数入力、[EXE] で確定\n"
    "6. RUN-MAT：通常計算、STAT：統計、TABLE：関数表、GRAPH：グラフ\n"
    "7. 角度設定：RUN-MATで [SHIFT][MENU](SETUP) → Angle：Deg/Rad → [EXE]\n"
    "8. クリア：入力中は [DEL]、全消去は [AC/ON]（注意）\n"
    "9. 分数/√/^：テンキー上のキーを使用。^ はべき乗。\n"
    "10. エラー時：メッセージ確認→[EXIT]→入力値/モードを見直し"
)

USAGE = (
    "使い方：\n"
    "1) 問題の写真を送る → 解析中… と出た後、最大2問まで『式/解/番号付き手順』を順に返信\n"
    "2) 二次方程式：『二次 a=1 b=-3 c=2』『二次 1,-3,2』『二次 1 -3 2』の形式に対応\n"
    "3) キー操作の一覧：『操作方法』 と送信\n"
)


# ---------------------- Webhook ----------------------
@app.post("/webhook")
async def webhook(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        print("json error:", e)
        return {"ok": True}

    events = body.get("events", [])
    for ev in events:
        try:
            await handle_event(ev)
        except Exception as e:
            print("handler error:", e)
    return {"ok": True}


async def handle_event(event: Dict[str, Any]):
    etype = event.get("type")
    if etype != "message":
        return

    msg = event.get("message", {})
    msg_type = msg.get("type")
    reply_token = event.get("replyToken")
    user_id = (event.get("source") or {}).get("userId", "")

    # -------- 画像 --------
    if msg_type == "image":
        # まず即時に「解析中…」を返信し、タイムアウト/無反応を防ぐ
        try:
            await line_reply(reply_token, "解析中… 少し待ってね。")
        except Exception as e:
            print("reply pre-error:", e)

        # 画像をすぐ取得（1分で失効）
        mid = msg.get("id", "")
        img_bytes = await fetch_line_image(mid)

        # 外部URL(Forwarded)対策
        if not img_bytes and msg.get("contentProvider", {}).get("type") == "external":
            orig = msg.get("contentProvider", {}).get("originalContentUrl")
            if orig:
                try:
                    async with httpx.AsyncClient(timeout=20) as ac:
                        r = await ac.get(orig)
                        r.raise_for_status()
                        img_bytes = r.content
                except Exception as e:
                    print("external fetch error:", e)

        if not img_bytes:
            await line_push(user_id, [
                "画像の取得に失敗しました。受信後1分以内にもう一度送ってください。",
                "撮影ヒント：画面いっぱい・正面・ピント・影少なめ・コントラストやや強め",
            ])
            return

        # 解析 → 結果を PUSH
        async def run_and_push():
            text = await openai_vision_solve(img_bytes)
            await line_push(user_id, [text])

        asyncio.create_task(run_and_push())
        return

    # -------- テキスト --------
    if msg_type == "text":
        text = (msg.get("text") or "").strip()

        # 操作方法
        if text in ["操作方法", "キー操作", "help", "ヘルプ"]:
            await line_reply(reply_token, KEY_GUIDE)
            return

        if text in ["使い方", "usage", "つかいかた"]:
            await line_reply(reply_token, USAGE)
            return

        # 二次方程式
        coef = parse_quadratic(text)
        if coef:
            a, b, c = coef["a"], coef["b"], coef["c"]
            try:
                answer = format_quadratic_answer(a, b, c)
                await line_reply(reply_token, answer)
                return
            except Exception as e:
                print("quad error:", e)
                await line_reply(reply_token, "係数の解釈に失敗しました。例：『二次 1,-3,2』")
                return

        # 既定（使い方案内）
        await line_reply(reply_token, USAGE)
        return

    # その他メッセージ種別
    if reply_token:
        await line_reply(reply_token, "未対応のメッセージです。テキストまたは画像を送ってください。")
