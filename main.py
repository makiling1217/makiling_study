# main.py
# LINE 画像→問題抽出→解答→fx-CG50の番号付きキー操作を返す
# テキスト指示:
#  - 「操作方法」…総合ガイド（F1〜F6/負号/EXEなど）
#  - 「二次 1,-3,2」「二次1 -3 2」「二次 a=1 b=-3 c=2」…解＋手順
# 環境変数: LINE_CHANNEL_ACCESS_TOKEN, OPENAI_API_KEY

import os
import io
import re
import json
import base64
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
from PIL import Image, ImageEnhance, ImageOps

# ---------- 環境 ----------
LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_VISION = os.getenv("OPENAI_MODEL_VISION", "gpt-4o-mini")  # vision対応
OPENAI_MODEL_TEXT = os.getenv("OPENAI_MODEL_TEXT", "gpt-4o-mini")

app = FastAPI()

# ---------- 共通 ----------
def line_headers(json_type: bool = True) -> Dict[str, str]:
    h = {"Authorization": f"Bearer {LINE_TOKEN}"}
    if json_type:
        h["Content-Type"] = "application/json"
    return h

async def line_reply(reply_token: str, messages: List[Dict[str, Any]]) -> None:
    url = "https://api.line.me/v2/bot/message/reply"
    body = {"replyToken": reply_token, "messages": messages[:5]}
    async with httpx.AsyncClient(timeout=30.0) as ac:
        r = await ac.post(url, headers=line_headers(True), json=body)
        r.raise_for_status()

async def line_push(user_id: str, messages: List[Dict[str, Any]]) -> None:
    url = "https://api.line.me/v2/bot/message/push"
    body = {"to": user_id, "messages": messages[:5]}
    async with httpx.AsyncClient(timeout=30.0) as ac:
        r = await ac.post(url, headers=line_headers(True), json=body)
        r.raise_for_status()

def ok(msg: str = "ok") -> Dict[str, Any]:
    return {"status": msg}

# ---------- fx-CG50 定型手順（共通） ----------
def cg50_quadratic_steps(a: float, b: float, c: float) -> List[str]:
    # 二次方程式の標準キー手順（EXE必須で細かく）
    steps = [
        "1. [MENU] → 画面の『EQUATION』を十字キーで選択 → [EXE]",
        "2. 『POLY』を選択 → [EXE]",
        "3. 次数『2』を選択 → [EXE]（二次方程式）",
        f"4. a= の枠で {a} と入力 → [EXE]",
        f"5. b= の枠で {b} と入力 → [EXE]",
        f"6. c= の枠で {c} と入力 → [EXE]",
        "7. 解（x1, x2）が表示される → 必要に応じて [▲]/[▼] で切替",
        "8. 終了は [EXIT] を数回"
    ]
    return steps

def cg50_key_guide() -> str:
    guide = [
        "【fx-CG50 キー操作・総合ガイド】",
        "1. 方向キー: 十字キーで項目を選ぶ → [EXE] で決定",
        "2. [SHIFT]: 黄印の機能 / [ALPHA]: 赤印の機能",
        "3. [(-)] は負号（マイナス）入力（例: -3 は [(-)]→3）",
        "4. [x10^] キーで×10のべき（科学表記）",
        "5. [DEL] で1文字削除 / [AC] で行クリア",
        "6. 小数点は [.] キー",
        "7. F1〜F6 は画面下のソフトキー（表示に合わせて働きが変わる）",
        "8. [EXIT] で一つ戻る、メニューに戻るまで数回押せばOK",
        "9. どの決定も基本は [EXE]。入力のたびに [EXE] を押すのを忘れない"
    ]
    return "\n".join(guide)

# ---------- 二次方程式 文字指示の解析 ----------
def parse_quadratic_text(s: str) -> Optional[Tuple[float, float, float]]:
    # 例: 「二次 1,-3,2」「二次1 -3 2」「二次 a=1 b=-3 c=2」「二次 1.-3.2」
    if "二次" not in s:
        return None
    # a,b,c の明示がある場合
    m = re.search(r"a\s*=\s*([+\-]?\d+(?:\.\d+)?)", s)
    if m:
        a = float(m.group(1))
        mb = re.search(r"b\s*=\s*([+\-]?\d+(?:\.\d+)?)", s)
        mc = re.search(r"c\s*=\s*([+\-]?\d+(?:\.\d+)?)", s)
        if mb and mc:
            return a, float(mb.group(1)), float(mc.group(1))
    # 数だけ拾う（カンマ/スペース/全角混在OK）
    nums = re.findall(r"[+\-]?\d+(?:\.\d+)?", s.replace("，", ",").replace("　", " "))
    if len(nums) >= 3:
        a, b, c = map(float, nums[:3])
        return a, b, c
    return None

def solve_quadratic(a: float, b: float, c: float) -> Dict[str, Any]:
    import math
    D = b*b - 4*a*c
    result: Dict[str, Any] = {"a": a, "b": b, "c": c, "D": D}
    if D > 0:
        r1 = (-b + math.sqrt(D)) / (2*a)
        r2 = (-b - math.sqrt(D)) / (2*a)
        kind = "実数解（異なる2解）"
        ans = f"x1={r1}, x2={r2}"
    elif D == 0:
        r = (-b) / (2*a)
        kind = "実数解（重解）"
        ans = f"x={r}"
    else:
        # 複素数形表示
        real = -b / (2*a)
        imag = math.sqrt(-D) / (2*a)
        kind = "複素数解"
        ans = f"x={real}±{imag}i"
    result.update({"kind": kind, "answer": ans})
    return result

# ---------- 画像前処理 ----------
def preprocess_image(img_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(img_bytes))
    # 200〜300%に拡大（短辺基準 1800px 目安）
    w, h = img.size
    scale = max(2.0, 1800 / min(w, h)) if min(w, h) < 1800 else 2.0
    new = img.convert("RGB").resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    # グレースケール + コントラスト強化
    new = ImageOps.grayscale(new)
    new = ImageEnhance.Contrast(new).enhance(1.8)
    buf = io.BytesIO()
    new.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

# ---------- Vision で問題抽出→解答→手順 ----------
async def ask_openai_for_math(image_png: bytes) -> Optional[List[Dict[str, Any]]]:
    """
    画像から最大2問の『式 / 答え / fx-CG50手順(番号付き・EXE含む)』を JSON 厳格形式で返す。
    失敗時は None。
    """
    if not OPENAI_API_KEY:
        return None

    b64 = base64.b64encode(image_png).decode("utf-8")
    system_prompt = (
        "You are a strict math solver and calculator-instructor for CASIO fx-CG50."
        " Read the photo (Japanese possible). Extract up to 2 distinct sub-problems."
        " For each, compute the final numeric answer (or simplest exact form)."
        " Then output numbered key steps for CASIO fx-CG50 so that a student can reproduce"
        " the answer. Every input confirmation must include [EXE]."
        " If a problem is not suitable for direct calculator operation, still give the"
        " correct answer and set steps to a short note like '手計算で求める' with 1–2 hints."
        " Output STRICT JSON ONLY with this shape:\n"
        "{ \"problems\":[ {"
        "  \"title\":\"...\", \"equation\":\"...\", \"answer\":\"...\","
        "  \"steps\":[\"1. ...\", \"2. ...\", ...] } ] }"
    )
    user_hint = (
        "日本語で。式はそのままの体裁で。手順は1. 2. 3. の配列。"
        " 例: 『MENU→EQUATION→POLY→次数2→a,b,c入力→EXE…』のようにEXE必須。"
    )

    payload = {
        "model": OPENAI_MODEL_VISION,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "input_text", "text": user_hint},
                {"type": "input_image", "image_data": b64, "mime_type": "image/png"}
            ]}
        ],
        "max_output_tokens": 1200,
        "temperature": 0.1
    }

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    for attempt in range(2):
        try:
            with httpx.Client(timeout=45.0) as c:
                r = c.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            text = data.get("output_text") or data.get("choices", [{}])[0].get("message", {}).get("content")
            if not text:
                # responses API は output[0].content[0].text のこともある
                out = data.get("output", [])
                if out and out[0].get("content"):
                    text = out[0]["content"][0].get("text", "")
            # JSONだけにする（前後のノイズ除去）
            m = re.search(r"\{[\s\S]*\}", text)
            if not m:
                continue
            obj = json.loads(m.group(0))
            probs = obj.get("problems", [])
            if probs:
                # stepsが1. 2. ...で始まるか最終チェック
                for p in probs:
                    p["steps"] = [s if re.match(r"^\d+\.", s) else f"{i+1}. {s}"
                                  for i, s in enumerate(p.get("steps", []))]
                return probs[:2]
        except Exception:
            continue
    return None

# ---------- LINE 画像処理 ----------
async def handle_image(message_id: str, user_id: str) -> str:
    # 1) 画像取得（即時）。過去メッセージは404になるので注意。
    content_url = f"https://api.line.me/v2/bot/message/{message_id}/content"
    data: Optional[bytes] = None
    async with httpx.AsyncClient(timeout=30.0) as ac:
        for _ in range(2):
            r = await ac.get(content_url, headers=line_headers(False))
            if r.status_code == 200:
                data = r.content
                break
    if not data:
        return "画像の取得に失敗しました。もう一度、正面から明るく撮って送ってください。"

    # 2) 前処理
    pre = preprocess_image(data)

    # 3) Visionで抽出→解答→手順
    probs = await ask_openai_for_math(pre)
    if not probs:
        return "式を特定できませんでした。画面いっぱい・正面・影少なめで再送してください。"

    # 4) 整形して返す
    out_lines: List[str] = []
    for idx, p in enumerate(probs, 1):
        title = p.get("title") or f"問題{idx}"
        eq = p.get("equation", "")
        ans = p.get("answer", "")
        steps = p.get("steps", []) or []
        out_lines.append(f"◆{title}")
        if eq:
            out_lines.append(f"式: {eq}")
        if ans:
            out_lines.append(f"答え: {ans}")
        if steps:
            out_lines.append("手順:")
            out_lines += [f"{s}" for s in steps]
        out_lines.append("")  # 区切り

    return "\n".join(out_lines).strip()

# ---------- FastAPI ----------
@app.get("/")
def root():
    return ok()

@app.post("/webhook")
async def webhook(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(ok("no json"))
    # print("WEBHOOK:", body)  # デバッグ

    events = body.get("events", [])
    for ev in events:
        try:
            etype = ev.get("type")
            reply_token = ev.get("replyToken")
            src = ev.get("source", {})
            user_id = src.get("userId", "")

            if etype == "message":
                msg = ev.get("message", {})
                mtype = msg.get("type")

                # ---- 画像 ----
                if mtype == "image":
                    # 先に「解析中…」返信 → 後からpush
                    if reply_token:
                        await line_reply(reply_token, [{"type": "text", "text": "解析中… 少し待ってね。"}])
                    mid = msg.get("id")
                    text = await handle_image(mid, user_id)
                    if text and user_id:
                        await line_push(user_id, [{"type": "text", "text": text}])
                    continue

                # ---- テキスト ----
                if mtype == "text":
                    t = msg.get("text", "").strip()

                    # 総合ガイド
                    if re.search(r"操作方法|ヘルプ|使い方", t):
                        await line_reply(reply_token, [{"type": "text", "text": cg50_key_guide()}])
                        continue

                    # 二次（各種表記）
                    abc = parse_quadratic_text(t)
                    if abc:
                        a, b, c = abc
                        info = solve_quadratic(a, b, c)
                        steps = cg50_quadratic_steps(a, b, c)
                        text = (
                            f"【二次方程式 ax^2+bx+c=0】\n"
                            f"a={a}, b={b}, c={c}\n"
                            f"判別式 D={info['D']} → {info['kind']}\n"
                            f"答え: {info['answer']}\n"
                            "手順:\n" + "\n".join(steps)
                        )
                        await line_reply(reply_token, [{"type": "text", "text": text}])
                        continue

                    # それ以外は案内
                    usage = (
                        "使い方：\n"
                        "1) 問題の写真を送る → 解析後『式＋答え＋番号付き手順』を返します（最大2問まで）\n"
                        "2) 二次方程式なら『二次 a=1 b=-3 c=2』や『二次 1,-3,2』『二次1 -3 2』でも解＋手順を返します\n"
                        "3) キー操作の一覧は『操作方法』と送信"
                    )
                    await line_reply(reply_token, [{"type": "text", "text": usage}])
                    continue

            # それ以外のイベントは無視
        except Exception as e:
            # 失敗しても 200 を返す（LINE再送を防ぐ）
            try:
                if ev.get("replyToken"):
                    await line_reply(ev["replyToken"], [{"type": "text", "text": f"内部エラーが発生しました。もう一度お試しください。"}])
            except Exception:
                pass
            # print("handler error:", e)

    return JSONResponse(ok())
