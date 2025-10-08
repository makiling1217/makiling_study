# main.py  — LINE × OpenAI（画像/テキスト両対応） & fx-CG50 手順返信
# 2025-10-08 全貼り版

from __future__ import annotations

import os
import re
import math
import json
import base64
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel

# =========================
# 環境変数
# =========================
LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Render 健康確認用
app = FastAPI()


@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/healthz")
async def healthz():
    return {"ok": True}


# =========================
# 共通ユーティリティ
# =========================
def line_headers(json_type: bool = True) -> Dict[str, str]:
    h = {"Authorization": f"Bearer {LINE_TOKEN}"}
    if json_type:
        h["Content-Type"] = "application/json"
    return h


async def line_reply(reply_token: str, messages: List[Dict[str, Any]]) -> None:
    """即時返信（1秒以内確実に返す）"""
    if not LINE_TOKEN:
        print("ERROR: LINE_CHANNEL_ACCESS_TOKEN 未設定")
        return
    url = "https://api.line.me/v2/bot/message/reply"
    body = {"replyToken": reply_token, "messages": messages[:5]}
    async with httpx.AsyncClient(timeout=30) as ac:
        r = await ac.post(url, headers=line_headers(True), json=body)
        try:
            r.raise_for_status()
        except Exception:
            print("LINE reply error:", r.status_code, r.text)


async def line_push(user_id: str, messages: List[Dict[str, Any]]) -> None:
    """後追い送信（解析結果を分割で送る）"""
    if not LINE_TOKEN:
        print("ERROR: LINE_CHANNEL_ACCESS_TOKEN 未設定")
        return
    url = "https://api.line.me/v2/bot/message/push"
    body = {"to": user_id, "messages": messages[:5]}
    async with httpx.AsyncClient(timeout=30) as ac:
        r = await ac.post(url, headers=line_headers(True), json=body)
        try:
            r.raise_for_status()
        except Exception:
            print("LINE push error:", r.status_code, r.text)


def chunks_by_chars(text: str, limit: int = 900) -> List[str]:
    """長文を安全に分割（LINE は1メッセージ上限があるため）"""
    res, buf = [], []
    count = 0
    for line in text.splitlines(keepends=True):
        if count + len(line) > limit and buf:
            res.append("".join(buf))
            buf, count = [line], len(line)
        else:
            buf.append(line)
            count += len(line)
    if buf:
        res.append("".join(buf))
    return res


def to_line_texts(text: str) -> List[Dict[str, Any]]:
    return [{"type": "text", "text": t} for t in chunks_by_chars(text)]


# =========================
# fx-CG50 二次方程式 手順生成
# =========================
def steps_fx_cg50_quadratic(a: float, b: float, c: float) -> str:
    # できるだけ実機の表示/押下順に忠実に
    lines = []
    lines.append("【fx-CG50：二次方程式 ax²+bx+c=0 の解】")
    lines.append("1. 電源ON → 〔EQUATION〕を選択（アイコンメニュー）→ 〔EXE〕")
    lines.append("2. 〔Polynomial〕→ 〔EXE〕")
    lines.append("3. Degree（次数）で『2』→ 〔EXE〕")
    lines.append(f"4. a= に {a} を入力 → 〔EXE〕")
    lines.append(f"5. b= に {b} を入力 → 〔EXE〕")
    lines.append(f"6. c= に {c} を入力 → 〔EXE〕")
    lines.append("7. 解が表示される（x1, x2）。")
    lines.append("8. 〔EXE〕でページ送り／必要なら 〔EXIT〕で戻る。")
    lines.append("（補足）負号は『(−)』キー、少数は『.』、確定は毎回〔EXE〕。")
    # Fキーの補助メニューも併記（要望対応）
    lines.append("")
    lines.append("▼補助（Fキー）")
    lines.append("F1: Solve/OK（確定）")
    lines.append("F2: Back/Return（戻る）")
    lines.append("F3: Type切替/Polynomial選択")
    lines.append("F4: Degree変更")
    lines.append("F5: Coef.入力欄移動")
    lines.append("F6: 表示切替/More")
    return "\n".join(lines)


# =========================
# 二次方程式 文字パース & 解
# =========================
_NUM = r"[＋+−\-]?\d+(?:\.\d+)?"
def _num(s: str) -> float:
    s = s.replace("＋", "+").replace("−", "-")
    return float(s)

def parse_quadratic(text: str) -> Optional[Tuple[float, float, float]]:
    t = text.strip()
    # 正規化
    t = t.replace("，", ",").replace("、", ",").replace("　", " ")
    t = t.replace("a＝", "a=").replace("b＝", "b=").replace("c＝", "c=")

    # 例: 二次 1,-3,2 / 二次1,-3,2
    m = re.search(r"二次\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")", t)
    if m:
        return _num(m.group(1)), _num(m.group(2)), _num(m.group(3))

    # 例: 二次1 -3 2（スペース区切り）
    m = re.search(r"二次\s*(" + _NUM + r")\s+(" + _NUM + r")\s+(" + _NUM + r")", t)
    if m:
        return _num(m.group(1)), _num(m.group(2)), _num(m.group(3))

    # 例: 二次 1.-3.2（ピリオドで区切り）
    m = re.search(r"二次\s*(" + _NUM + r")\.\s*(" + _NUM + r")\.\s*(" + _NUM + r")", t)
    if m:
        return _num(m.group(1)), _num(m.group(2)), _num(m.group(3))

    # 例: a=1 b=-3 c=2
    m = re.search(r"a\s*=\s*(" + _NUM + r")[^\d\-+]*b\s*=\s*(" + _NUM + r")[^\d\-+]*c\s*=\s*(" + _NUM + r")", t, re.IGNORECASE)
    if m:
        return _num(m.group(1)), _num(m.group(2)), _num(m.group(3))

    return None


def solve_quadratic(a: float, b: float, c: float) -> Tuple[str, str]:
    D = b * b - 4 * a * c
    disc_comment = f"判別式 D = b² − 4ac = {b}² − 4×{a}×{c} = {D}"
    if D > 0:
        x1 = (-b + math.sqrt(D)) / (2 * a)
        x2 = (-b - math.sqrt(D)) / (2 * a)
        roots = f"実数解（異なる2解）: x₁ = {x1},  x₂ = {x2}"
    elif D == 0:
        x = (-b) / (2 * a)
        roots = f"重解: x = {x}"
    else:
        real = (-b) / (2 * a)
        imag = math.sqrt(-D) / (2 * a)
        roots = f"虚数解: x = {real} ± {imag} i"
    return disc_comment, roots


def build_quadratic_message(a: float, b: float, c: float) -> str:
    disc_comment, roots = solve_quadratic(a, b, c)
    eq = f"{a}x² + {b}x + {c} = 0"
    steps = steps_fx_cg50_quadratic(a, b, c)
    msg = []
    msg.append("【二次方程式（係数から計算）】")
    msg.append(f"式：{eq}")
    msg.append(disc_comment)
    msg.append(roots)
    msg.append("")
    msg.append(steps)
    return "\n".join(msg)


# =========================
# 画像 → 数式抽出（OpenAI 画像理解）
# =========================
async def ocr_extract_math(image_bytes: bytes) -> Optional[List[Dict[str, Any]]]:
    """
    画像から最大2問の『式』を抽出し、JSONで返す。
    返り値例:
        [{"type":"quadratic","equation":"x^2-3x+2=0","a":1,"b":-3,"c":2}, {...}]
    失敗時は None
    """
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY 未設定")
        return None

    b64 = base64.b64encode(image_bytes).decode("ascii")
    system = (
        "You are a math OCR assistant. Read the page (Japanese allowed). "
        "Return up to 2 problems found. Prefer explicit algebraic equations. "
        "If a quadratic ax^2+bx+c=0 appears, also return coefficients a,b,c as numbers. "
        "Respond ONLY as strict JSON array with objects having keys: "
        "type (e.g., 'quadratic'|'equation'|'text'), equation (string), "
        "and optional a,b,c (numbers). If nothing found return []."
    )
    user_prompt = (
        "Extract math problems. Return at most two. "
        "If multiple areas exist, choose the clearest ones."
    )
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{b64}",
                    },
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
    }
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=60) as ac:
            r = await ac.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            # response_format=json_object のため content は JSON 文字列
            obj = json.loads(content)
            # 期待形：{"data":[ ... ]} or 直接配列
            if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
                return obj["data"]
            if isinstance(obj, list):
                return obj
            # 一応配列でなくても頑張って拾う
            if isinstance(obj, dict) and "problems" in obj and isinstance(obj["problems"], list):
                return obj["problems"]
            return None
    except Exception as e:
        print("OpenAI Vision error:", e)
        return None


async def fetch_line_image_content(message_id: str) -> Optional[bytes]:
    """LINE 画像バイナリを取得"""
    if not LINE_TOKEN:
        return None
    url = f"https://api.line.me/v2/bot/message/{message_id}/content"
    try:
        async with httpx.AsyncClient(timeout=60) as ac:
            r = await ac.get(url, headers=line_headers(False))
            if r.status_code == 200:
                return r.content
            else:
                print("image fetch error:", r.status_code, r.text[:200])
                return None
    except Exception as e:
        print("image fetch exception:", e)
        return None


# =========================
# 「操作方法」総合ガイド
# =========================
GUIDE_OPS = """\
【fx-CG50 操作キーの総合ガイド】
1. 電源ON → アイコン〔EQUATION〕→ 〔EXE〕
2. 〔Polynomial〕→ 〔EXE〕 → Degree は問題に応じて 2/3 を選択 → 〔EXE〕
3. 係数 a,b,c,… を順に入力 → 各項目ごとに 〔EXE〕
4. 負の値は (−) キー、少数は .（小数点）
5. 解表示後、〔EXE〕でページ送り／〔EXIT〕で戻る

▼Fキー（画面下）
F1: OK / Solve    F2: Back/Return   F3: Type/Polynomial
F4: Degree        F5: 係数欄移動     F6: 表示切替/More

（ヒント）
・方程式モード：EQUATION → Polynomial
・連立は Simultaneous（同画面内）
・四則/関数入力は RUN-MAT（ホーム）で計算可能
"""


# =========================
# テキストメッセージ処理
# =========================
async def handle_text(user_id: str, reply_token: str, text: str) -> None:
    t = text.strip()

    # 「操作方法」
    if "操作方法" in t or t.lower() in {"help", "ヘルプ"}:
        await line_reply(reply_token, to_line_texts(GUIDE_OPS))
        return

    # 二次係数の解釈
    coefs = parse_quadratic(t)
    if coefs:
        a, b, c = coefs
        message = build_quadratic_message(a, b, c)
        await line_reply(reply_token, to_line_texts(message))
        return

    # デフォルトの案内
    usage = """\
使い方：
1) 問題の写真を送る → 先に『解析中…』→ その後『式＋答え＋番号付き手順』（最大2問）
2) 係数から解く：例「二次 1,-3,2」「二次1 -3 2」「二次 a=1 b=-3 c=2」
3) キー操作の一覧：「操作方法」
"""
    await line_reply(reply_token, to_line_texts(usage))


# =========================
# 画像メッセージ処理
# =========================
async def handle_image(user_id: str, reply_token: str, message_id: str) -> None:
    # まず即時返信
    await line_reply(reply_token, [{"type": "text", "text": "解析中… 少し待ってね。"}])

    # 画像取得
    image_bytes = await fetch_line_image_content(message_id)
    if not image_bytes:
        await line_push(
            user_id,
            to_line_texts("画像の取得に失敗しました。もう一度送ってください。")
        )
        return

    # OCR & 数式抽出
    probs = await ocr_extract_math(image_bytes)
    if not probs:
        msg = ("式を特定できませんでした。A4でもOKです。"
               "斜め/影/反射を避け、文字が濃いモードで再撮影をお願いします。\n"
               "※急ぎなら「二次 1,-3,2」の形式でも送れます。")
        await line_push(user_id, to_line_texts(msg))
        return

    # 最大2問を整形して送信
    msgs: List[str] = []
    for i, p in enumerate(probs[:2], start=1):
        typ = str(p.get("type", "equation"))
        eq = str(p.get("equation", "")).replace("**", "^")
        msgs.append(f"【問題{i}】\n種類: {typ}\n式: {eq}")

        if typ == "quadratic" or all(k in p for k in ("a", "b", "c")):
            try:
                a = float(p.get("a"))
                b = float(p.get("b"))
                c = float(p.get("c"))
                msgs.append(build_quadratic_message(a, b, c))
            except Exception:
                # 係数が取れなければ式だけ案内
                msgs.append("（係数が読み取れなかったため、式のみ表示）")
        else:
            msgs.append("（この種類の自動手順は準備中。式はRUN-MATで直接計算できます）")

    all_text = "\n\n".join(msgs)
    await line_push(user_id, to_line_texts(all_text))


# =========================
# LINE Webhook
# =========================
class LineEvent(BaseModel):
    type: str
    replyToken: Optional[str] = None
    message: Optional[Dict[str, Any]] = None
    source: Dict[str, Any]
    timestamp: Optional[int] = None
    webhookEventId: Optional[str] = None
    deliveryContext: Optional[Dict[str, Any]] = None
    mode: Optional[str] = None


class LineWebhook(BaseModel):
    destination: Optional[str] = None
    events: List[LineEvent] = []


@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        body = await request.json()
    except Exception as e:
        print("WEBHOOK parse error:", e)
        return {"ok": True}

    print("WEBHOOK:", body)
    data = LineWebhook(**body)

    for ev in data.events:
        user_id = ev.source.get("userId") if ev.source else None
        if ev.type == "message" and ev.message:
            mtype = ev.message.get("type")
            if mtype == "text":
                text = ev.message.get("text", "")
                # テキストはそのまま処理（即返信でOK）
                await handle_text(user_id or "", ev.replyToken or "", text)
            elif mtype == "image":
                message_id = ev.message.get("id")
                # 画像は「解析中…」を即時返信し、解析はバックグラウンドで push
                if ev.replyToken:
                    background_tasks.add_task(handle_image, user_id or "", ev.replyToken, message_id)
            else:
                if ev.replyToken:
                    await line_reply(ev.replyToken, [{"type": "text", "text": "対応していないメッセージ種別です。"}])
        else:
            if ev.replyToken:
                await line_reply(ev.replyToken, [{"type": "text", "text": "イベントを認識できませんでした。"}])

    return {"ok": True}
