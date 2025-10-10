import os
import hmac
import hashlib
import base64
import io
import json
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import cv2  # opencv-python-headless

import httpx
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

# ====== 環境変数 ======
LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ====== OpenAI client ======
# openai>=1.0.0
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()


# ---------- ユーティリティ ----------
def verify_line_signature(body: bytes, signature: str) -> None:
    if not LINE_SECRET:
        return
    mac = hmac.new(LINE_SECRET.encode("utf-8"), body, hashlib.sha256).digest()
    expect = base64.b64encode(mac).decode("utf-8")
    if not hmac.compare_digest(expect, signature):
        raise HTTPException(status_code=400, detail="Invalid signature")


def line_headers(json_type: bool = True) -> Dict[str, str]:
    h = {"Authorization": f"Bearer {LINE_TOKEN}"}
    if json_type:
        h["Content-Type"] = "application/json"
    return h


async def line_reply(reply_token: str, messages: List[Dict[str, Any]]) -> None:
    url = "https://api.line.me/v2/bot/message/reply"
    body = {"replyToken": reply_token, "messages": messages[:5]}
    async with httpx.AsyncClient(timeout=30) as ac:
        r = await ac.post(url, headers=line_headers(True), json=body)
        r.raise_for_status()


async def line_push(user_id: str, messages: List[Dict[str, Any]]) -> None:
    url = "https://api.line.me/v2/bot/message/push"
    body = {"to": user_id, "messages": messages[:5]}
    async with httpx.AsyncClient(timeout=30) as ac:
        r = await ac.post(url, headers=line_headers(True), json=body)
        r.raise_for_status()


def pil_enhance_for_ocr(pil: Image.Image) -> Image.Image:
    """A4想定：200%拡大＋コントラスト強化＋二値化（しきい値自動）"""
    # 200%拡大（Lanczos）
    w, h = pil.size
    pil = pil.resize((int(w * 2), int(h * 2)), Image.LANCZOS)

    # グレースケール
    gray = pil.convert("L")

    # コントラスト自動補正
    gray = ImageOps.autocontrast(gray)

    # 軽くシャープ
    gray = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    # OpenCVで自動二値化（大津）
    np_img = np.array(gray)
    _, th = cv2.threshold(np_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 文字の細り対策で少し膨張→収縮（開閉処理）
    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    return Image.fromarray(th)


def img_to_data_url(pil: Image.Image) -> str:
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------- 数学：二次式パーサ＆解法 ----------
def parse_quadratic(text: str) -> Optional[Tuple[float, float, float]]:
    """
    入力例：
    - '二次 1,-3,2'
    - '二次 1 -3 2'
    - '二次 a=1 b=-3 c=2'
    - '二次 a=1,b=-3,c=2'
    ※ 小数「1.5」はOK。区切りの「.」と紛らわしい '1.-3.2' は弾く。
    """
    s = text.strip().replace("　", " ")
    if not s.startswith("二次"):
        return None
    s2 = s[2:].strip()  # 「二次」を除いた残り

    # a=,b=,c= 形式
    if "a=" in s2 or "A=" in s2:
        # 記号を標準化
        s2 = s2.replace("，", ",").replace("、", ",")
        parts = s2.replace("A=", "a=").replace("B=", "b=").replace("C=", "c=")
        vals = {}
        for token in parts.split(","):
            token = token.strip()
            if "=" in token:
                k, v = token.split("=", 1)
                k = k.strip().lower()
                try:
                    vals[k] = float(v.strip())
                except Exception:
                    return None
        if set(vals.keys()) >= {"a", "b", "c"}:
            return float(vals["a"]), float(vals["b"]), float(vals["c"])
        return None

    # 単純な並び（カンマ/スペース/スラッシュ/日本語区切り）
    # ただし「ピリオド」は小数点専用に扱いたいので、
    # 区切りとしての . は無視（= '1.-3.2' は不正として None）
    tmp = (
        s2.replace("，", ",")
        .replace("、", ",")
        .replace("/", ",")
        .replace("／", ",")
        .replace(" ,", ",")
        .replace(", ", ",")
    )
    # ピリオド区切りはサポートしない（誤爆が多い）
    if "." in tmp and tmp.count(".") > 0 and any(seg.strip() == "" for seg in tmp.split(".")):
        # '1.-3.2' のような不正フォーマット
        pass  # 何もしないで次へ
    # カンマ or スペース
    tokens = []
    if "," in tmp:
        tokens = [t for t in tmp.split(",") if t.strip() != ""]
    else:
        tokens = [t for t in tmp.split(" ") if t.strip() != ""]

    nums: List[float] = []
    for t in tokens:
        try:
            nums.append(float(t))
        except Exception:
            pass
    if len(nums) == 3:
        return tuple(nums)  # type: ignore
    return None


def solve_quadratic(a: float, b: float, c: float) -> Dict[str, Any]:
    if a == 0:
        if b == 0:
            return {"type": "degenerate", "message": "a=b=0 → 恒等式または矛盾"}
        return {"type": "linear", "x": -c / b}
    D = b * b - 4 * a * c
    if D > 0:
        rD = D ** 0.5
        x1 = (-b + rD) / (2 * a)
        x2 = (-b - rD) / (2 * a)
        return {"type": "real2", "x1": x1, "x2": x2, "D": D}
    elif D == 0:
        x = -b / (2 * a)
        return {"type": "double", "x": x, "D": D}
    else:
        rD = (-D) ** 0.5
        # 複素数表示
        real = -b / (2 * a)
        imag = rD / (2 * a)
        return {"type": "complex", "real": real, "imag": imag, "D": D}


def fxcg50_steps_for_quadratic(a: float, b: float, c: float) -> str:
    """
    fx-CG50 の “EQUA（多項式）” で解く手順と、
    GRAPH で解を読みに行く手順を両方出す（キー名は角カッコ、必ず EXE を明示）
    """
    # EQUA
    equa = (
        "【EQUA（多項式）で解く】\n"
        "1) [MENU] → EQUA → [EXE]\n"
        "2) [F2](Polynomial) → [EXE]\n"
        "3) degree は 2 を選択 → [EXE]\n"
        f"4) 係数を上から順に入力 → [EXE]： a={a}, b={b}, c={c}\n"
        "5) [F6](SOLVE) または [EXE] で解を表示\n"
        "   ※ 上段に x1, 下段に x2。"
    )

    # GRAPH
    graph = (
        "【GRAPH で解（x切片）を読む】\n"
        "1) [MENU] → GRAPH → [EXE]\n"
        "2) Y1 行へカーソル → [F1](SELECT)（左端「＝」を濃く）\n"
        "3) [DEL] でY1を空にして、次を入力 → [EXE]\n"
        "   y = a×x^2 + b×x + c\n"
        "   入力列： [ALPHA][log](A) = a, [ALPHA][10^x](B) = b を使わず、\n"
        "   今回は そのまま数値で  a, b, c を直接入れて OK\n"
        "   （x と二乗は [X,θ,T], [x²]）\n"
        "4) [F6](DRAW)\n"
        "5) [SHIFT][F5](G-Solv) → [F1](ROOT)\n"
        "   左右キーで根の位置へ → [EXE] で x を読む。"
    )
    return equa + "\n\n" + graph


def build_fx_cg50_general_preamble() -> str:
    return (
        "《fx-CG50 固有の注意》\n"
        "・代入（STO▶）は **白い [→] キー** を単押し。SHIFTは不要。\n"
        "・A/B/C…は [ALPHA] のあと、ピンクの印字キー： A=[log], B=[10^x], C=[ln]\n"
        "・xは [X,θ,T]、二乗は [x²]、必ず式末で [EXE]。"
    )


# ---------- Visionで「A4最大2問」を読んで解く ----------
VISION_SYSTEM_PROMPT = """\
あなたは日本語の学習支援ボットです。A4用紙の写真（最大2問）から問題文を読み取り、
各問について (1) 正解 (2) 必要なら途中式 (3) fx-CG50のキー列（EXE含む／初心者向け）を日本語で短く出力します。
必ず各問を『Q1』『Q2』で区切り、最後に全体の一行要約を付けてください。
"""

def vision_solve_from_images(images: List[Image.Image]) -> str:
    contents = [{"type": "text", "text": VISION_SYSTEM_PROMPT}]
    # 元画像＋強調画像の両方を投げる
    for im in images:
        enhanced = pil_enhance_for_ocr(im)
        contents.append({"type": "image_url", "image_url": {"url": img_to_data_url(im)}})
        contents.append({"type": "image_url", "image_url": {"url": img_to_data_url(enhanced)}})

    # モデルは “GPT-4o mini / o3-mini” のビジョン対応を想定
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": contents}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# ---------- LINE Webhook ----------
@app.get("/")
async def root():
    return PlainTextResponse("ok")

@app.post("/webhook")
async def webhook(request: Request, x_line_signature: str = Header(None)):
    body = await request.body()
    try:
        verify_line_signature(body, x_line_signature or "")
    except HTTPException:
        # サイン失敗でもデバッグしやすいよう 200 を返す（本番は 400 推奨）
        return JSONResponse({"status": "ng-signature"}, status_code=200)

    data = await request.json()
    events = data.get("events", [])
    for ev in events:
        typ = ev.get("type")
        if typ != "message":
            continue

        msg = ev.get("message", {})
        user_id = ev.get("source", {}).get("userId")
        reply_token = ev.get("replyToken")

        mtype = msg.get("type")
        # まずは即時返信（1秒以内）
        if reply_token:
            try:
                await line_reply(reply_token, [{"type": "text", "text": "解析中… 少し待ってね。"}])
            except Exception:
                pass

        # 画像
        if mtype == "image":
            try:
                mid = msg.get("id")
                content_url = f"https://api-data.line.me/v2/bot/message/{mid}/content"
                async with httpx.AsyncClient(timeout=30) as ac:
                    r = await ac.get(content_url, headers=line_headers(False))
                    r.raise_for_status()
                    raw = r.content
                pil = Image.open(io.BytesIO(raw)).convert("RGB")

                answer = vision_solve_from_images([pil])

                if user_id:
                    await line_push(user_id, [{"type": "text", "text": answer}])
            except Exception as e:
                if user_id:
                    await line_push(user_id, [{
                        "type": "text",
                        "text": "画像の取得/解析に失敗しました。A4を正面・全体・ピントで撮影し直して送ってください。"
                    }])

        # テキスト（例：二次 1,-3,2）
        elif mtype == "text":
            text = msg.get("text", "").strip()
            a_b_c = parse_quadratic(text)
            if a_b_c is not None:
                a, b, c = a_b_c
                sol = solve_quadratic(a, b, c)

                pre = build_fx_cg50_general_preamble()
                steps = fxcg50_steps_for_quadratic(a, b, c)

                if sol["type"] == "real2":
                    ans = f"[答] x = {sol['x1']} , {sol['x2']}（判別式D={sol['D']}）"
                elif sol["type"] == "double":
                    ans = f"[答] 重解 x = {sol['x']}（D=0）"
                elif sol["type"] == "linear":
                    ans = f"[答] 一次方程式：x = {sol['x']}"
                elif sol["type"] == "complex":
                    ans = f"[答] 複素数解 x = {sol['real']} ± {abs(sol['imag'])}i（D={sol['D']}）"
                else:
                    ans = "[答] a=b=0 などの特別な場合で解が一意に決まりません。"

                out = (
                    f"二次方程式 ax^2+bx+c=0\n"
                    f"a={a}, b={b}, c={c}\n"
                    f"{ans}\n\n"
                    f"{pre}\n\n"
                    f"{steps}"
                )
                if user_id:
                    await line_push(user_id, [{"type": "text", "text": out}])
            else:
                # ヘルプ
                help_txt = (
                    "テキスト解法の例：\n"
                    "・二次 1,-3,2\n"
                    "・二次 1 -3 2\n"
                    "・二次 a=1 b=-3 c=2\n"
                    "（ピリオド区切り『1.-3.2』は不可｜小数はOK）\n\n"
                    "画像はA4を正面から撮影で送ってください。"
                )
                if user_id:
                    await line_push(user_id, [{"type": "text", "text": help_txt}])

    return JSONResponse({"status": "ok"})
