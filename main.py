# main.py — LINE 画像強化版：前処理あり + 2問まで式/解/手順を返信
# - 画像取得: api-data.line.me（404回避）
# - 画像前処理: EXIF回転, グレー化, 自動コントラスト, 2.2倍拡大, くっきり化, 自動トリミング, 上下分割
# - OpenAI Vision: 前処理画像+上下分割を同時投入し「①/② 形式」で式→解→fx-CG50手順
# - テキスト: 「操作方法」「使い方」「二次 ...」各種表記を受理
# - 例外時も必ず応答して無反応を防止

import os, io, re, math, base64, asyncio
from typing import Dict, List, Any, Optional

import httpx
from fastapi import FastAPI, Request

from PIL import Image, ImageOps, ImageEnhance

LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

app = FastAPI()

@app.get("/") async def root(): return {"status": "ok"}
@app.get("/healthz") async def health(): return {"ok": True}

# ---------------- LINE I/O ----------------
def line_headers(json_type=True)->Dict[str,str]:
    h={"Authorization":f"Bearer {LINE_TOKEN}"}
    if json_type: h["Content-Type"]="application/json"
    return h

async def line_reply(reply_token:str, text:str):
    url="https://api.line.me/v2/bot/message/reply"
    body={"replyToken":reply_token, "messages":[{"type":"text","text":text[:4900]}]}
    async with httpx.AsyncClient(timeout=30) as ac:
        r=await ac.post(url, headers=line_headers(True), json=body); r.raise_for_status()

async def line_push(user_id:str, texts:List[str]):
    url="https://api.line.me/v2/bot/message/push"
    body={"to":user_id,"messages":[{"type":"text","text":t[:4900]} for t in texts[:5]]}
    async with httpx.AsyncClient(timeout=30) as ac:
        r=await ac.post(url, headers=line_headers(True), json=body); r.raise_for_status()

async def fetch_line_image(message_id:str)->Optional[bytes]:
    url=f"https://api-data.line.me/v2/bot/message/{message_id}/content"
    try:
        async with httpx.AsyncClient(timeout=20) as ac:
            r=await ac.get(url, headers={"Authorization":f"Bearer {LINE_TOKEN}"}); r.raise_for_status()
            return r.content
    except Exception as e:
        print("image fetch error:",e)
        return None

# ---------------- 画像前処理 ----------------
def _upscale(img:Image.Image, max_side:int=2400)->Image.Image:
    scale=max_side/max(img.size)
    if scale>1.0:
        new=(int(img.width*scale), int(img.height*scale))
        img=img.resize(new, Image.LANCZOS)
    return img

def preprocess(img_bytes:bytes)->Image.Image:
    img=Image.open(io.BytesIO(img_bytes))
    img=ImageOps.exif_transpose(img)       # 回転補正
    img=img.convert("L")                    # グレー
    img=ImageOps.autocontrast(img)         # 自動コントラスト
    img=_upscale(img, 2400)                # 拡大（細字対策）
    img=ImageEnhance.Sharpness(img).enhance(1.2)
    img=ImageEnhance.Contrast(img).enhance(1.4)

    # 粗二値化で非白領域の bbox を取って余白を落とす
    bw=img.point(lambda p: 255 if p>215 else 0)
    bbox=bw.getbbox()
    if bbox:
        l,t,r,b=bbox
        pad=30
        l=max(0,l-pad); t=max(0,t-pad); r=min(img.width,r+pad); b=min(img.height,b+pad)
        img=img.crop((l,t,r,b))
    return img

def to_data_url(img:Image.Image)->str:
    buf=io.BytesIO(); img.save(buf, format="PNG"); b64=base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def split_top_bottom(img:Image.Image)->List[Image.Image]:
    if img.height<1000: return [img]
    mid=img.height//2
    top=img.crop((0,0,img.width,mid))
    bot=img.crop((0,mid,img.width,img.height))
    return [img, top, bot]   # 全体 + 上 + 下 を投げる

# ---------------- OpenAI Vision ----------------
async def openai_vision_multi(img_bytes:bytes)->str:
    if not OPENAI_API_KEY:
        return "サーバの OPENAI_API_KEY が未設定のため、画像解析を実行できません。"

    pre=preprocess(img_bytes)
    imgs=split_top_bottom(pre)
    contents=[{"type":"text","text":
        ("次の画像群から最大2題の問題を抽出し、日本語で厳密に出力。\n"
         "各題は次の形式のみ（余計な説明禁止）で：\n"
         "①\n"
         "【式】...\n"
         "【解】...\n"
         "【電卓手順（fx-CG50）】番号付き 1. 2. 3. ... すべての [EXE] を明示。"
         "（方程式→Equation→Polynomial、確率・統計→STATやRUN-MAT など適切なアプリとキーを示す）\n"
         "②（2題ある場合のみ同様に）")}]
    for im in imgs:
        contents.append({"type":"image_url","image_url":{"url":to_data_url(im)}})

    payload={"model":"gpt-4o-mini",
             "messages":[{"role":"system","content":"Answer in Japanese. Output must follow the requested format strictly."},
                         {"role":"user","content":contents}],
             "temperature":0}

    try:
        async with httpx.AsyncClient(timeout=90) as ac:
            r=await ac.post("https://api.openai.com/v1/chat/completions",
                            headers={"Authorization":f"Bearer {OPENAI_API_KEY}",
                                     "Content-Type":"application/json"},
                            json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("OpenAI Vision error:",e)
        return ("式を特定できませんでした。写真は『画面いっぱい』『正面』『ピント』『影少なめ』で再送してください。"
                "（内部解析エラー）")

# ---------------- 二次方程式（テキスト系） ----------------
_num=r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
def parse_quad(text:str)->Optional[Dict[str,float]]:
    t=text.replace("　"," ").replace("，",",").strip()
    # よくある3形式
    m=re.search(r"二次\s*a\s*=\s*(%s)\s*b\s*=\s*(%s)\s*c\s*=\s*(%s)"%(_num,_num,_num),t)
    if m: return {"a":float(m.group(1)),"b":float(m.group(2)),"c":float(m.group(3))}
    m=re.search(r"二次\s*(%s)\s*,\s*(%s)\s*,\s*(%s)"%(_num,_num,_num),t)
    if m: return {"a":float(m.group(1)),"b":float(m.group(2)),"c":float(m.group(3))}
    m=re.search(r"二次\s*(%s)\s+(%s)\s+(%s)"%(_num,_num,_num),t)
    if m: return {"a":float(m.group(1)),"b":float(m.group(2)),"c":float(m.group(3))}
    # 変種: 「二次 1.-3.2」→ ドットを区切りとみなす
    if re.search(r"二次",t):
        nums=re.findall(r"[+-]?\d+", t)  # 整数優先で拾う
        if len(nums)>=3:
            a,b,c = map(float, nums[:3])
            return {"a":a,"b":b,"c":c}
    return None

def quad_solve(a:float,b:float,c:float)->Dict[str,Any]:
    D=b*b-4*a*c
    if a==0: return {"type":"linear","x":(-c/b) if b!=0 else None,"eq":f"{b}x+{c}=0"}
    if D>0:
        s=math.sqrt(D); return {"type":"real2","x1":(-b+s)/(2*a),"x2":(-b-s)/(2*a),"D":D}
    if D==0:
        return {"type":"double","x":(-b)/(2*a),"D":D}
    s=math.sqrt(-D); return {"type":"imag","re":(-b)/(2*a),"im":s/(2*a),"D":D}

def steps_fxcg50_quadratic(a,b,c)->str:
    return (
        "1. [MENU] → 『Equation(方程式)』 → [EXE]\n"
        "2. [F2] Polynomial → 次数に 2 を入力 → [EXE]\n"
        f"3. a に {a} を入力 → [EXE]\n"
        f"4. b に {b} を入力 → [EXE]\n"
        f"5. c に {c} を入力 → [EXE]\n"
        "6. 解が表示 → [F6] Solve / [EXE] で確認\n"
        "7. 変更は [EXIT]→係数修正→[EXE]\n"
        "（Fキー：F1 Simul / F2 Poly / F3 Solver / F4 Del / F5 = / F6 Solve）"
    )

def quad_answer(a,b,c)->str:
    res=quad_solve(a,b,c); eq=f"{a}x^2 + {b}x + {c} = 0"
    if res["type"]=="real2":
        head=f"【式】{eq}\n【判別式】D={res['D']}\n【解】x1={res['x1']}, x2={res['x2']}\n"
    elif res["type"]=="double":
        head=f"【式】{eq}\n【判別式】D=0\n【解】x={res['x']}（重解）\n"
    elif res["type"]=="imag":
        head=f"【式】{eq}\n【判別式】D={res['D']}<0\n【解】x={res['re']}±{abs(res['im'])}i\n"
    else:
        x=res["x"]; head=f"【式】{res['eq']}\n【解】x={x if x is not None else '不可'}\n"
    return head + "【電卓手順（fx-CG50）】\n" + steps_fxcg50_quadratic(a,b,c)

KEY_GUIDE=(
    "【fx-CG50 キー操作の総合ガイド】\n"
    "1.[MENU] 2.[EXIT] 3.[SHIFT]/[ALPHA] 4.[F1]〜[F6]（Equation: F1 Simul/F2 Poly/F3 Solver/F4 Del/F5 =/F6 Solve）\n"
    "5.数値・小数点『.』・負号『(-)』で入力→[EXE]  6.RUN-MAT/STAT/TABLE/GRAPH の使い分け\n"
    "7.角度設定:[SHIFT][MENU](SETUP)→Angle  8.消去:[DEL]/[AC/ON]  9.^,√,分数キー  10.エラー時は[EXIT]→見直し"
)
USAGE=(
    "使い方：\n"
    "1) 問題の写真を送る → 解析中… の後、最大2題について『式/解/番号付き手順』を返信\n"
    "2) 二次：『二次 a=1 b=-3 c=2』『二次 1,-3,2』『二次 1 -3 2』『二次 1.-3.2』も可\n"
    "3) キー操作の一覧：『操作方法』 と送信"
)

# ---------------- Webhook ----------------
@app.post("/webhook")
async def webhook(req:Request):
    try: data=await req.json()
    except Exception as e:
        print("json error:",e); return {"ok":True}
    for ev in data.get("events",[]):
        try: await handle_event(ev)
        except Exception as e: print("handler error:",e)
    return {"ok":True}

async def handle_event(ev:Dict[str,Any]):
    if ev.get("type")!="message": return
    m=ev.get("message",{}); mtype=m.get("type"); reply=ev.get("replyToken"); uid=(ev.get("source") or {}).get("userId","")

    if mtype=="image":
        # 即返信→後でPUSH（ハング防止）
        try: await line_reply(reply,"解析中… 少し待ってね。")
        except Exception as e: print("pre-reply error:",e)

        img=await fetch_line_image(m.get("id",""))
        if not img and m.get("contentProvider",{}).get("type")=="external":
            url=m.get("contentProvider",{}).get("originalContentUrl")
            if url:
                try:
                    async with httpx.AsyncClient(timeout=20) as ac:
                        r=await ac.get(url); r.raise_for_status(); img=r.content
                except Exception as e: print("external fetch error:",e)

        if not img:
            await line_push(uid,["画像の取得に失敗しました。受信後1分以内に再送してください。",
                                 "撮影ヒント：画面いっぱい・正面・ピント・影少なめ・コントラスト強め"])
            return

        async def run():
            ans=await openai_vision_multi(img)
            await line_push(uid,[ans])
        asyncio.create_task(run())
        return

    if mtype=="text":
        t=(m.get("text") or "").strip()
        if t in ["操作方法","キー操作","help","ヘルプ"]: await line_reply(reply,KEY_GUIDE); return
        if t in ["使い方","usage","つかいかた"]: await line_reply(reply,USAGE); return

        cf=parse_quad(t)
        if cf:
            try: await line_reply(reply, quad_answer(cf["a"],cf["b"],cf["c"]))
            except Exception as e:
                print("quad error:",e); await line_reply(reply,"係数の解釈に失敗。例：『二次 1,-3,2』")
            return

        await line_reply(reply,USAGE); return

    if reply: await line_reply(reply,"未対応のメッセージです。テキストまたは画像を送ってください。")
