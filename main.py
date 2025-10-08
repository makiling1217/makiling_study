# main.py — 正答最優先版
# 方針:
#  1) 画像: OpenAIは「問題タイプと係数の抽出」のみ(JSON)。解くのはローカル実装。
#  2) テキスト: 二次(各種表記)を厳密パース→ローカルで解く→fx-CG50手順(必ず [EXE] を明示)を生成。
#  3) 画像前処理は継続(回転/コントラスト/拡大/トリミング/上下分割)。最大2問。

import os, io, re, math, json, base64, asyncio
from typing import Dict, List, Any, Optional

import httpx
from fastapi import FastAPI, Request
from PIL import Image, ImageOps, ImageEnhance

LINE_TOKEN      = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")

app = FastAPI()

# -------------------- 基本応答 --------------------
@app.get("/")       async def root():   return {"ok": True}
@app.get("/health") async def health(): return {"ok": True}

def line_headers(json_type=True)->Dict[str,str]:
    h={"Authorization": f"Bearer {LINE_TOKEN}"}
    if json_type: h["Content-Type"]="application/json"
    return h

async def line_reply(reply_token:str, text:str):
    try:
        async with httpx.AsyncClient(timeout=30) as ac:
            await ac.post(
                "https://api.line.me/v2/bot/message/reply",
                headers=line_headers(True),
                json={"replyToken": reply_token, "messages":[{"type":"text","text": text[:4900]}]},
            )
    except Exception as e:
        print("reply error:", e)

async def line_push(user_id:str, texts:List[str]):
    if not user_id: return
    try:
        async with httpx.AsyncClient(timeout=30) as ac:
            await ac.post(
                "https://api.line.me/v2/bot/message/push",
                headers=line_headers(True),
                json={"to": user_id, "messages":[{"type":"text","text": t[:4900]} for t in texts[:5]]},
            )
    except Exception as e:
        print("push error:", e)

async def fetch_line_image(message_id:str)->Optional[bytes]:
    url=f"https://api-data.line.me/v2/bot/message/{message_id}/content"
    try:
        async with httpx.AsyncClient(timeout=20) as ac:
            r=await ac.get(url, headers={"Authorization":f"Bearer {LINE_TOKEN}"})
            r.raise_for_status()
            return r.content
    except Exception as e:
        print("image fetch error:", e)
        return None

# -------------------- 画像前処理 --------------------
def _upscale(img:Image.Image, max_side:int=2600)->Image.Image:
    s=max_side/max(img.size)
    if s>1.0:
        img=img.resize((int(img.width*s), int(img.height*s)), Image.LANCZOS)
    return img

def preprocess(img_bytes:bytes)->Image.Image:
    img=Image.open(io.BytesIO(img_bytes))
    img=ImageOps.exif_transpose(img)
    img=img.convert("L")
    img=ImageOps.autocontrast(img)
    img=_upscale(img, 2600)
    img=ImageEnhance.Sharpness(img).enhance(1.25)
    img=ImageEnhance.Contrast(img).enhance(1.45)
    # 余白除去
    bw=img.point(lambda p: 255 if p>215 else 0)
    bbox=bw.getbbox()
    if bbox:
        l,t,r,b=bbox; pad=30
        l=max(0,l-pad); t=max(0,t-pad); r=min(img.width,r+pad); b=min(img.height,b+pad)
        img=img.crop((l,t,r,b))
    return img

def to_data_url(img:Image.Image)->str:
    buf=io.BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode("ascii")

def split_top_bottom(img:Image.Image)->List[Image.Image]:
    if img.height < 1100: return [img]
    mid=img.height//2
    return [img, img.crop((0,0,img.width,mid)), img.crop((0,mid,img.width,img.height))]

# -------------------- 数式パース & 厳密計算 --------------------
_num = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"

def parse_quadratic_text(text:str)->Optional[Dict[str,float]]:
    t = text.replace("　"," ").replace("，",",").strip()
    # a=,b=,c=
    m=re.search(r"二次.*?a\s*=\s*(%s).*?b\s*=\s*(%s).*?c\s*=\s*(%s)"%(_num,_num,_num), t, re.I)
    if m: return {"a":float(m.group(1)), "b":float(m.group(2)), "c":float(m.group(3))}
    # 二次 1,-3,2
    m=re.search(r"二次\s*(%s)\s*,\s*(%s)\s*,\s*(%s)"%(_num,_num,_num), t, re.I)
    if m: return {"a":float(m.group(1)), "b":float(m.group(2)), "c":float(m.group(3))}
    # 二次 1 -3 2
    m=re.search(r"二次\s*(%s)\s+(%s)\s+(%s)"%(_num,_num,_num), t, re.I)
    if m: return {"a":float(m.group(1)), "b":float(m.group(2)), "c":float(m.group(3))}
    # 変種: 二次 1.-3.2 → 整数を3つ拾う
    if "二次" in t:
        nums=re.findall(r"[+-]?\d+", t)
        if len(nums)>=3:
            a,b,c = map(float, nums[:3]); return {"a":a,"b":b,"c":c}
    return None

def solve_quadratic(a:float,b:float,c:float)->Dict[str,Any]:
    if a==0:
        # 1次 bx+c=0
        if b==0: return {"type":"invalid"}
        return {"type":"linear", "x":(-c/b), "eq": f"{b}x + {c} = 0"}
    D=b*b-4*a*c
    if D>0:  s=math.sqrt(D); return {"type":"real2","D":D,"x1":(-b+s)/(2*a),"x2":(-b-s)/(2*a)}
    if D==0: return {"type":"double","D":0,"x":(-b)/(2*a)}
    s=math.sqrt(-D); return {"type":"imag","D":D,"re":(-b)/(2*a),"im":s/(2*a)}

def solve_simul_2x2(a,b,c,d,e,f)->Dict[str,Any]:
    # a x + b y = c
    # d x + e y = f
    det=a*e-b*d
    if det==0: return {"type":"singular"}
    x=(c*e-b*f)/det; y=(a*f-c*d)/det
    return {"type":"ok","x":x,"y":y}

# -------------------- fx-CG50 手順（必ず [EXE] を明示） --------------------
def steps_quadratic(a,b,c)->str:
    return (
        "【電卓手順（fx-CG50）】\n"
        "1. [MENU] → 『Equation(方程式)』 → [EXE]\n"
        "2. [F2] Polynomial を選択 → 次数に 2 を入力 → [EXE]\n"
        f"3. a に {a} を入力 → [EXE]\n"
        f"4. b に {b} を入力 → [EXE]\n"
        f"5. c に {c} を入力 → [EXE]\n"
        "6. [EXE] で解を表示 → 必要なら [F6] Solve → [EXE]\n"
        "7. 係数を直すときは [EXIT] → 値を修正 → [EXE]\n"
        "（F1:Simul / F2:Poly / F3:Solver / F4:Del / F5:= / F6:Solve）"
    )

def steps_simul2(a,b,c,d,e,f)->str:
    return (
        "【電卓手順（fx-CG50）】\n"
        "1. [MENU] → 『Equation(方程式)』 → [EXE]\n"
        "2. [F1] Simultaneous を選択 → Unknowns に 2 → [EXE]\n"
        f"3. 1行目: a={a} → [EXE], b={b} → [EXE], 右辺={c} → [EXE]\n"
        f"4. 2行目: a'={d} → [EXE], b'={e} → [EXE], 右辺'={f} → [EXE]\n"
        "5. [EXE] で解を表示（x, y）\n"
        "6. 直すときは [EXIT] → 値を修正 → [EXE]\n"
        "（F1:Simul / F2:Poly / F3:Solver / F4:Del / F5:= / F6:Solve）"
    )

KEY_GUIDE = (
    "【fx-CG50 キー操作の総合ガイド】\n"
    "1.[MENU] 2.[EXIT] 3.[SHIFT]/[ALPHA]\n"
    "4.[F1]〜[F6]：Equation内は F1 Simul / F2 Poly / F3 Solver / F4 Del / F5 = / F6 Solve\n"
    "5.負号は『(-)』、小数点は『.』、各入力の確定は必ず [EXE]\n"
    "6.角度設定: [SHIFT][MENU](SETUP) → Angle\n"
    "7.アプリ: RUN-MAT / STAT / TABLE / GRAPH を用途で使い分け\n"
    "8.消去: [DEL] / 画面クリア: [AC/ON]\n"
)

USAGE = (
    "使い方：\n"
    "・写真を送る → 解析中… の後、最大2題について『式/解/番号付き手順』を返信\n"
    "・二次：『二次 a=1 b=-3 c=2』『二次 1,-3,2』『二次 1 -3 2』『二次 1.-3.2』の各表記に対応\n"
    "・『操作方法』でキー操作の総合ガイドを表示\n"
)

# -------------------- 画像→(タイプ+係数)抽出（JSON限定） --------------------
def _vision_prompt()->str:
    return (
        "次の画像群から、最大2題の“数式問題”だけを抽出してJSONで返してください。"
        "絶対に解かないでください。計算はしません。抽出だけです。\n"
        "返却は必ず JSON オブジェクト {\"problems\": [...]} 形式。各problemは以下いずれか：\n"
        "  ● 二次方程式: {\"type\":\"quadratic\", \"a\":数値, \"b\":数値, \"c\":数値, \"latex\":\"式(任意)\"}\n"
        "  ● 連立2元:    {\"type\":\"simul2\",    \"a\":数,\"b\":数,\"c\":数, \"d\":数,\"e\":数,\"f\":数, \"latex\":\"任意\"}\n"
        "数値は10進小数で。見つからない場合は空配列 problems: []. 余計な文章は一切禁止。"
    )

async def extract_problems_from_images(img_bytes:bytes)->List[Dict[str,Any]]:
    if not OPENAI_API_KEY:
        return []
    pre=preprocess(img_bytes)
    imgs=split_top_bottom(pre)
    payload={
        "model":"gpt-4o-mini",
        "response_format":{"type":"json_object"},
        "messages":[
            {"role":"system","content":"You only output strict JSON."},
            {"role":"user","content":[{"type":"text","text":_vision_prompt()}]+[
                {"type":"image_url","image_url":{"url":to_data_url(im)}} for im in imgs
            ]}
        ],
        "temperature":0
    }
    try:
        async with httpx.AsyncClient(timeout=90) as ac:
            r=await ac.post("https://api.openai.com/v1/chat/completions",
                            headers={"Authorization":f"Bearer {OPENAI_API_KEY}",
                                     "Content-Type":"application/json"},
                            json=payload)
            r.raise_for_status()
            txt=r.json()["choices"][0]["message"]["content"]
            obj=json.loads(txt)
            probs=obj.get("problems",[])
            # 型・キー簡易バリデーション
            out=[]
            for p in probs:
                if p.get("type")=="quadratic" and all(k in p for k in ["a","b","c"]):
                    out.append({"type":"quadratic","a":float(p["a"]), "b":float(p["b"]), "c":float(p["c"]), "latex":p.get("latex","")})
                elif p.get("type")=="simul2" and all(k in p for k in ["a","b","c","d","e","f"]):
                    out.append({"type":"simul2", "a":float(p["a"]),"b":float(p["b"]),"c":float(p["c"]),
                                "d":float(p["d"]),"e":float(p["e"]),"f":float(p["f"]), "latex":p.get("latex","")})
            return out[:2]
    except Exception as e:
        print("vision extract error:", e)
        return []

# -------------------- 返信生成 --------------------
def fmt_quadratic_answer(a,b,c)->str:
    res=solve_quadratic(a,b,c)
    eq=f"{a}x^2 + {b}x + {c} = 0"
    if res["type"]=="real2":
        head=f"【式】{eq}\n【判別式】D={res['D']}\n【解】x1={res['x1']}, x2={res['x2']}\n"
    elif res["type"]=="double":
        head=f"【式】{eq}\n【判別式】D=0\n【解】x={res['x']}（重解）\n"
    elif res["type"]=="imag":
        head=f"【式】{eq}\n【判別式】D={res['D']}<0\n【解】x={res['re']}±{abs(res['im'])}i\n"
    elif res["type"]=="linear":
        head=f"【式】{res['eq']}\n【解】x={res['x']}\n"
    else:
        head=f"【式】{eq}\n【解】解けません（a,b,c を確認）\n"
    return head + steps_quadratic(a,b,c)

def fmt_simul2_answer(a,b,c,d,e,f)->str:
    res=solve_simul_2x2(a,b,c,d,e,f)
    eq=f"{a}x + {b}y = {c},   {d}x + {e}y = {f}"
    if res["type"]=="ok":
        head=f"【式】{eq}\n【解】x={res['x']}, y={res['y']}\n"
    else:
        head=f"【式】{eq}\n【解】一意に定まりません（係数が特異）\n"
    return head + steps_simul2(a,b,c,d,e,f)

def numbered_blocks(blocks:List[str])->List[str]:
    out=[]
    for i,b in enumerate(blocks,1):
        out.append(f"【{i}問目】\n"+b)
    return out

# -------------------- Webhook --------------------
@app.post("/webhook")
async def webhook(req:Request):
    try:
        body=await req.json()
    except Exception as e:
        print("json error:", e); return {"ok":True}

    for ev in body.get("events",[]):
        try:
            await handle_event(ev)
        except Exception as e:
            print("event error:", e)
    return {"ok":True}

async def handle_event(ev:Dict[str,Any]):
    if ev.get("type")!="message": return
    msg=ev.get("message",{})
    mtype=msg.get("type")
    reply=ev.get("replyToken")
    uid=(ev.get("source") or {}).get("userId","")

    if mtype=="text":
        txt=(msg.get("text") or "").strip()
        if txt in ["操作方法","キー操作","help","ヘルプ"]:
            await line_reply(reply, KEY_GUIDE); return
        if txt in ["使い方","usage","つかいかた"]:
            await line_reply(reply, USAGE); return
        quad=parse_quadratic_text(txt)
        if quad:
            await line_reply(reply, fmt_quadratic_answer(quad["a"],quad["b"],quad["c"]))
            return
        # 将来拡張: 2元連立のテキストパースを追加可能
        await line_reply(reply, USAGE); return

    if mtype=="image":
        # 先に応答してハングを避ける
        await line_reply(reply, "解析中… 少し待ってね。")
        img=await fetch_line_image(msg.get("id",""))
        # 外部URLの場合（滅多にない）
        if not img and msg.get("contentProvider",{}).get("type")=="external":
            url=msg["contentProvider"].get("originalContentUrl")
            if url:
                try:
                    async with httpx.AsyncClient(timeout=20) as ac:
                        r=await ac.get(url); r.raise_for_status(); img=r.content
                except Exception as e:
                    print("ext img error:",e)
        if not img:
            await line_push(uid, ["画像の取得に失敗しました。受信後1分以内に再送してください。"])
            return

        async def run():
            probs=await extract_problems_from_images(img)
            if not probs:
                await line_push(uid, [
                    "式を特定できませんでした。撮影ヒント：画面いっぱい／正面／ピント／影少なめ／本文コントラスト強め。"
                ])
                return
            blocks=[]
            for p in probs:
                if p["type"]=="quadratic":
                    blocks.append(fmt_quadratic_answer(p["a"],p["b"],p["c"]))
                elif p["type"]=="simul2":
                    blocks.append(fmt_simul2_answer(p["a"],p["b"],p["c"],p["d"],p["e"],p["f"]))
            if not blocks:
                await line_push(uid, ["この問題タイプは現状未対応でした。順次拡張します。"])
                return
            for chunk in numbered_blocks(blocks):
                await line_push(uid, [chunk])
        asyncio.create_task(run())
        return

    # その他
    if reply:
        await line_reply(reply, "テキストまたは画像を送ってください。")
