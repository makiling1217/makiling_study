# --- LINE 画像取得（差し替え） ---
async def fetch_line_image(message_id: str) -> bytes | None:
    """
    LINEの画像バイナリを取得。message_id は image メッセージの event["message"]["id"]。
    取得先は api-data.line.me。1分以内に取得しないと404が返る。
    """
    url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
    headers = {"Authorization": f"Bearer {LINE_TOKEN}"}  # Content-Typeは不要
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
