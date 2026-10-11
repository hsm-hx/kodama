"""Presentation IDs only. Clients advertise verified speaker capabilities.

No paths, code, bone values or persistent conversation schema belong here.
"""
from collections.abc import Mapping, Sequence

PURPOSES = {
    "nod": "短い同意・受け止め",
    "wave": "挨拶・別れの小さな手振り",
    "bow": "お礼に添える小さな会釈",
    "shake": "穏やかな不同意（相手の感情・体験を否定しない）",
    "glanceDown": "少し考えて目を戻す",
    "relax": "一区切りの安堵",
    "lean": "興味を示す小さな身乗り",
    "shy": "親しい穏やかな場面の照れ",
    "chuckle": "和やかな場面の小さな笑い（悲しい話・謝罪には選ばない）",
    "gentleOffer": "低い手をそっと差し出す（物の受け渡しは行わない）",
    "listenTilt": "穏やかな問いかけの首傾げ",
    "gentleStretch": "休憩に添える小さな伸び",
}


def capabilities(character_ids: Sequence[str], offered: Mapping | None = None) -> dict[str, list[str]]:
    offered = offered if isinstance(offered, Mapping) else {}
    return {s: list(dict.fromkeys(x for x in offered.get(s, ()) if isinstance(x, str) and x in PURPOSES))
            if isinstance(offered.get(s, ()), (list, tuple)) else [] for s in character_ids}


def normalize_gesture(speaker: str, value: object, offered: Mapping | None = None) -> str:
    allowed = capabilities([speaker], offered)[speaker]
    return value if isinstance(value, str) and value in allowed else "none"


def gesture_rules(offered: Mapping) -> str:
    lines = ["# 表示用の任意の仕草", "- utterances の各台詞に gesture を付けてよい。省略・通常は none。毎回動かさず、台詞に自然な理由があるときだけ一つ選ぶ。",
             "- 重い話、悲しい話、謝罪、長い説明、迷うときは none。笑い・手振り・照れを雰囲気だけで付けない。同じ仕草を連続させない。",
             "- 表示側が停止や動きを控える設定なら再生しない。音声・表情・物体操作を意味しない。パス、URL、コード、骨の値は返さない。"]
    for speaker, ids in offered.items():
        lines.append(f"- {speaker}: none" + "".join(f" / {g} ({PURPOSES[g]})" for g in ids))
    return "\n".join(lines)
