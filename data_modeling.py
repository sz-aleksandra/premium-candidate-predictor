import json
import math
import csv
from dateutil import parser

def import_file(path):
    with open(path, 'r', encoding='utf-8') as json_file:
        json_list = list(json_file)

    return [json.loads(json_str) for json_str in json_list]

def calculate_diversity(song_listens):
    total_listens = sum(song_listens.values())
    num_songs = len(song_listens)
    if total_listens == 0 or num_songs == 0:
        return 0

    shannon_index = -sum(
        (listens / total_listens) * math.log(listens / total_listens)
        for listens in song_listens.values() if listens > 0
    )

    return shannon_index / math.log(num_songs)

def calculate_gini(artist_listens):
    listens = sorted(artist_listens.values())
    total_listens = sum(listens)
    if total_listens == 0:
        return 0

    cumulative = sum(i * count for i, count in enumerate(listens, start=1))
    n = len(listens)
    gini = (2 * cumulative) / (n * total_listens) - (n + 1) / n
    return gini

users = import_file("content/users.jsonl")
sessions = import_file("content/sessions.jsonl")
tracks = import_file("content/tracks.jsonl")

# Preprocess tracks into a dictionary for quick lookup
tracks_dic = {track["id"]: track for track in tracks}

# Group sessions by user
user_sessions = {}
for session in sessions:
    session["timestamp"] = parser.isoparse(session["timestamp"])
    user_sessions.setdefault(session["user_id"], []).append(session)

for user in users:
    u_sess = user_sessions.get(user["user_id"], [])
    sessions_by_id = {}

    stats = {
        "likes": 0,
        "skips": 0,
        "plays": 0,
        "ads": 0,
        "ad_quits": 0,
        "track_params": {param: 0 for param in tracks[0] if param not in ["id", "artist_id", "name", "duration_ms", "release_date", "mode"]},
        "year": 0,
        "artists": {},
        "timestamps": []
    }

    for session in u_sess:
        sessions_by_id.setdefault(session["session_id"], []).append(session)

        event = session["event_type"]
        if event == "Play":
            stats["plays"] += 1
            track = tracks_dic[session["track_id"]]
            for param in stats["track_params"]:
                stats["track_params"][param] += track[param]
            stats["year"] += int(track["release_date"][:4])
            artist_id = track["artist_id"]
            stats["artists"][artist_id] = stats["artists"].get(artist_id, 0) + 1
        elif event == "Skip":
            stats["skips"] += 1
        elif event == "Like":
            stats["likes"] += 1
        elif event == "Advertisement":
            stats["ads"] += 1
        elif event == "BuyPremium":
            stats["premium_time"] = session["timestamp"]
            break

    for session_id, session_events in sessions_by_id.items():
        session_duration = (session_events[-1]["timestamp"] - session_events[0]["timestamp"]).total_seconds() / 60
        session_end_event = session_events[-1]["event_type"]

        # Add session duration
        stats["timestamps"].append(session_duration)

        # Check if the session ended with an Advertisement
        if session_end_event == "Advertisement":
            stats["ad_quits"] += 1

    first_session_time = u_sess[0]["timestamp"] if u_sess else None
    last_session_time = stats.get("premium_time", u_sess[-1]["timestamp"] if u_sess else None)
    total_days = (last_session_time - first_session_time).total_seconds() / (60 * 60 * 24) if first_session_time and last_session_time else 1

    user.pop("name", None)
    user.pop("street", None)

    user.update({
        "frequency": len(sessions_by_id) / total_days,
        "avr_len": sum((s[-1]["timestamp"] - s[0]["timestamp"]).total_seconds() / 60 for s in sessions_by_id.values()) / len(sessions_by_id),
        "likes": stats["likes"] / max(total_days, 1),
        "skips": stats["skips"] / max(total_days, 1),
        "plays": stats["plays"] / max(total_days, 1),
        "ads": stats["ads"] / max(total_days, 1),
        "ad_quits": stats["ad_quits"] / max(len(sessions_by_id), 1),
        "artist_diversity": calculate_diversity(stats["artists"]),
        "artist_gini": calculate_gini(stats["artists"]),
        "year": stats["year"] / max(stats["plays"], 1)
    })

    for param in stats["track_params"]:
        user[param] = stats["track_params"][param] / max(stats["plays"], 1)

# Save curated data to CSV
keys = users[0].keys()
with open('content/processed_data.csv', mode='w', newline='') as file:
    writer = csv.DictWriter(file, keys)
    writer.writeheader()
    writer.writerows(users)

print("Data processing and export completed.")


