import json
import math
import csv
from dateutil import parser

from tqdm import tqdm

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

def keep_from_list_dict_only_chosen_attributes(list_dict,chosen_attributes):
    new_list_dict = []
    for val_dict in list_dict:
        record = {}
        for chosen_attribute in chosen_attributes:
            if chosen_attribute not in val_dict.keys():
                raise("Chosen attribute not in data")
            record[chosen_attribute] = val_dict[chosen_attribute]
        new_list_dict.append(record)
    return new_list_dict

def keep_from_list_dict_all_but_chosen_attributes(list_dict, not_chosen_attributes):
    new_list_dict = []
    for val_dict in list_dict:
        record = {}
        for key in val_dict.keys():
            if key not in not_chosen_attributes:
                record[key] = val_dict[key]
        new_list_dict.append(record)
    return new_list_dict

def write_processed_data_to_csv(path, data):
    keys = data[0].keys()
    with open(path, mode='w', newline='',encoding='utf-8') as file:
        writer = csv.DictWriter(file, keys)
        writer.writeheader()
        writer.writerows(data)




users = import_file("content/users.jsonl")
sessions = import_file("content/sessions.jsonl")
tracks = import_file("content/tracks.jsonl")

# Preprocess tracks into a dictionary for quick lookup
tracks_dic = {track["id"]: track for track in tracks}

# Group sessions by user
user_sessions = {}

print("Processing sessions")
for session in tqdm(sessions):
    session["timestamp"] = parser.isoparse(session["timestamp"])
    user_sessions.setdefault(session["user_id"], []).append(session)


print("Processing Users")
for user in tqdm(users):
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
        "listening_frequency": len(sessions_by_id) / total_days,
        "avr_session_len": sum((s[-1]["timestamp"] - s[0]["timestamp"]).total_seconds() / 60 for s in sessions_by_id.values()) / len(sessions_by_id),
        "like_rate_per_day": stats["likes"] / max(total_days, 1),
        "skip_rate_per_day": stats["skips"] / max(total_days, 1),
        "play_rate_per_day": stats["plays"] / max(total_days, 1),
        "ad_rate_per_day": stats["ads"] / max(total_days, 1),
        "rate_of_quits_of_sessions_with_ads": stats["ad_quits"] / max(len(sessions_by_id), 1),
        "artist_diversity": calculate_diversity(stats["artists"]),
        "artist_diversity_gini": calculate_gini(stats["artists"]),
        "avg_year_published_of_songs_played": stats["year"] / max(stats["plays"], 1)
    })

    for param in stats["track_params"]:
        param_name = "avg_song_"+param
        user[param_name] = stats["track_params"][param] / max(stats["plays"], 1)

    user['favourite_genres_count'] = len(user['favourite_genres'])
    user.pop("favourite_genres", None)
    user.pop("user_id", None)
    user.pop("city", None)


Y = keep_from_list_dict_only_chosen_attributes(users,["premium_user"])
Y_binarised = [{"premium_user":1} if y["premium_user"] == True else {"premium_user":0} for y in Y ]
print(Y_binarised[0].keys())

write_processed_data_to_csv("content/custom_data/processed_Y.csv", Y_binarised)


X = keep_from_list_dict_only_chosen_attributes(users, ["play_rate_per_day", "ad_rate_per_day", "like_rate_per_day", "skip_rate_per_day", "artist_diversity_gini", "avr_session_len"])
print(X[0].keys())
X_features = [str(key) for key in X[0].keys()]
with open('content/custom_data/attributes_required.json', 'w') as file:
    json.dump(X_features, file, indent=4)

write_processed_data_to_csv("content/custom_data/processed_X.csv", X)


all_attributes_X = keep_from_list_dict_all_but_chosen_attributes(users,["premium_user"])
print(all_attributes_X[0].keys())
write_processed_data_to_csv("content/custom_data/all_attributes_X.csv", all_attributes_X)




