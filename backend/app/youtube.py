import os
from dotenv import load_dotenv
from googleapiclient.discovery import build

# Load env
load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    raise ValueError("❌ YOUTUBE_API_KEY is missing")

youtube = build("youtube", "v3", developerKey=API_KEY)


def get_comments(video_id, max_results=50):
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results
    )

    response = request.execute()

    comments = []
    for item in response.get("items", []):
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)

    return comments