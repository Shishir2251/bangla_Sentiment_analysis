from googleapiclient.discovery import build

API_KEY = ""

youtube = build("youtube", "v3", developerKey=API_KEY)

def get_comments(video_id, max_results=50):
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results
    )
    response = request.execute()

    comments = []
    for item in response["items"]:
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)

    return comments