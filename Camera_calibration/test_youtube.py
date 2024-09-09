import os
from googleapiclient.discovery import build

# Set your API key here
API_KEY = "YOUR_API_KEY"

# Set the channel ID (extract it from the channel URL)
CHANNEL_ID = "CHANNEL_ID"

def get_channel_videos(api_key, channel_id):
    youtube = build("youtube", "v3", developerKey=api_key)
    request = youtube.search().list(part="id", channelId=channel_id, type="video", maxResults=50)
    response = request.execute()
    video_ids = [item["id"]["videoId"] for item in response.get("items", [])]
    return video_ids

def get_video_duration(api_key, video_id):
    youtube = build("youtube", "v3", developerKey=api_key)
    request = youtube.videos().list(part="contentDetails", id=video_id)
    response = request.execute()
    duration = response["items"][0]["contentDetails"]["duration"]
    return duration

def main():
    total_duration = 0
    video_ids = get_channel_videos(API_KEY, CHANNEL_ID)
    for video_id in video_ids:
        duration = get_video_duration(API_KEY, video_id)
        # Parse duration (e.g., PT1H30M15S -> 1 hour, 30 minutes, 15 seconds)
        # You can use regex or string manipulation to extract hours, minutes, and seconds.
        # Add up the durations to get the total.
        # Example: total_duration += parsed_duration_in_seconds
    print(f"Total video duration: {total_duration} seconds")

if __name__ == "__main__":
    main()
