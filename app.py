import os
from flask import Flask, request, jsonify
import yt_dlp
import whisper

app = Flask(__name__)

# Load OpenAI Whisper model (Free Open-Source AI)
model = whisper.load_model("base")

@app.route('/clip', methods=['POST'])
def handle_clip():
    data = request.get_json()
    video_url = data.get('video_url')
    title = data.get('title', 'video')

    if not video_url:
        return jsonify({"error": "No URL provided"}), 400

    print(f"Processing video: {title} ({video_url})")

    # 1. Download YouTube Video using yt-dlp
    ydl_opts = {'outtmpl': 'downloaded_video.mp4', 'format': 'mp4'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    # 2. Transcribe Audio & Get Timestamps with Whisper AI
    result = model.transcribe("downloaded_video.mp4")
    print("Transcription complete:", result['text'][:100])

    # 3. Clean up local files
    if os.path.exists("downloaded_video.mp4"):
        os.remove("downloaded_video.mp4")

    return jsonify({"status": "success", "message": "Video processed"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
