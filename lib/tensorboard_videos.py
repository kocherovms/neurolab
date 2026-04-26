# https://share.google/aimode/Lxjii23lsAA5gYfxf

from flask import Flask, render_template_string, send_from_directory
import os

app = Flask(__name__)
VIDEO_DIR = '/videos'

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Video Browser</title>
    <style>
        html, body {
            height: 100%;
            margin: 0;
            padding: 0;
            overflow: hidden; 
            font-family: sans-serif;
            background-color: white;
        }

        .container {
            display: flex;
            flex-direction: row;
            height: 100vh;
            width: 100vw;
        }

        .sidebar {
            flex: 0 0 20%;
            min-width: 250px;
            border-right: 1px solid #ccc;
            padding: 20px;
            box-sizing: border-box;
            overflow-y: auto;
            color: black;
        }

        .video-section {
            flex: 1;
            display: flex;
            flex-direction: column; /* Stack Title and Video vertically */
            background-color: #f0f0f0;
            padding: 20px;
            box-sizing: border-box;
            min-width: 0;
        }

        .video-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
            text-align: center;
            flex-shrink: 0;
        }

        .player-wrapper {
            flex-grow: 1;
            display: flex;
            justify-content: center;
            min-height: 0;
        }

        video {
            max-width: 100%;
            max-height: 100%;
            background: black;
        }

        /* Highlight the active video in the list */
        .active-video {
            font-weight: bold;
            color: black;
            background-color: #eef;
        }

        h1 { font-size: 1.2em; margin-top: 0; }
        a { color: #0000EE; text-decoration: underline; }
        ul { padding-left: 20px; }
        li { margin-bottom: 8px; }
        hr { border: 0; border-top: 1px solid #ccc; margin: 15px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h1>/{{ subpath }}</h1>

            <hr>
            <strong>Videos:</strong>
            <ul>
            {% for video in videos %}
                <li class="{{ 'active-video' if video == current_video else '' }}">
                    <a href="/video/{{ subpath }}/{{ video }}">{{ video }}</a>
                </li>
            {% endfor %}
            </ul>
        </div>

        <div class="video-section">
            {% if current_video %}
                <div class="video-title">Playing: {{ current_video }}</div>
                <div class="player-wrapper">
                    <video controls autoplay>
                        <source src="/stream/{{ subpath }}/{{ current_video }}" type="video/mp4">
                    </video>
                </div>
            {% else %}
                <div class="player-wrapper" style="align-items: center;">
                    <p style="color: #999;">Select a video from the sidebar to start playback</p>
                </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
'''


@app.route('/', defaults={'req_path': ''})
@app.route('/<path:req_path>')
@app.route('/video/<path:req_path>')
def index(req_path):
    # Ensure the path stays inside the /videos directory for security
    abs_path = os.path.normpath(os.path.join(VIDEO_DIR, req_path))
    
    if not abs_path.startswith(VIDEO_DIR) or not os.path.exists(abs_path):
        return abort(404)

    # Determine if we are viewing a folder or playing a file
    is_video = os.path.isfile(abs_path)
    current_dir = os.path.dirname(req_path) if is_video else req_path
    list_path = os.path.join(VIDEO_DIR, current_dir)

    # Separate files and folders
    items = os.listdir(list_path)
    dirs = [i for i in items if os.path.isdir(os.path.join(list_path, i))]
    videos = [i for i in items if i.lower().endswith(('.mp4', '.webm')) and os.path.isfile(os.path.join(list_path, i))]

    return render_template_string(
        HTML_TEMPLATE, 
        subpath=current_dir.strip('/'),
        parent_dir=os.path.dirname(current_dir.strip('/')),
        dirs=dirs, 
        videos=videos,
        current_video=os.path.basename(req_path) if is_video else None
    )

@app.route('/stream/<path:filename>')
def stream(filename):
    # send_from_directory safely handles path traversal within the specified directory
    return send_from_directory(VIDEO_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
