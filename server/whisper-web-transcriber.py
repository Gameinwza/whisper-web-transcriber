from flask import Flask, request, jsonify
import threading, os, uuid, tempfile, subprocess
import whisper
from enum import Enum

app = Flask(__name__)
model = whisper.load_model("base", device="cpu")

class Status(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    DONE = "done"
    ERROR = "error"

current_status = Status.IDLE
current_text = ""
current_error = ""
lock = threading.Lock()

def convert_to_wav(path):
    out = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "16000", out],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )
    return out

@app.route("/status")
def status():
    return jsonify(
        status=current_status.value,
        text=current_text,
        error=current_error
    )

@app.route("/transcribe", methods=["POST"])
def transcribe():
    global current_status, current_text, current_error

    if lock.locked():
        return jsonify({"error": "busy"}), 409

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "no file"}), 400

    tmp = os.path.join(tempfile.gettempdir(), file.filename)
    file.save(tmp)

    def run():
        global current_status, current_text, current_error
        with lock:
            try:
                current_status = Status.LISTENING
                use = tmp
                if not tmp.endswith(".wav"):
                    use = convert_to_wav(tmp)

                result = model.transcribe(use, language="th", fp16=False)
                current_text = result["text"]
                current_status = Status.DONE
            except Exception as e:
                current_error = str(e)
                current_status = Status.ERROR
            finally:
                for f in [tmp, use]:
                    if os.path.exists(f):
                        os.remove(f)

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "uploaded"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import threading, os, uuid, tempfile, subprocess
# import whisper
# from enum import Enum

# app = Flask(__name__)
# CORS(app)

# print("⏳ Loading Whisper model...")
# model = whisper.load_model("medium", device="cpu")
# print("✅ Whisper ready")

# # =============================
# # State
# # =============================
# class Status(Enum):
#     IDLE = "idle"
#     UPLOADING = "uploading"
#     LISTENING = "listening"
#     DONE = "done"
#     ERROR = "error"

# current_status = Status.IDLE
# current_text = ""
# current_error = ""
# lock = threading.Lock()

# # =============================
# # Utils
# # =============================
# def convert_to_wav(path):
#     out = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.wav")
#     subprocess.run([
#         "ffmpeg", "-y",
#         "-i", path,
#         "-af", "silenceremove=start_periods=1:start_silence=0.5:start_threshold=-50dB",
#         "-ac", "1",
#         "-ar", "16000",
#         out
#     ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
#     return out

# # =============================
# # Routes
# # =============================
# @app.route("/status")
# def status():
#     return jsonify(
#         status=current_status.value,
#         text=current_text,
#         error=current_error
#     )

# @app.route("/transcribe", methods=["POST"])
# def transcribe():
#     global current_status, current_text, current_error

#     if lock.locked():
#         return jsonify({"error": "busy"}), 409

#     file = request.files.get("file")
#     if not file:
#         return jsonify({"error": "no file"}), 400

#     current_status = Status.UPLOADING
#     current_text = ""
#     current_error = ""

#     tmp = os.path.join(tempfile.gettempdir(), file.filename)
#     file.save(tmp)

#     def run():
#         global current_status, current_text, current_error
#         with lock:
#             use = tmp
#             try:
#                 current_status = Status.LISTENING

#                 if not tmp.lower().endswith(".wav"):
#                     use = convert_to_wav(tmp)

#                 result = model.transcribe(
#                     use,
#                     language="th",
#                     fp16=False,
#                     temperature=0,
#                     no_speech_threshold=0.6,
#                     condition_on_previous_text=False
#                 )

#                 current_text = result["text"]
#                 current_status = Status.DONE

#             except Exception as e:
#                 current_error = str(e)
#                 current_status = Status.ERROR

#             finally:
#                 for f in [tmp, use]:
#                     if os.path.exists(f):
#                         os.remove(f)

#     threading.Thread(target=run, daemon=True).start()
#     return jsonify({"status": "ok"})

# # =============================
# # Run
# # =============================
# if __name__ == "__main__":
#     print("🚀 Whisper server running on :5000")
#     app.run(host="0.0.0.0", port=5000, debug=False)