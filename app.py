"""
Flask Web Application — Hệ thống Gợi ý Phim Cá nhân hoá
Chạy: python app.py  →  truy cập http://localhost:5000
"""
import os
import sys
import json
import zipfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen, urlretrieve

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Khởi tạo Recommender (lazy — chỉ tải khi dữ liệu sẵn sàng)
# ---------------------------------------------------------------------------
recommender = None
data_ready = False
model_ready = False


def _load_env_file(env_path):
    """Load simple KEY=VALUE pairs from .env into process environment."""
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ[key] = value


def _github_get_json(url, headers):
    req = Request(url, headers=headers, method="GET")
    with urlopen(req, timeout=8) as resp:
        payload = resp.read().decode("utf-8")
        return json.loads(payload)


def _fetch_team_members():
    collaborators_url = "https://api.github.com/repos/thuynhi2004/GoiYPhim/collaborators?per_page=100"
    token = os.getenv("GITHUB_TOKEN", "").strip()

    if not token:
        print("[WARN] Thiếu GITHUB_TOKEN, không thể lấy danh sách thành viên động.", file=sys.stderr)
        return []

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "GoiYPhim-Flask-App",
    }
    try:
        collaborators = _github_get_json(collaborators_url, headers)

        team_members = {}

        for collaborator in collaborators:
            if collaborator.get("type") != "User":
                continue
            login = collaborator.get("login", "")
            if not login:
                continue
            team_members[login] = {
                "login": login,
                "name": collaborator.get("login", login),
                "html_url": collaborator.get("html_url", "#"),
                "avatar_url": collaborator.get("avatar_url", ""),
                "contributions": 0,
            }

        members = sorted(
            team_members.values(),
            key=lambda m: (-int(m.get("contributions", 0)), m.get("login", "").lower()),
        )
        return members
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"[WARN] Không lấy được collaborators từ GitHub: {exc}", file=sys.stderr)
        return []


def _get_team_members_for_render():
    """Render team members from GitHub API only."""
    return _fetch_team_members()


def _ensure_movielens_data(data_file_path):
    """Auto-download MovieLens 100K if required files are missing."""
    if os.path.exists(data_file_path):
        return True

    project_root = os.path.dirname(__file__)
    data_dir = os.path.join(project_root, "data")
    archive_path = os.path.join(data_dir, "ml-100k.zip")
    dataset_url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

    os.makedirs(data_dir, exist_ok=True)
    print("[INFO] Không tìm thấy dữ liệu MovieLens. Bắt đầu tải tự động...", file=sys.stderr)
    try:
        urlretrieve(dataset_url, archive_path)
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(data_dir)
        if os.path.exists(archive_path):
            os.remove(archive_path)
    except Exception as exc:
        print(f"[ERROR] Tải dữ liệu tự động thất bại: {exc}", file=sys.stderr)
        return False

    return os.path.exists(data_file_path)


def _init_recommender():
    global recommender, data_ready, model_ready
    data_file = os.path.join(os.path.dirname(__file__), "data", "ml-100k", "u.data")
    data_ready = _ensure_movielens_data(data_file)
    if not data_ready:
        return

    try:
        from model.recommender import MovieRecommender
        recommender = MovieRecommender()
        model_ready = True
    except Exception as exc:
        print(f"[ERROR] Không khởi tạo được Recommender: {exc}", file=sys.stderr)
        model_ready = False


_init_recommender()

_load_env_file(os.path.join(os.path.dirname(__file__), ".env"))

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(app.static_folder, "favicon.svg", mimetype="image/svg+xml")

@app.route("/")
def index():
    total_users  = recommender.get_total_users()  if model_ready else 943
    total_movies = recommender.get_total_movies() if model_ready else 1682
    team_members = _get_team_members_for_render()
    return render_template(
        "index.html",
        data_ready=data_ready,
        model_ready=model_ready,
        total_users=total_users,
        total_movies=total_movies,
        team_members=team_members,
    )


@app.route("/recommend", methods=["POST"])
def recommend():
    if not model_ready:
        return redirect(url_for("index"))

    user_id = request.form.get("user_id", type=int)
    n       = request.form.get("n",       type=int, default=10)
    max_u   = recommender.get_total_users()

    if not user_id or not (1 <= user_id <= max_u):
        return render_template(
            "index.html",
            error=f"User ID phải nằm trong khoảng 1 đến {max_u}.",
            data_ready=data_ready,
            model_ready=model_ready,
            total_users=max_u,
            total_movies=recommender.get_total_movies(),
            team_members=_get_team_members_for_render(),
        )

    recs       = recommender.recommend(user_id, n)
    history    = recommender.get_user_ratings(user_id, n=10)
    evaluation = recommender.get_evaluation()

    return render_template(
        "recommend.html",
        user_id=user_id,
        recommendations=recs,
        history=history,
        evaluation=evaluation,
    )


@app.route("/evaluate")
def evaluate():
    if not model_ready:
        return redirect(url_for("index"))
    return render_template("evaluate.html", evaluation=recommender.get_evaluation())


@app.route("/retrain", methods=["POST"])
def retrain():
    """Huấn luyện lại mô hình (API endpoint, gọi bằng AJAX)."""
    if not data_ready:
        return jsonify({"error": "Dữ liệu chưa sẵn sàng — chạy download_data.py trước."}), 400
    global model_ready
    try:
        result = recommender.train_and_evaluate()
        model_ready = True
        return jsonify({"success": True, "metrics": result})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/recommend")
def api_recommend():
    """JSON API: GET /api/recommend?user_id=42&n=10"""
    if not model_ready:
        return jsonify({"error": "Mô hình chưa sẵn sàng"}), 503
    user_id = request.args.get("user_id", type=int)
    n       = request.args.get("n",       type=int, default=10)
    if not user_id:
        return jsonify({"error": "user_id là bắt buộc"}), 400
    return jsonify({"user_id": user_id, "recommendations": recommender.recommend(user_id, n)})


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
