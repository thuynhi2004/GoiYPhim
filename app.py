"""
Flask Web Application — Hệ thống Gợi ý Phim Cá nhân hoá
Chạy: python app.py  →  truy cập http://localhost:5000
"""
import os
import sys

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Khởi tạo Recommender (lazy — chỉ tải khi dữ liệu sẵn sàng)
# ---------------------------------------------------------------------------
recommender = None
data_ready = False
model_ready = False


def _init_recommender():
    global recommender, data_ready, model_ready
    data_file = os.path.join(os.path.dirname(__file__), "data", "ml-100k", "u.data")
    data_ready = os.path.exists(data_file)
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
    return render_template(
        "index.html",
        data_ready=data_ready,
        model_ready=model_ready,
        total_users=total_users,
        total_movies=total_movies,
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
