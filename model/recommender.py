"""
Mô hình gợi ý phim sử dụng thuật toán SVD (Collaborative Filtering)
Thư viện: scikit-surprise  |  Dữ liệu: MovieLens 100K
"""
import os
import pickle

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split

# Tên 19 thể loại theo đúng thứ tự cột trong u.item
GENRES = [
    "Unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

_BASE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_BASE, "..", "data", "ml-100k")
_MODEL_PATH = os.path.join(_BASE, "saved_model.pkl")


class MovieRecommender:
    """
    Lớp bao bọc toàn bộ pipeline: tải dữ liệu → huấn luyện SVD → đánh giá → gợi ý.
    """

    def __init__(self):
        self.model = None
        self.movies_df: pd.DataFrame = None
        self.ratings_df: pd.DataFrame = None
        self.evaluation: dict = None

        self._load_data()
        self._load_or_train_model()

    # ------------------------------------------------------------------
    # Tải dữ liệu thô
    # ------------------------------------------------------------------
    def _load_data(self):
        data_dir = os.path.normpath(_DATA_DIR)

        # --- Phim (u.item) ---
        col_names = ["movie_id", "title", "release_date", "video_release", "imdb_url"] + GENRES
        self.movies_df = pd.read_csv(
            os.path.join(data_dir, "u.item"),
            sep="|",
            names=col_names,
            encoding="latin-1",
        )

        # Rút trích thể loại thành chuỗi "Action, Drama, ..."
        self.movies_df["genre_str"] = self.movies_df.apply(
            lambda r: ", ".join(g for g in GENRES if r.get(g, 0) == 1) or "Unknown",
            axis=1,
        )
        # Rút năm từ tiêu đề  "Toy Story (1995)" → "1995"
        self.movies_df["year"] = self.movies_df["title"].str.extract(r"\((\d{4})\)$")
        self.movies_df = self.movies_df[["movie_id", "title", "year", "genre_str"]].copy()

        # --- Đánh giá (u.data) ---
        self.ratings_df = pd.read_csv(
            os.path.join(data_dir, "u.data"),
            sep="\t",
            names=["user_id", "movie_id", "rating", "timestamp"],
        )

    # ------------------------------------------------------------------
    # Load mô hình đã lưu, hoặc huấn luyện mới
    # ------------------------------------------------------------------
    def _load_or_train_model(self):
        model_path = os.path.normpath(_MODEL_PATH)
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    saved = pickle.load(f)
                self.model = saved["model"]
                self.evaluation = saved["evaluation"]
                print("✓ Đã tải mô hình từ cache.")
                return
            except Exception:
                pass  # file lỗi → huấn luyện lại
        self.train_and_evaluate()

    # ------------------------------------------------------------------
    # Huấn luyện + đánh giá
    # ------------------------------------------------------------------
    def train_and_evaluate(self) -> dict:
        """
        1. Xây dựng dataset Surprise từ ratings_df
        2. 5-fold cross-validation → CV RMSE, CV MAE
        3. Train/test split (80/20) → Test RMSE, Test MAE
        4. Huấn luyện lại trên TOÀN BỘ dữ liệu để dùng cho gợi ý
        5. Lưu mô hình
        """
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            self.ratings_df[["user_id", "movie_id", "rating"]], reader
        )

        algo = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)

        # Bước 2 — Kiểm tra chéo 5-fold
        print("Đang thực hiện kiểm tra chéo 5-fold (có thể mất 2-3 phút)...")
        cv_results = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5, verbose=False)

        # Bước 3 — Đánh giá trên tập test
        print("Đánh giá trên tập test 20%...")
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
        algo.fit(trainset)
        preds = algo.test(testset)
        test_rmse = float(accuracy.rmse(preds, verbose=False))
        test_mae = float(accuracy.mae(preds, verbose=False))

        # Bước 4 — Huấn luyện lại trên toàn bộ dữ liệu
        print("Huấn luyện mô hình trên toàn bộ dữ liệu...")
        full_trainset = data.build_full_trainset()
        algo.fit(full_trainset)

        self.model = algo
        self.evaluation = {
            "algorithm": "SVD (Phân tích ma trận)",
            "n_factors": 100,
            "n_epochs": 20,
            "learning_rate": 0.005,
            "regularization": 0.02,
            "cv_rmse_mean": float(cv_results["test_rmse"].mean()),
            "cv_rmse_std":  float(cv_results["test_rmse"].std()),
            "cv_mae_mean":  float(cv_results["test_mae"].mean()),
            "cv_mae_std":   float(cv_results["test_mae"].std()),
            "test_rmse": test_rmse,
            "test_mae":  test_mae,
            "total_users":   int(self.ratings_df["user_id"].nunique()),
            "total_movies":  int(self.movies_df["movie_id"].nunique()),
            "total_ratings": int(len(self.ratings_df)),
        }

        # Lưu
        with open(os.path.normpath(_MODEL_PATH), "wb") as f:
            pickle.dump({"model": self.model, "evaluation": self.evaluation}, f)

        print(
            f"✓ Xong! CV RMSE={self.evaluation['cv_rmse_mean']:.4f} | "
            f"Test RMSE={test_rmse:.4f} | Test MAE={test_mae:.4f}"
        )
        return self.evaluation

    # ------------------------------------------------------------------
    # Gợi ý Top-N phim cho một người dùng
    # ------------------------------------------------------------------
    def recommend(self, user_id: int, n: int = 10) -> list[dict]:
        if self.model is None:
            return []

        # Phim người dùng đã đánh giá
        rated_ids = set(
            self.ratings_df[self.ratings_df["user_id"] == user_id]["movie_id"].tolist()
        )

        # Dự đoán điểm cho TẤT CẢ phim chưa xem
        unrated = [mid for mid in self.movies_df["movie_id"].tolist() if mid not in rated_ids]
        preds = [(mid, self.model.predict(user_id, mid).est) for mid in unrated]
        preds.sort(key=lambda x: x[1], reverse=True)

        results = []
        for movie_id, est in preds[:n]:
            row = self.movies_df[self.movies_df["movie_id"] == movie_id]
            if row.empty:
                continue
            r = row.iloc[0]
            results.append(
                {
                    "movie_id": int(movie_id),
                    "title": r["title"],
                    "year": str(r["year"]) if pd.notna(r["year"]) else "",
                    "genres": r["genre_str"],
                    "predicted_rating": round(float(est), 2),
                    "score_pct": round(float(est) / 5.0 * 100),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Lịch sử đánh giá của người dùng (Top-N theo điểm thực tế)
    # ------------------------------------------------------------------
    def get_user_ratings(self, user_id: int, n: int = 10) -> list[dict]:
        user_df = (
            self.ratings_df[self.ratings_df["user_id"] == user_id]
            .sort_values("rating", ascending=False)
            .head(n)
        )
        results = []
        for _, row in user_df.iterrows():
            m = self.movies_df[self.movies_df["movie_id"] == row["movie_id"]]
            if m.empty:
                continue
            m = m.iloc[0]
            results.append(
                {
                    "movie_id": int(row["movie_id"]),
                    "title": m["title"],
                    "year": str(m["year"]) if pd.notna(m["year"]) else "",
                    "genres": m["genre_str"],
                    "actual_rating": int(row["rating"]),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_evaluation(self) -> dict:
        return self.evaluation

    def get_total_users(self) -> int:
        return int(self.ratings_df["user_id"].nunique()) if self.ratings_df is not None else 943

    def get_total_movies(self) -> int:
        return int(len(self.movies_df)) if self.movies_df is not None else 1682
