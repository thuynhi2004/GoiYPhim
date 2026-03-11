/* main.js — CineSuggest frontend interactions */

// Màu sắc cho từng thể loại phim
const GENRE_COLORS = {
  'Action':      '#e50914',
  'Adventure':   '#e67e22',
  'Animation':   '#9b59b6',
  "Children's":  '#2ecc71',
  'Comedy':      '#f39c12',
  'Crime':       '#e74c3c',
  'Documentary': '#1abc9c',
  'Drama':       '#3498db',
  'Fantasy':     '#8e44ad',
  'Film-Noir':   '#666',
  'Horror':      '#c0392b',
  'Musical':     '#16a085',
  'Mystery':     '#2980b9',
  'Romance':     '#e91e8c',
  'Sci-Fi':      '#00bcd4',
  'Thriller':    '#d35400',
  'War':         '#7f8c8d',
  'Western':     '#a0522d',
};

document.addEventListener('DOMContentLoaded', function () {

  // ---- Hiển thị loading overlay khi submit form ----
  const form    = document.getElementById('recommendForm');
  const overlay = document.getElementById('loadingOverlay');

  if (form && overlay) {
    form.addEventListener('submit', function () {
      overlay.classList.add('show');
    });
  }

  // ---- Set progress bar width từ data-attribute ----
  document.querySelectorAll('.progress-bar[data-width]').forEach(function (bar) {
    bar.style.width = bar.dataset.width + '%';
  });

  // ---- Tô màu genre badges ----
  document.querySelectorAll('.genre-badge').forEach(function (badge) {
    const genre = badge.dataset.genre || badge.textContent.trim();
    const color = GENRE_COLORS[genre];
    if (color) {
      badge.style.backgroundColor = color + '22';
      badge.style.borderColor     = color + '66';
      badge.style.color           = color;
    }
  });

});
