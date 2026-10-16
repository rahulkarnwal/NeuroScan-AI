import { useCallback, useEffect, useState } from "react";

import "./App.css";
const API_URL = "http://127.0.0.1:8000";
const CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"];

function formatClassName(name = "") {
  if (name === "notumor") return "No Tumor";
  return name ? name.charAt(0).toUpperCase() + name.slice(1) : "Unknown";
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [clearingHistory, setClearingHistory] = useState(false);

  const loadHistory = useCallback(async () => {
    setHistoryLoading(true);
    setHistoryError("");

    try {
      const response = await fetch(`${API_URL}/history`);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Could not load prediction history.");
      }

      setHistory(Array.isArray(data.history) ? data.history : []);
    } catch (err) {
      setHistoryError(
        err.message === "Failed to fetch"
          ? "Cannot connect to the backend. Make sure FastAPI is running on port 8000."
          : err.message
      );
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  function handleFileChange(event) {
    const selectedFile = event.target.files?.[0];

    setResult(null);
    setError("");

    if (!selectedFile) return;

    const allowedTypes = ["image/jpeg", "image/png", "image/webp"];

    if (!allowedTypes.includes(selectedFile.type)) {
      setFile(null);
      setPreview("");
      setError("Please select a JPEG, PNG, or WebP image.");
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setFile(null);
      setPreview("");
      setError("Please select an image smaller than 10 MB.");
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
  }

  async function handlePredict() {
    if (!file) {
      setError("Please upload an MRI image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

      setResult(data);
      await loadHistory();
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Cannot connect to the backend. Make sure FastAPI is running on port 8000."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleClearHistory() {
    const confirmed = window.confirm(
      "Clear all saved prediction history? This cannot be undone."
    );

    if (!confirmed) return;

    setClearingHistory(true);
    setHistoryError("");

    try {
      const response = await fetch(`${API_URL}/history`, {
        method: "DELETE",
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Could not clear prediction history.");
      }

      setHistory([]);
    } catch (err) {
      setHistoryError(
        err.message === "Failed to fetch"
          ? "Cannot connect to the backend. Make sure FastAPI is running."
          : err.message
      );
    } finally {
      setClearingHistory(false);
    }
  }

  const totalPredictions = history.length;

  const classDistribution = CLASS_NAMES.map((name) => ({
    name,
    count: history.filter((item) => item.predicted_class === name).length,
  }));

  const mostFrequentClass =
    totalPredictions === 0
      ? "—"
      : formatClassName(
          [...classDistribution].sort((a, b) => b.count - a.count)[0].name
        );

  const averageScore =
    totalPredictions === 0
      ? 0
      : history.reduce(
          (sum, item) => sum + Number(item.confidence || 0),
          0
        ) / totalPredictions;

  const recentPredictions = [...history]
    .sort((a, b) => Number(b.id) - Number(a.id))
    .slice(0, 7)
    .reverse();

  function renderScores(probabilities = {}) {
    return CLASS_NAMES.map((name) => {
      const score = Number(probabilities[name] ?? 0);

      return (
        <div className="score-item" key={name}>
          <div className="score-label">
            <span>{formatClassName(name)}</span>
            <span>{score.toFixed(2)}%</span>
          </div>

          <div className="score-track">
            <div
              className="score-fill"
              style={{
                width: `${Math.min(100, Math.max(0, score))}%`,
              }}
            />
          </div>
        </div>
      );
    });
  }

  return (
    <div className="app">
      <header className="navbar">
        <a className="brand" href="/">
          <span className="brand-icon">N</span>
          <span>
            NeuroScan<span className="brand-dot">.</span>
          </span>
        </a>

        <div className="nav-actions">
          <a className="nav-label" href="#performance-lab">
            MODEL PERFORMANCE
          </a>
          <span className="nav-label">AI RESEARCH PROJECT</span>
        </div>
      </header>

      <main className="main-content">
        <section className="hero">
          <span className="eyebrow">BRAIN MRI ANALYSIS</span>
          <h1>Brain Tumor Detection</h1>
          <p>
            Upload an MRI image to explore predictions from a deep-learning
            image classification model.
          </p>
        </section>

        <section className="workspace">
          <div className="panel upload-panel">
            <div className="panel-heading">
              <div>
                <span className="step-label">STEP 01</span>
                <h2>Upload MRI scan</h2>
              </div>
              <span className="file-tag">JPG · PNG · WEBP</span>
            </div>

            <label className="upload-area" htmlFor="mri-upload">
              {preview ? (
                <img
                  className="preview-image"
                  src={preview}
                  alt="Selected MRI scan"
                />
              ) : (
                <div className="upload-placeholder">
                  <span className="upload-icon">↑</span>
                  <strong>Choose an MRI image</strong>
                  <span>Click here to browse your computer</span>
                  <small>Maximum file size: 10 MB</small>
                </div>
              )}
            </label>

            <input
              id="mri-upload"
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={handleFileChange}
              hidden
            />

            {file && <p className="file-name">Selected: {file.name}</p>}

            <button
              className="predict-button"
              onClick={handlePredict}
              disabled={!file || loading}
            >
              {loading ? "Analyzing image..." : "Analyze MRI image  →"}
            </button>

            <p className="helper-text">
              The image is processed by your locally running model API.
            </p>
          </div>

          <div className="panel result-panel">
            <div className="panel-heading">
              <div>
                <span className="step-label">STEP 02</span>
                <h2>Prediction results</h2>
              </div>
              <span className="status-dot">● API</span>
            </div>

            {!result && !loading && (
              <div className="empty-state">
                <div className="scan-icon">◎</div>
                <h3>Results will appear here</h3>
                <p>
                  Upload an image and run the model to see its predicted class
                  and scores.
                </p>
              </div>
            )}

            {loading && (
              <div className="empty-state">
                <div className="loader" />
                <h3>Processing MRI image</h3>
                <p>Please wait while the model generates its prediction.</p>
              </div>
            )}

            {result && (
              <div className="results">
                <span className="result-caption">MODEL PREDICTION</span>

                <h3 className="prediction-title">
                  {formatClassName(result.predicted_class)}
                </h3>

                <p className="confidence">
                  {Number(result.confidence).toFixed(2)}%{" "}
                  <span>model score</span>
                </p>

                <div className="score-heading">
                  <strong>Class scores</strong>
                  <span>0–100%</span>
                </div>

                <div className="score-list">
                  {renderScores(result.probabilities)}
                </div>

                <section
                  className="gradcam-section"
                  aria-label="Grad-CAM explanation"
                >
                  <div className="gradcam-heading">
                    <div>
                      <span className="step-label">
                        MODEL EXPLAINABILITY
                      </span>
                      <h3>Grad-CAM visualization</h3>
                    </div>
                    <span className="gradcam-tag">AI explanation</span>
                  </div>

                  {result.gradcam_overlay ? (
                    <>
                      <img
                        className="gradcam-image"
                        src={result.gradcam_overlay}
                        alt="Grad-CAM heatmap overlay highlighting regions influencing the model prediction"
                      />

                      <div className="gradcam-legend">
                        <span>
                          <i className="legend-cool" /> Lower relative influence
                        </span>
                        <span>
                          <i className="legend-hot" /> Higher relative influence
                        </span>
                      </div>

                      <p className="gradcam-description">
                        {result.gradcam_notice ||
                          "Highlighted regions indicate areas that influenced the model score, not confirmed tumor boundaries."}
                      </p>
                    </>
                  ) : (
                    <p className="gradcam-description">
                      The prediction was generated, but Grad-CAM could not be
                      generated for this image. Check the backend terminal for
                      details.
                    </p>
                  )}
                </section>

                <p className="notice">
                  {result.notice ||
                    "Educational prototype only; not a medical diagnosis."}
                </p>
              </div>
            )}
          </div>
        </section>

        {error && <div className="error-message">{error}</div>}

        <section className="analytics-section">
          <div className="analytics-header">
            <div>
              <span className="step-label">MODEL OVERVIEW</span>
              <h2>Dashboard Analytics</h2>
              <p>Insights calculated from your saved MRI predictions.</p>
            </div>

            <span className="analytics-live">
              <span className="live-dot" />
              Based on saved records
            </span>
          </div>

          <div className="analytics-stats">
            <article className="analytics-card">
              <div className="analytics-card-top">
                <span>Total predictions</span>
                <span className="analytics-icon purple">Σ</span>
              </div>
              <strong className="analytics-number">{totalPredictions}</strong>
              <small>Saved prediction records</small>
            </article>

            <article className="analytics-card">
              <div className="analytics-card-top">
                <span>Most frequent class</span>
                <span className="analytics-icon blue">◎</span>
              </div>
              <strong className="analytics-number class-stat">
                {mostFrequentClass}
              </strong>
              <small>Based on prediction count</small>
            </article>

            <article className="analytics-card">
              <div className="analytics-card-top">
                <span>Average model score</span>
                <span className="analytics-icon green">%</span>
              </div>
              <strong className="analytics-number">
                {totalPredictions ? `${averageScore.toFixed(2)}%` : "—"}
              </strong>
              <small>Mean score across saved predictions</small>
            </article>
          </div>

          <div className="analytics-charts analytics-visual-charts">
            <article className="analytics-chart-card">
              <div className="analytics-chart-heading">
                <div>
                  <h3>Class Mix</h3>
                  <p>Share of saved predictions by class</p>
                </div>
                <span className="chart-tag">DONUT CHART</span>
              </div>

              {totalPredictions === 0 ? (
                <div className="chart-empty">
                  Make a prediction to see the class mix.
                </div>
              ) : (
                <div className="donut-layout">
                  <div
                    className="donut-chart"
                    role="img"
                    aria-label={classDistribution
                      .map(
                        (item) =>
                          `${formatClassName(item.name)} ${item.count}`
                      )
                      .join(", ")}
                    style={{
                      background: `conic-gradient(${(() => {
                        let angle = 0;
                        const colors = {
                          glioma: "#8b76eb",
                          meningioma: "#5b6be8",
                          notumor: "#36a984",
                          pituitary: "#e6a24b",
                        };

                        return classDistribution
                          .map((item) => {
                            const start = angle;
                            angle +=
                              (item.count / totalPredictions) * 360;

                            return `${colors[item.name]} ${start}deg ${angle}deg`;
                          })
                          .join(", ");
                      })()})`,
                    }}
                  >
                    <div className="donut-hole">
                      <strong>{totalPredictions}</strong>
                      <span>predictions</span>
                    </div>
                  </div>

                  <div className="donut-legend">
                    {classDistribution.map((item) => (
                      <div className="donut-legend-item" key={item.name}>
                        <span className={`legend-dot fill-${item.name}`} />
                        <span className="legend-name">
                          {formatClassName(item.name)}
                        </span>
                        <strong>{item.count}</strong>
                        <small>
                          {(
                            (item.count / totalPredictions) *
                            100
                          ).toFixed(1)}
                          %
                        </small>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <p className="chart-footnote">
                Distribution reflects saved model predictions, not confirmed
                diagnoses.
              </p>
            </article>

            <article className="analytics-chart-card">
              <div className="analytics-chart-heading">
                <div>
                  <h3>Daily Prediction Trend</h3>
                  <p>Saved predictions by calendar day</p>
                </div>
                <span className="chart-tag">DAILY TREND</span>
              </div>

              {totalPredictions === 0 ? (
                <div className="chart-empty">
                  Make a prediction to start building your trend.
                </div>
              ) : (
                <div className="trend-chart">
                  {(() => {
                    const byDay = new Map();

                    history.forEach((item) => {
                      const date = new Date(item.created_at);
                      if (Number.isNaN(date.getTime())) return;

                      const key = `${date.getFullYear()}-${String(
                        date.getMonth() + 1
                      ).padStart(2, "0")}-${String(
                        date.getDate()
                      ).padStart(2, "0")}`;

                      byDay.set(key, (byDay.get(key) || 0) + 1);
                    });

                    const days = [...byDay.entries()]
                      .sort((a, b) => a[0].localeCompare(b[0]))
                      .slice(-7);

                    const maxCount = Math.max(
                      ...days.map((day) => day[1]),
                      1
                    );

                    return days.map(([day, count]) => (
                      <div
                        className="trend-column"
                        key={day}
                        title={`${day}: ${count} predictions`}
                      >
                        <span className="trend-count">{count}</span>
                        <div className="trend-bar-area">
                          <div
                            className="trend-bar"
                            style={{
                              height: `${Math.max(
                                8,
                                (count / maxCount) * 100
                              )}%`,
                            }}
                          />
                        </div>
                        <span className="trend-date">
                          {new Date(`${day}T12:00:00`).toLocaleDateString(
                            undefined,
                            {
                              month: "short",
                              day: "numeric",
                            }
                          )}
                        </span>
                      </div>
                    ));
                  })()}
                </div>
              )}

              <p className="chart-footnote">
                Uses the saved timestamp for each prediction. Only days with
                saved records are shown.
              </p>
            </article>
          </div>

          <div className="analytics-charts">
            <article className="analytics-chart-card">
              <div className="analytics-chart-heading">
                <div>
                  <h3>Class Distribution</h3>
                  <p>Number of predictions per class</p>
                </div>
                <span className="chart-tag">DISTRIBUTION</span>
              </div>

              {totalPredictions === 0 ? (
                <div className="chart-empty">
                  Make a prediction to see class distribution.
                </div>
              ) : (
                <div className="distribution-list">
                  {classDistribution.map((item) => {
                    const percentage =
                      (item.count / totalPredictions) * 100;

                    return (
                      <div className="distribution-item" key={item.name}>
                        <div className="distribution-label">
                          <span>{formatClassName(item.name)}</span>
                          <span>
                            {item.count}{" "}
                            <small>({percentage.toFixed(1)}%)</small>
                          </span>
                        </div>

                        <div
                          className="distribution-track"
                          role="img"
                          aria-label={`${formatClassName(item.name)}: ${item.count} predictions`}
                        >
                          <div
                            className={`distribution-fill fill-${item.name}`}
                            style={{ width: `${percentage}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              <p className="chart-footnote">
                This chart shows saved predictions, not dataset class balance
                or confirmed diagnoses.
              </p>
            </article>

            <article className="analytics-chart-card">
              <div className="analytics-chart-heading">
                <div>
                  <h3>Recent Model Scores</h3>
                  <p>Latest {recentPredictions.length} saved predictions</p>
                </div>
                <span className="chart-tag">MODEL SCORES</span>
              </div>

              {recentPredictions.length === 0 ? (
                <div className="chart-empty">
                  Saved model scores will appear here.
                </div>
              ) : (
                <div className="recent-score-chart">
                  {recentPredictions.map((item) => {
                    const score = Math.max(
                      0,
                      Math.min(100, Number(item.confidence || 0))
                    );

                    return (
                      <div className="recent-score-item" key={item.id}>
                        <div className="recent-score-label">
                          <span title={item.filename}>
                            #{item.id} · {item.filename}
                          </span>
                          <strong>{score.toFixed(2)}%</strong>
                        </div>

                        <div className="distribution-track">
                          <div
                            className="recent-score-fill"
                            style={{ width: `${score}%` }}
                          />
                        </div>

                        <span className="recent-score-class">
                          {formatClassName(item.predicted_class)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}

              <p className="chart-footnote">
                Model scores are not measures of clinical accuracy.
              </p>
            </article>
          </div>
        </section>

        <section className="panel history-panel">
          <div className="panel-heading history-heading">
            <div>
              <span className="step-label">SAVED RECORDS</span>
              <h2>Prediction History</h2>
              <p>Review predictions saved by your local backend.</p>
            </div>

            <div className="history-actions">
              <button
                className="history-button"
                onClick={loadHistory}
                disabled={historyLoading}
              >
                {historyLoading ? "Refreshing..." : "↻ Refresh"}
              </button>

              <button
                className="history-button clear-button"
                onClick={handleClearHistory}
                disabled={
                  clearingHistory ||
                  historyLoading ||
                  totalPredictions === 0
                }
              >
                {clearingHistory ? "Clearing..." : "Clear History"}
              </button>
            </div>
          </div>

          {historyError && (
            <div className="error-message">{historyError}</div>
          )}

          <p className="history-count">
            {totalPredictions}{" "}
            {totalPredictions === 1 ? "prediction" : "predictions"} saved
          </p>

          {historyLoading && history.length === 0 ? (
            <div className="chart-empty">Loading saved records...</div>
          ) : totalPredictions === 0 ? (
            <div className="history-empty">
              <div className="scan-icon">◎</div>
              <h3>No saved predictions yet</h3>
              <p>
                Upload an MRI image and run a prediction to create your first
                record.
              </p>
            </div>
          ) : (
            <div className="history-table-wrapper">
              <table className="history-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Filename</th>
                    <th>Predicted class</th>
                    <th>Model score</th>
                    <th>Date and time</th>
                  </tr>
                </thead>

                <tbody>
                  {history.map((item) => (
                    <tr key={item.id}>
                      <td>#{item.id}</td>
                      <td className="history-filename">{item.filename}</td>
                      <td>
                        <span className="history-class">
                          {formatClassName(item.predicted_class)}
                        </span>
                      </td>
                      <td>{Number(item.confidence).toFixed(2)}%</td>
                      <td>{formatDate(item.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {/* Model Performance Lab */}
        <section className="performance-lab" id="performance-lab">
          <div className="performance-header">
            <div>
              <span className="step-label">EVALUATION REPORT</span>
              <h2>Model Performance Lab</h2>
              <p>
                Results from the existing EfficientNetB0 test-set evaluation.
                No model was retrained.
              </p>
            </div>
            <span className="performance-badge">1,584 test images</span>
          </div>

          <div className="performance-kpis">
            <article className="performance-kpi">
              <span>Test accuracy</span>
              <strong>89.58%</strong>
              <small>Overall correct predictions</small>
            </article>
            <article className="performance-kpi">
              <span>Macro F1-score</span>
              <strong>89.21%</strong>
              <small>Equal weight per class</small>
            </article>
            <article className="performance-kpi">
              <span>Weighted F1-score</span>
              <strong>89.28%</strong>
              <small>Weighted by class support</small>
            </article>
            <article className="performance-kpi">
              <span>Test set</span>
              <strong>1,584</strong>
              <small>Held-out images</small>
            </article>
          </div>

          <div className="performance-grid">
            <article className="performance-card">
              <div className="performance-card-heading">
                <div>
                  <h3>Confusion Matrix</h3>
                  <p>Rows = actual class · columns = predicted class</p>
                </div>
                <span className="chart-tag">TEST SET</span>
              </div>

              <div className="matrix-wrap">
                <table
                  className="confusion-matrix"
                  aria-label="Confusion matrix for the existing EfficientNetB0 model"
                >
                  <thead>
                    <tr>
                      <th>Actual ↓ / Pred. →</th>
                      <th>Glioma</th>
                      <th>Meningioma</th>
                      <th>No tumor</th>
                      <th>Pituitary</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <th>Glioma</th>
                      <td className="matrix-diagonal">286</td>
                      <td className="matrix-error">60</td>
                      <td className="matrix-error">20</td>
                      <td className="matrix-error">20</td>
                    </tr>
                    <tr>
                      <th>Meningioma</th>
                      <td className="matrix-error">20</td>
                      <td className="matrix-diagonal">336</td>
                      <td className="matrix-error">13</td>
                      <td className="matrix-error">29</td>
                    </tr>
                    <tr>
                      <th>No tumor</th>
                      <td>0</td>
                      <td>0</td>
                      <td className="matrix-diagonal">400</td>
                      <td>0</td>
                    </tr>
                    <tr>
                      <th>Pituitary</th>
                      <td className="matrix-error">1</td>
                      <td className="matrix-error">2</td>
                      <td>0</td>
                      <td className="matrix-diagonal">397</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <p className="performance-note">
                The diagonal cells are correct predictions. Off-diagonal
                cells show where the model confused one class with another.
              </p>
            </article>

            <article className="performance-card">
              <div className="performance-card-heading">
                <div>
                  <h3>Per-class metrics</h3>
                  <p>Precision, recall, and F1-score</p>
                </div>
                <span className="chart-tag">CLASS REPORT</span>
              </div>

              <div className="metric-table-wrap">
                <table className="metric-table">
                  <thead>
                    <tr>
                      <th>Class</th>
                      <th>Precision</th>
                      <th>Recall</th>
                      <th>F1</th>
                      <th>Support</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <th>Glioma</th>
                      <td>93.16%</td>
                      <td className="metric-attention">74.09%</td>
                      <td>82.54%</td>
                      <td>386</td>
                    </tr>
                    <tr>
                      <th>Meningioma</th>
                      <td>84.42%</td>
                      <td>84.42%</td>
                      <td>84.42%</td>
                      <td>398</td>
                    </tr>
                    <tr>
                      <th>No tumor</th>
                      <td>92.38%</td>
                      <td>100.00%</td>
                      <td>96.04%</td>
                      <td>400</td>
                    </tr>
                    <tr>
                      <th>Pituitary</th>
                      <td>89.01%</td>
                      <td>99.25%</td>
                      <td>93.85%</td>
                      <td>400</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <p className="performance-note">
                <strong>Review focus:</strong> Glioma recall is lower than
                the other classes in this test set. This is an evaluation
                finding, not a diagnosis or a claim about an individual scan.
              </p>
            </article>
          </div>

          <article className="performance-card class-recall-card">
            <div className="performance-card-heading">
              <div>
                <h3>Recall by class</h3>
                <p>
                  Share of actual examples correctly identified in the test set
                </p>
              </div>
            </div>

            {[
              { name: "Glioma", value: 74.09 },
              { name: "Meningioma", value: 84.42 },
              { name: "No tumor", value: 100 },
              { name: "Pituitary", value: 99.25 },
            ].map((item) => (
              <div className="recall-row" key={item.name}>
                <div className="recall-label">
                  <span>{item.name}</span>
                  <strong>{item.value.toFixed(2)}%</strong>
                </div>
                <div className="recall-track">
                  <div
                    className={`recall-fill ${
                      item.name === "Glioma" ? "recall-warning" : ""
                    }`}
                    style={{ width: `${item.value}%` }}
                  />
                </div>
              </div>
            ))}

            <p className="performance-note">
              Metrics are from the saved evaluation report. They do not
              guarantee performance on new images or in clinical settings.
            </p>
          </article>
        </section>

        <section className="disclaimer">
          <strong>Important disclaimer</strong>
          <p>
            This is an educational AI prototype, not a medical diagnostic
            system. Predictions may be incorrect and must not be used to make
            medical decisions. Consult a qualified medical professional for
            medical interpretation.
          </p>
        </section>
      </main>

      <footer>NeuroScan · Educational Deep Learning Project</footer>
    </div>
  );
}

export default App;
