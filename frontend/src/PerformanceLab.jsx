import "./PerformanceLab.css";

const metrics = [
  { label: "Accuracy", value: "89.58%" },
  { label: "Macro F1 Score", value: "89.21%" },
  { label: "Weighted F1 Score", value: "89.28%" },
  { label: "Test Images", value: "1,584" },
];

const classes = [
  { name: "Glioma", precision: 93.16, recall: 74.09, f1: 82.54, support: 386 },
  { name: "Meningioma", precision: 84.42, recall: 84.42, f1: 84.42, support: 398 },
  { name: "No Tumor", precision: 92.38, recall: 100.0, f1: 96.04, support: 400 },
  { name: "Pituitary", precision: 89.01, recall: 99.25, f1: 93.85, support: 400 },
];

const matrix = [
  [286, 60, 20, 20],
  [20, 336, 13, 29],
  [0, 0, 400, 0],
  [1, 2, 0, 397],
];

const labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"];

export default function PerformanceLab() {
  const maxValue = 400;

  return (
    <section className="performance-lab" id="performance-lab">
      <div className="pl-heading">
        <div>
          <p className="pl-eyebrow">MODEL EVALUATION</p>
          <h2>Model Performance Lab</h2>
          <p className="pl-subtitle">
            Evaluate the performance of your trained EfficientNetB0 model.
          </p>
        </div>
        <span className="pl-status">Test Results</span>
      </div>

      <div className="pl-metrics">
        {metrics.map((metric) => (
          <article className="pl-card pl-kpi" key={metric.label}>
            <p>{metric.label}</p>
            <h3>{metric.value}</h3>
          </article>
        ))}
      </div>

      <article className="pl-card pl-section">
        <div className="pl-section-heading">
          <div>
            <h3>Confusion Matrix</h3>
            <p>Rows represent actual classes; columns represent predictions.</p>
          </div>
        </div>

        <div className="pl-table-wrap">
          <table className="pl-matrix">
            <thead>
              <tr>
                <th>Actual / Predicted</th>
                {labels.map((label) => <th key={label}>{label}</th>)}
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, i) => (
                <tr key={labels[i]}>
                  <th>{labels[i]}</th>
                  {row.map((value, j) => (
                    <td
                      key={j}
                      className={i === j ? "pl-correct" : ""}
                      title={`${value} images`}
                    >
                      {value}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="pl-note">
          Diagonal cells show correct predictions. Off-diagonal cells show
          misclassifications.
        </p>
      </article>

      <article className="pl-card pl-section">
        <div className="pl-section-heading">
          <div>
            <h3>Per-Class Metrics</h3>
            <p>Precision, recall, F1 score and number of test images.</p>
          </div>
        </div>

        <div className="pl-table-wrap">
          <table className="pl-metrics-table">
            <thead>
              <tr>
                <th>Class</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
                <th>Support</th>
              </tr>
            </thead>
            <tbody>
              {classes.map((item) => (
                <tr key={item.name}>
                  <td>{item.name}</td>
                  <td>{item.precision.toFixed(2)}%</td>
                  <td>{item.recall.toFixed(2)}%</td>
                  <td>{item.f1.toFixed(2)}%</td>
                  <td>{item.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </article>

      <article className="pl-card pl-section">
        <div className="pl-section-heading">
          <div>
            <h3>Recall by Class</h3>
            <p>Percentage of actual images correctly identified per class.</p>
          </div>
        </div>

        <div className="pl-bars">
          {classes.map((item) => (
            <div className="pl-bar-row" key={item.name}>
              <div className="pl-bar-label">
                <span>{item.name}</span>
                <strong>{item.recall.toFixed(2)}%</strong>
              </div>
              <div
                className="pl-bar-track"
                role="progressbar"
                aria-label={`${item.name} recall`}
                aria-valuenow={item.recall}
                aria-valuemin={0}
                aria-valuemax={100}
              >
                <div
                  className="pl-bar-fill"
                  style={{ width: `${item.recall}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </article>

      <p className="pl-disclaimer">
        These are saved evaluation results for this dataset, not a guarantee
        of clinical accuracy. This project is for educational purposes and
        must not be used as a substitute for professional medical diagnosis.
      </p>
    </section>
  );
}
