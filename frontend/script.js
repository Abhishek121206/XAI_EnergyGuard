// =====================================================
// CONFIGURATION
// =====================================================

const API_BASE = "http://127.0.0.1:5000";


// =====================================================
// DOM HELPERS
// =====================================================

const $ = (id) => document.getElementById(id);


// =====================================================
// API
// =====================================================

async function fetchJSON(endpoint) {

    const response = await fetch(
        `${API_BASE}${endpoint}`
    );

    if (!response.ok) {

        throw new Error(
            `API request failed: ${response.status}`
        );
    }

    return response.json();
}


// =====================================================
// FORMAT TIME
// =====================================================

function formatTimestamp(timestamp) {

    if (!timestamp) {
        return "--";
    }

    const date = new Date(
        Number(timestamp) * 1000
    );

    return date.toLocaleString();
}


// =====================================================
// STATUS CLASS
// =====================================================

function statusClass(status) {

    if (
        status === "NORMAL" ||
        status === "NO_LOAD"
    ) {

        return "success";
    }

    return "danger";
}


// =====================================================
// MODEL CLASSES
// =====================================================

const CLASS_LABELS = {
    NORMAL: "Normal",
    IE: "Idle equipment",
    OC: "Overconsumption",
    UP: "Unusual pattern"
};

function classLabel(code) {

    return CLASS_LABELS[code] ?? code ?? "--";
}

function percent(value) {

    return `${(Number(value ?? 0) * 100).toFixed(1)}%`;
}


// =====================================================
// LOAD HEALTH
// =====================================================

async function loadHealth() {

    try {

        const data = await fetchJSON(
            "/api/health"
        );

        $("backendStatus").textContent =
            data.status === "healthy"
                ? "Connected"
                : "Unavailable";

    } catch (error) {

        $("backendStatus").textContent =
            "Unavailable";

        console.error(error);
    }
}


// =====================================================
// LOAD ALL DATA
// =====================================================

async function loadLivePrediction() {

    // Written per MQTT message by data_collector.py;
    // missing when the collector or model isn't running.

    try {

        return await fetchJSON("/api/live");

    } catch (error) {

        return null;
    }
}


async function loadData() {

    const data = await fetchJSON(
        "/api/data?limit=10"
    );

    if (!Array.isArray(data) || data.length === 0) {

        return;
    }


    // -------------------------------------------------
    // LATEST READING (live prediction if newer)
    // -------------------------------------------------

    const live = await loadLivePrediction();

    const latestBatch = data[data.length - 1];

    const latest =
        live && live.timestamp >= latestBatch.timestamp
            ? live
            : latestBatch;


    $("currentPrediction").textContent =
        classLabel(latest.waste_type);

    $("currentPrediction").className =
        `status-text ${statusClass(
            latest.waste_type
        )}`;

    $("currentPredictionConfidence").textContent =
        latest.waste_confidence === undefined
            ? ""
            : percent(latest.waste_confidence);


    $("currentPower").textContent =
        Number(latest.power_W ?? 0).toFixed(3);


    $("currentPresence").textContent =
        latest.presence_binary === 1
            ? "PRESENT"
            : "ABSENT";


    $("presenceDetail").textContent =
        latest.presence_binary === 1
            ? "Detected"
            : "No presence";


    $("temperature").textContent =
        latest.temperature_C ?? "--";


    $("humidity").textContent =
        latest.humidity_percent ?? "--";


    $("current").textContent =
        Number(
            latest.current_A ?? 0
        ).toFixed(3);


    $("voltage").textContent =
        Number(
            latest.voltage_V ?? 0
        ).toFixed(2);


    $("distance").textContent =
        latest.distance_cm === null ||
        latest.distance_cm === undefined
            ? "--"
            : Number(
                latest.distance_cm
            ).toFixed(1);


    $("currentStatus").textContent =
        latest.status ?? "--";


    $("currentStatus").className =
        `status-text ${statusClass(
            latest.status
        )}`;


    $("latestTimestamp").textContent =
        formatTimestamp(
            latest.timestamp
        );


    // -------------------------------------------------
    // RECENT TABLE
    // -------------------------------------------------

    const recent = [
        ...data
    ]
    .slice(-10)
    .reverse();


    const table = $("readingsTable");

    table.innerHTML = "";


    recent.forEach((row) => {

        const tr = document.createElement("tr");

        tr.innerHTML = `
            <td>
                ${formatTimestamp(row.timestamp)}
            </td>

            <td>
                ${Number(row.power_W ?? 0).toFixed(3)} W
            </td>

            <td>
                ${Number(row.current_A ?? 0).toFixed(3)} A
            </td>

            <td>
                ${
                    row.presence_binary === 1
                        ? "PRESENT"
                        : "ABSENT"
                }
            </td>

            <td>
                ${row.temperature_C ?? "--"} °C
            </td>

            <td>
                <span class="table-status ${statusClass(
                    row.status
                )}">
                    ${row.status ?? "--"}
                </span>
            </td>

            <td>
                <span class="table-status ${statusClass(
                    row.waste_type
                )}">
                    ${row.waste_type ?? "--"}
                </span>
            </td>
        `;

        table.appendChild(tr);
    });
}


// =====================================================
// LOAD SUMMARY
// =====================================================

async function loadSummary() {

    const summary = await fetchJSON(
        "/api/summary"
    );

    $("totalReadings").textContent =
        summary.total_readings;

    $("anomalyCount").textContent =
        summary.anomaly_count;

    $("anomalyDetail").textContent =
        `waste events (${(
            100 * summary.anomaly_count /
            Math.max(summary.total_readings, 1)
        ).toFixed(0)}% of readings)`;

    $("classChips").innerHTML =
        Object.entries(summary.class_counts)
            .map(([code, count]) => `
                <div class="chip ${statusClass(code)}">
                    <strong>${count}</strong>
                    ${code} · ${classLabel(code)}
                </div>
            `)
            .join("");
}


// =====================================================
// LOAD ANOMALIES
// =====================================================

async function loadAnomalies() {

    const anomalies = await fetchJSON(
        "/api/anomalies?limit=200"
    );


    const table = $("anomaliesTable");

    table.innerHTML = "";


    if (anomalies.length === 0) {

        table.innerHTML = `
            <tr>
                <td colspan="6">
                    No anomalies detected
                </td>
            </tr>
        `;

        return;
    }


    const recent = [
        ...anomalies
    ]
    .reverse();


    recent.forEach((row) => {

        const tr = document.createElement("tr");

        tr.innerHTML = `
            <td>
                ${formatTimestamp(row.timestamp)}
            </td>

            <td>
                ${Number(
                    row.power_W ?? 0
                ).toFixed(3)} W
            </td>

            <td>
                ${
                    row.presence_binary === 1
                        ? "PRESENT"
                        : "ABSENT"
                }
            </td>

            <td>
                ${row.waste_type ?? "UNKNOWN"}
            </td>

            <td>
                ${(
                    Number(
                        row.waste_confidence ?? 0
                    ) * 100
                ).toFixed(1)}%
            </td>

            <td>
                ${Number(
                    row.anomaly_score ?? 0
                ).toFixed(4)}
            </td>
        `;

        table.appendChild(tr);
    });


    // -------------------------------------------------
    // LATEST ANOMALY CARD
    // -------------------------------------------------

    const latest =
        anomalies[anomalies.length - 1];


    $("latestAnomaly").innerHTML = `
        <div class="alert-event">

            <div class="alert-type">
                ${classLabel(latest.waste_type)}
            </div>

            <div class="alert-power">
                ${Number(
                    latest.power_W ?? 0
                ).toFixed(3)} W
            </div>

            <div class="alert-meta">
                Time:
                ${formatTimestamp(
                    latest.timestamp
                )}
                <br>

                Presence:
                ${
                    latest.presence_binary === 1
                        ? "PRESENT"
                        : "ABSENT"
                }
                <br>

                Confidence:
                ${(
                    Number(
                        latest.waste_confidence ?? 0
                    ) * 100
                ).toFixed(1)}%
            </div>

        </div>
    `;
}


// =====================================================
// LOAD EXPLANATIONS
// =====================================================

async function loadExplanations() {

    const explanations =
        await fetchJSON(
            "/api/explanations?limit=20"
        );


    const container =
        $("explanationsContainer");


    container.innerHTML = "";


    if (
        !Array.isArray(explanations) ||
        explanations.length === 0
    ) {

        container.innerHTML = `
            <div class="panel">
                <div class="empty-state">
                    No explanations available
                </div>
            </div>
        `;

        return;
    }


    // Most recent first

    const recent = [
        ...explanations
    ]
    .reverse();


    recent.forEach((item) => {

        const card =
            document.createElement("div");

        card.className =
            "explanation-card";


        card.innerHTML = `

            <div class="explanation-role">
                FACILITY MANAGER
            </div>

            <h4>
                ${classLabel(item.waste_type)}
                <small class="explanation-time">
                    ${formatTimestamp(item.timestamp)}
                </small>
            </h4>

            <p class="explanation-text">
                ${item.facility_manager}
            </p>

            <span class="explanation-type">
                Confidence:
                ${(
                    Number(
                        item.waste_confidence ?? 0
                    ) * 100
                ).toFixed(1)}%
            </span>


            <div
                class="explanation-role"
                style="margin-top: 24px;"
            >
                STUDENT
            </div>

            <p class="explanation-text">
                ${item.student}
            </p>


            <div
                class="explanation-role"
                style="margin-top: 24px;"
            >
                ADMINISTRATOR
            </div>

            <p class="explanation-text">
                ${item.administrator}
            </p>


            <div
                class="explanation-role"
                style="margin-top: 24px;"
            >
                COUNTERFACTUAL
                (${item.counterfactual_method ?? "--"})
            </div>

            <p class="explanation-text">
                ${item.counterfactual ?? "--"}
            </p>

        `;


        container.appendChild(card);
    });
}


// =====================================================
// NAVIGATION
// =====================================================

function setupNavigation() {

    const buttons =
        document.querySelectorAll(
            ".nav-item"
        );


    const sections =
        document.querySelectorAll(
            ".page-section"
        );


    const pageTitle =
        $("pageTitle");


    buttons.forEach((button) => {

        button.addEventListener(
            "click",
            () => {

                const sectionId =
                    button.dataset.section;


                buttons.forEach(
                    (item) => {
                        item.classList.remove(
                            "active"
                        );
                    }
                );


                sections.forEach(
                    (section) => {
                        section.classList.remove(
                            "active"
                        );
                    }
                );


                button.classList.add(
                    "active"
                );


                const section =
                    document.getElementById(
                        sectionId
                    );


                if (section) {

                    section.classList.add(
                        "active"
                    );
                }


                if (
                    sectionId ===
                    "overview"
                ) {

                    pageTitle.textContent =
                        "Energy Overview";

                } else if (
                    sectionId ===
                    "anomalies"
                ) {

                    pageTitle.textContent =
                        "Detected Anomalies";

                } else if (
                    sectionId ===
                    "explanations"
                ) {

                    pageTitle.textContent =
                        "AI Explanations";

                } else if (
                    sectionId ===
                    "model"
                ) {

                    pageTitle.textContent =
                        "Model Performance";
                }
            }
        );
    });
}


// =====================================================
// LOAD MODEL METRICS
// =====================================================

async function loadModel() {

    let metrics;

    try {

        metrics = await fetchJSON(
            "/api/model/metrics"
        );

    } catch (error) {

        $("modelInfo").textContent =
            "No trained model. Run python train.py.";

        return;
    }

    const test = metrics.test;
    const splits = metrics.splits;

    $("modelInfo").textContent =
        `Trained ${new Date(
            metrics.trained_at
        ).toLocaleString()} on ${metrics.rows} real readings ` +
        `from ${metrics.data_file}. Tested on the most recent ` +
        `${splits.test.rows} readings ` +
        `(${new Date(splits.test.start).toLocaleDateString()} – ` +
        `${new Date(splits.test.end).toLocaleDateString()}), ` +
        `which the model never saw while training.`;


    // -------------------------------------------------
    // SCORE CARDS
    // -------------------------------------------------

    const scores = [
        ["XGBOOST MACRO F1", test.xgboost.macro_f1, "selected model"],
        ["XGBOOST ACCURACY", test.xgboost.accuracy, "test set"],
        ["FIRMWARE RULE F1", test.firmware_rule_baseline.macro_f1, "baseline (no UP)"],
        ["LOGISTIC REG. F1", test.logistic_regression.macro_f1, "baseline"]
    ];

    $("modelScores").innerHTML = scores
        .map(([label, value, sub]) => `
            <div class="stat-card">
                <div class="stat-label">${label}</div>
                <div class="stat-value">
                    ${Number(value).toFixed(3)}
                </div>
                <div class="stat-sub">${sub}</div>
            </div>
        `)
        .join("");


    // -------------------------------------------------
    // PER-CLASS TABLE
    // -------------------------------------------------

    $("perClassTable").innerHTML = Object.entries(
        test.xgboost.per_class
    )
        .map(([code, values]) => `
            <tr>
                <td>${code} · ${classLabel(code)}</td>
                <td>${values.precision.toFixed(3)}</td>
                <td>${values.recall.toFixed(3)}</td>
                <td>${values.f1.toFixed(3)}</td>
                <td>${values.support}</td>
            </tr>
        `)
        .join("");


    // -------------------------------------------------
    // FEATURE IMPORTANCE BARS
    // -------------------------------------------------

    const importance = Object.entries(
        metrics.feature_importance
    );

    const maxImportance = Math.max(
        ...importance.map(([, value]) => value),
        1e-9
    );

    $("featureImportance").innerHTML = importance
        .map(([feature, value]) => `
            <div class="bar-row">
                <span class="bar-label">${feature}</span>
                <div class="bar-track">
                    <div
                        class="bar-fill"
                        style="width: ${(
                            100 * value / maxImportance
                        ).toFixed(1)}%"
                    ></div>
                </div>
                <span class="bar-value">
                    ${value.toFixed(3)}
                </span>
            </div>
        `)
        .join("");


    $("labelNote").textContent =
        metrics.label_note;
}


// =====================================================
// LOAD DASHBOARD
// =====================================================

async function loadDashboard() {

    await loadHealth();

    try {

        await loadSummary();

        await loadData();

        await loadAnomalies();

        await loadExplanations();

    } catch (error) {

        console.error(
            "Dashboard loading error:",
            error
        );
    }
}


// =====================================================
// START
// =====================================================

setupNavigation();

loadDashboard();

// Metrics only change when train.py is re-run
loadModel();


// Refresh data every 5 seconds

setInterval(
    loadDashboard,
    5000
);