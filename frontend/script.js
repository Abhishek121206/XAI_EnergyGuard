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

async function loadData() {

    const data = await fetchJSON(
        "/api/data"
    );

    if (!Array.isArray(data) || data.length === 0) {

        $("totalReadings").textContent = "0";

        return;
    }


    // -------------------------------------------------
    // LATEST READING
    // -------------------------------------------------

    const latest = data[data.length - 1];


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


    $("totalReadings").textContent =
        data.length;


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
        `;

        table.appendChild(tr);
    });
}


// =====================================================
// LOAD ANOMALIES
// =====================================================

async function loadAnomalies() {

    const anomalies = await fetchJSON(
        "/api/anomalies"
    );


    $("anomalyCount").textContent =
        anomalies.length;


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
                ${latest.waste_type ?? "ANOMALY"}
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
            "/api/explanations"
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
                ${item.waste_type}
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
                }
            }
        );
    });
}


// =====================================================
// LOAD DASHBOARD
// =====================================================

async function loadDashboard() {

    await loadHealth();

    try {

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


// Refresh data every 5 seconds

setInterval(
    loadDashboard,
    5000
);