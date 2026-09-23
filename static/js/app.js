let rulChart = null;
let failureChart = null;
let statusChart = null;

const engineSelect = document.getElementById("engine");
const analyseButton = document.getElementById("analyse");
const dashboard = document.getElementById("dashboard");
const loading = document.getElementById("loading");
const errorBox = document.getElementById("error");

async function getJSON(url) {

    const response = await fetch(url, {
        method: "GET",
        headers: {
            "Accept": "application/json"
        },
        cache: "no-store"
    });

    const text = await response.text();

    if (!text || !text.trim()) {
        throw new Error(
            `Server returned an empty response (${response.status}).`
        );
    }

    let result;

    try {
        result = JSON.parse(text);
    } catch (error) {
        console.error("Non-JSON server response:", text);

        throw new Error(
            `Server returned invalid JSON (${response.status}).`
        );
    }

    if (!response.ok) {
        throw new Error(
            result.error ||
            `Server error (${response.status}).`
        );
    }

    return result;
}


async function loadEngines() {

    try {

        const result =
            await getJSON("/api/engines");

        if (!result.ok) {
            throw new Error(
                result.error ||
                "Could not load engines."
            );
        }

        engineSelect.innerHTML =
            result.engines
                .map(
                    id =>
                        `<option value="${id}">
                            Engine ${id}
                        </option>`
                )
                .join("");

        if (result.engines.length > 0) {

            await analyseEngine(
                result.engines[0]
            );

        }

    } catch (error) {

        showError(
            "Unable to load engine list: " +
            error.message
        );

    }
}


async function analyseEngine(id) {

    if (!id) {
        return;
    }

    loading.classList.remove("hidden");

    errorBox.classList.add("hidden");

    dashboard.classList.add("hidden");

    analyseButton.disabled = true;

    analyseButton.textContent =
        "Analysing...";


    try {

        const result =
            await getJSON(
                `/api/engine/${id}`
            );

        if (!result.ok) {

            throw new Error(
                result.error ||
                `Could not analyse Engine ${id}.`
            );

        }

        if (!result.data) {

            throw new Error(
                `Engine ${id} returned no prediction data.`
            );

        }

        render(result.data);

    } catch (error) {

        console.error(
            `Engine ${id} analysis failed:`,
            error
        );

        showError(
            `Engine ${id}: ${error.message}`
        );

    } finally {

        loading.classList.add("hidden");

        analyseButton.disabled = false;

        analyseButton.innerHTML =
            'Analyse <span>→</span>';

    }
}


function showError(message) {

    dashboard.classList.add("hidden");

    errorBox.textContent = message;

    errorBox.classList.remove("hidden");
}


function render(data) {

    dashboard.classList.remove("hidden");

    const statusCard =
        document.getElementById("statusCard");

    statusCard.className =
        "status card status-" +
        data.status.toLowerCase();


    document.getElementById("status")
        .textContent = data.status;


    document.getElementById("recommendation")
        .textContent =
        data.recommendation;


    document.getElementById("engineId")
        .textContent =
        data.engine_id;


    document.getElementById("rul")
        .textContent =
        Number(data.predicted_rul)
            .toFixed(2);


    document.getElementById("failure")
        .textContent =
        Number(data.failure_probability)
            .toFixed(2);


    document.getElementById("risk")
        .style.width =
        Math.min(
            100,
            Number(data.failure_probability)
        ) + "%";


    document.getElementById("cycle")
        .textContent =
        data.latest_cycle;


    document.getElementById("window")
        .textContent =
        `Cycles ${data.window_start}–${data.window_end}`;


    document.getElementById("decisionStatus")
        .textContent =
        data.status;


    document.getElementById("decisionText")
        .textContent =
        data.recommendation;


    document.getElementById("observed")
        .textContent =
        data.observed_cycles;


    document.getElementById("inputWindow")
        .textContent =
        `${data.window_start}–${data.window_end}`;


    const decision =
        document.getElementById("decision");


    decision.className =
        "decision-box " +
        data.status.toLowerCase();


    if (
        !Array.isArray(data.trajectory) ||
        data.trajectory.length === 0
    ) {

        throw new Error(
            "No trajectory data was returned."
        );

    }


    renderCharts(
        data.trajectory
    );
}


function baseOptions() {

    return {

        responsive: true,

        maintainAspectRatio: false,

        animation: {
            duration: 500
        },

        interaction: {
            mode: "index",
            intersect: false
        },

        plugins: {

            legend: {
                display: false
            },

            tooltip: {
                backgroundColor: "#152942",

                padding: 11,

                cornerRadius: 8
            }
        },

        scales: {

            x: {

                grid: {
                    display: false
                },

                ticks: {

                    color: "#8793a5",

                    maxTicksLimit: 10,

                    font: {
                        size: 10
                    }
                }
            },

            y: {

                grid: {
                    color: "#edf0f4"
                },

                ticks: {

                    color: "#8793a5",

                    font: {
                        size: 10
                    }
                }
            }
        }
    };
}


function renderCharts(rows) {

    const cycles =
        rows.map(
            row => row.cycle
        );

    const rul =
        rows.map(
            row => row.rul
        );

    const probability =
        rows.map(
            row =>
                row.failure_probability
        );


    const states =
        rows.map(
            row => {

                if (
                    row.status === "CRITICAL"
                ) {
                    return 3;
                }

                if (
                    row.status === "WARNING"
                ) {
                    return 2;
                }

                if (
                    row.status === "MONITOR"
                ) {
                    return 1;
                }

                return 0;
            }
        );


    if (rulChart) {
        rulChart.destroy();
    }

    if (failureChart) {
        failureChart.destroy();
    }

    if (statusChart) {
        statusChart.destroy();
    }


    rulChart =
        new Chart(
            document.getElementById(
                "rulChart"
            ),
            {

                type: "line",

                data: {

                    labels: cycles,

                    datasets: [{

                        data: rul,

                        borderColor:
                            "#3e70ed",

                        backgroundColor:
                            "rgba(62,112,237,.08)",

                        fill: true,

                        borderWidth: 2,

                        pointRadius: 0,

                        tension: .3
                    }]
                },

                options:
                    baseOptions()
            }
        );


    const failureOptions =
        baseOptions();


    failureOptions.scales.y = {

        min: 0,

        max: 100,

        grid: {
            color: "#edf0f4"
        },

        ticks: {

            color: "#8793a5",

            callback:
                value => value + "%"
        }
    };


    failureChart =
        new Chart(
            document.getElementById(
                "failureChart"
            ),
            {

                type: "line",

                data: {

                    labels: cycles,

                    datasets: [{

                        data: probability,

                        borderColor:
                            "#d94755",

                        backgroundColor:
                            "rgba(217,71,85,.08)",

                        fill: true,

                        borderWidth: 2,

                        pointRadius: 0,

                        tension: .3
                    }]
                },

                options:
                    failureOptions
            }
        );


    const statusOptions =
        baseOptions();


    statusOptions.scales.y = {

        min: 0,

        max: 3,

        ticks: {

            stepSize: 1,

            color: "#8793a5",

            callback: value => {

                return [
                    "NORMAL",
                    "MONITOR",
                    "WARNING",
                    "CRITICAL"
                ][value] || "";

            }
        },

        grid: {
            color: "#edf0f4"
        }
    };


    statusChart =
        new Chart(
            document.getElementById(
                "statusChart"
            ),
            {

                type: "line",

                data: {

                    labels: cycles,

                    datasets: [{

                        data: states,

                        stepped: true,

                        borderColor:
                            "#152942",

                        backgroundColor:
                            "rgba(21,41,66,.05)",

                        fill: true,

                        borderWidth: 2,

                        pointRadius: 0
                    }]
                },

                options:
                    statusOptions
            }
        );
}


analyseButton.addEventListener(
    "click",
    () =>
        analyseEngine(
            engineSelect.value
        )
);


engineSelect.addEventListener(
    "change",
    () =>
        analyseEngine(
            engineSelect.value
        )
);


loadEngines();
