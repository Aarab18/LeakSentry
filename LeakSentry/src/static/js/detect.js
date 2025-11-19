let chartInstance = null;
const dripSound = new Audio("https://www.soundjay.com/nature/drip-1.mp3");
const clickSound = new Audio("https://www.soundjay.com/buttons/button-3.mp3");
let demoInterval;

document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("input[type='range']").forEach(slider => {
        const valueDisplay = document.getElementById(`${slider.id}_value`);
        const errorDisplay = document.getElementById(`${slider.id}_error`);
        slider.addEventListener("input", () => {
            const unit = slider.id.includes("PSI") ? "PSI" : slider.id.includes("GPM") ? "GPM" : slider.id.includes("Cel") ? "°C" : slider.id.includes("Percent") ? "%" : "dB";
            valueDisplay.textContent = `${slider.value} ${unit}`;
            validateInput(slider, errorDisplay);
            updateLiveProbability();
        });
    });
    updateLiveProbability();

    document.getElementById('demoMode').addEventListener('change', function() {
        if (this.checked) {
            demoInterval = setInterval(() => {
                fetch('/demo-data')
                    .then(response => response.json())
                    .then(data => {
                        Object.keys(data).forEach(key => {
                            const slider = document.getElementById(key);
                            slider.value = data[key];
                            slider.dispatchEvent(new Event('input'));
                        });
                    });
            }, 3000);
        } else {
            clearInterval(demoInterval);
        }
    });
});

function validateInput(slider, errorDisplay) {
    const value = parseFloat(slider.value);
    const min = parseFloat(slider.min);
    const max = parseFloat(slider.max);
    errorDisplay.textContent = (value < min || value > max) ? `Value must be between ${min} and ${max}` : '';
}

function getInputs() {
    return {
        Pressure_PSI: document.getElementById("Pressure_PSI").value,
        FlowRate_GPM: document.getElementById("FlowRate_GPM").value,
        Temperature_Cel: document.getElementById("Temperature_Cel").value,
        Moisture_Percent: document.getElementById("Moisture_Percent").value,
        Acoustic_dB: document.getElementById("Acoustic_dB").value,
    };
}

function updateLiveProbability() {
    const inputs = getInputs();
    fetch("/predict/live", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inputs),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) throw new Error(data.error);
        const prob = (data.probability * 100).toFixed(2);
        const liveProb = document.getElementById("live-probability");
        liveProb.textContent = `Live Leak Probability: ${prob}%`;
        liveProb.className = prob > 75 ? "high" : prob > 50 ? "medium" : "low";
        updateChart(prob);
    })
    .catch(error => {
        document.getElementById("live-probability").textContent = "Error fetching probability";
    });
}

function updateChart(probability) {
    const ctx = document.getElementById("probabilityChart").getContext("2d");
    if (chartInstance) chartInstance.destroy();
    chartInstance = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Leak Risk", "Safe"],
            datasets: [{
                data: [probability, 100 - probability],
                backgroundColor: [probability > 75 ? "#ef5350" : probability > 50 ? "#ffca28" : "#66bb6a", "rgba(255, 255, 255, 0.2)"],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            cutout: "70%",
            plugins: { legend: { display: false } },
            animation: { animateRotate: true, animateScale: true }
        }
    });
}

function predict() {
    clickSound.play();
    const progressBar = document.getElementById("progressBar");
    const spinner = document.getElementById("spinner");
    const waterSpill = document.getElementById("waterSpill");
    progressBar.style.display = "none";
    spinner.style.display = "block";
    waterSpill.style.display = "none";

    const inputs = getInputs();
    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inputs),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) throw new Error(data.error);
        setTimeout(() => {
            spinner.style.display = "none";
            const inputs = data.input_values;
            const probability = data.probability;

            const modal = document.getElementById("resultModal");
            const modalContent = document.getElementById("modalContent");
            const modalTitle = document.getElementById("modalTitle");
            const modalProbability = document.getElementById("modalProbability");
            const modalInputs = document.getElementById("modalInputs");
            const modalRecommendation = document.getElementById("modalRecommendation");

            modalTitle.textContent = data.leak ? "🚨 Leak Detected!" : "✅ Pipes Secure!";
            modalContent.className = "modal-content " + (data.leak ? "leak" : "no-leak");
            modalProbability.textContent = `Probability: ${(probability * 100).toFixed(2)}%`;
            modalInputs.innerHTML = `
                <strong>Evidence:</strong><br>
                💨 Pressure: ${inputs.Pressure_PSI} PSI<br>
                🌊 Flow: ${inputs.FlowRate_GPM} GPM<br>
                🌡️ Temp: ${inputs.Temperature_Cel} °C<br>
                💧 Moisture: ${inputs.Moisture_Percent} %<br>
                🔊 Acoustic: ${inputs.Acoustic_dB} dB
                ${data.leak ? "<br><strong>Alerts Dispatched!</strong>" : ""}
                ${data.alert ? `<br><em>${data.alert}</em>` : ""}
            `;
            modalRecommendation.textContent = data.recommendation ? `Action: ${data.recommendation}` : '';
            modal.style.display = "flex";

            if (data.leak) {
                waterSpill.style.display = "block";
                dripSound.play();
            }
        }, 1500);
    })
    .catch(error => {
        spinner.style.display = "none";
        showModal("⚠️ Case Unsolved!", `Error: ${error.message}`, "", "");
    });
}

function showModal(title, probability, inputs, recommendation) {
    const modal = document.getElementById("resultModal");
    document.getElementById("modalTitle").textContent = title;
    document.getElementById("modalProbability").textContent = probability;
    document.getElementById("modalInputs").textContent = inputs;
    document.getElementById("modalRecommendation").textContent = recommendation;
    modal.style.display = "flex";
}

function closeModal() {
    document.getElementById("resultModal").style.display = "none";
    document.getElementById("waterSpill").style.display = "none";
    dripSound.pause();
    dripSound.currentTime = 0;
}