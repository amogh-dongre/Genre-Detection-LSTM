// Auto-play jazz music on load and fade out after 10 seconds
function playJazz() {
    let audio = document.getElementById("jazzAudio");

    // Play audio only if not already playing
    if (audio.paused) {
        audio.volume = 0.5;
        audio.play();
    }

    setTimeout(() => {
        let fadeOut = setInterval(() => {
            if (audio.volume > 0.05) {
                audio.volume -= 0.05;
            } else {
                clearInterval(fadeOut);
                audio.pause();
                audio.currentTime = 0;
            }
        }, 500);
    }, 10000);
}

// Ensure music plays every time user switches tab or reloads
document.addEventListener("visibilitychange", function() {
    if (!document.hidden) {
        playJazz();
    }
});

// File upload alert
function uploadFile() {
    let fileInput = document.getElementById("audio-upload");
    if (fileInput.files.length === 0) {
        alert("Please upload an audio file!");
    } else {
        alert("File uploaded successfully!");
    }
}

// Classify music button function
function classifyMusic() {
    alert("Classifying music... Please wait!");
}
// Radar Chart for 16 Parameters using Chart.js
document.addEventListener("DOMContentLoaded", function () {
    const ctx = document.getElementById("radarChart").getContext("2d");

    const data = {
        labels: ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16"],
        datasets: [{
            label: "Parameter Values",
            data: [12, 15, 8, 10, 18, 9, 14, 13, 16, 11, 7, 20, 12, 10, 15, 17], // Sample data
            backgroundColor: "rgba(0, 123, 255, 0.2)",
            borderColor: "rgba(0, 123, 255, 1)",
            borderWidth: 2
        }]
    };

    new Chart(ctx, {
        type: "radar",
        data: data,
        options: {
            responsive: true,
            scale: {
                ticks: { beginAtZero: true }
            }
        }
    });
});
