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

// File upload validation
function uploadFile() {
    let fileInput = document.getElementById("audio-file");
    if (fileInput.files.length === 0) {
        alert("Please upload an audio file!");
    } else {
        alert("File uploaded successfully!");
    }
}

// Radar Chart for 16 Parameters using Chart.js
document.addEventListener("DOMContentLoaded", function () {
    // Initialize the chart
    initRadarChart();
    
    // Setup form submission
    document.getElementById('upload-form').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const fileInput = document.getElementById('audio-file');
        const file = fileInput.files[0];
        
        if (!file) {
            alert('Please select a file');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', file);
        
        // Show loader
        document.getElementById('loader').style.display = 'block';
        document.getElementById('result').style.display = 'none';
        
        // Send request
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Hide loader
            document.getElementById('loader').style.display = 'none';
            
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }
            
            // Show result
            document.getElementById('genre').textContent = data.genre;
            document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(2) + '%';
            
            // Display all probabilities
            const probContainer = document.getElementById('probabilities');
            probContainer.innerHTML = '';
            
            const genres = Object.keys(data.all_probs).sort();
            genres.forEach(genre => {
                const prob = data.all_probs[genre];
                const percentage = (prob * 100).toFixed(2) + '%';
                
                const label = document.createElement('div');
                label.textContent = genre + ': ' + percentage;
                
                const barContainer = document.createElement('div');
                barContainer.className = 'progress-bar';
                
                const bar = document.createElement('div');
                bar.className = 'progress';
                bar.style.width = percentage;
                bar.textContent = percentage;
                
                barContainer.appendChild(bar);
                probContainer.appendChild(label);
                probContainer.appendChild(barContainer);
            });
            
            document.getElementById('result').style.display = 'block';
            
            // Update radar chart with the prediction data
            updateRadarChart(data);
        })
        .catch(error => {
            document.getElementById('loader').style.display = 'none';
            alert('Error: ' + error);
        });
    });
});

// Image switcher function
let images = ["./image1.jpg", "./image2.jpg", "./image3.png"];
let index = 0;
function switchImage() {
    document.getElementById("blinking-image").src = images[index];
    index = (index + 1) % images.length;
}
setInterval(switchImage, 3000);

// Initialize radar chart
function initRadarChart() {
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

    window.radarChart = new Chart(ctx, {
        type: "radar",
        data: data,
        options: {
            responsive: true,
            scale: {
                ticks: { beginAtZero: true }
            }
        }
    });
}

// Update radar chart with prediction data
function updateRadarChart(data) {
    // Extract probabilities and normalize them to fit the radar chart
    const genres = Object.keys(data.all_probs);
    let probs = Object.values(data.all_probs);
    
    // If we have less than 16 parameters, pad with zeroes
    while(probs.length < 16) {
        probs.push(0);
    }
    
    // If we have more than 16 parameters, take the first 16
    if(probs.length > 16) {
        probs = probs.slice(0, 16);
    }
    
    // Scale values to 0-20 range for visualization
    probs = probs.map(val => val * 20);
    
    // Update chart data
    window.radarChart.data.datasets[0].data = probs;
    
    // If we have enough genres, update labels
    const labels = genres.slice(0, 16);
    while(labels.length < 16) {
        labels.push("P" + (labels.length + 1));
    }
    window.radarChart.data.labels = labels;
    
    // Update chart
    window.radarChart.update();
}
