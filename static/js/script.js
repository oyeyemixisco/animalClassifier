// Javascript code for handling image upload and prediction
function predict() {
    const input = document.getElementById("fileInput");
    const loader = document.getElementById("loader-container");
    const result = document.getElementById("result");
    const preview = document.getElementById("preview");

    if (!input.files[0]) {
        result.innerHTML = "❌ Please select an image.";
        return;
    }

    // Preview image
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.style.display = "block";
    };
    reader.readAsDataURL(input.files[0]);

    // Show loader
    loader.style.display = "flex";
    result.innerHTML = "";

    const formData = new FormData();
    formData.append("file", input.files[0]);

    fetch(`${window.location.origin}/predict`, {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        // hide loader after result is received
        loader.style.display = "none"; 
        
        if (data.success) {
            const confidence = data.confidence;
            let level = "";
            let color = "";

            if (confidence >= 0.70) {  
                level = "High";
                color = "green";
            } else {
                level = "Low";
                color = "red";
            }

            result.innerHTML = `
                Prediction: <b>${data.prediction}</b><br>
                Confidence Level: 
                <span style="color:${color}; font-weight:bold;">
                    ${level} (${confidence.toFixed(2)})
                </span>
            `;
        } else {
            result.innerHTML = `❌ Error: ${data.error}`;
        }
    })

    .catch(() => {
        loader.style.display = "none";
        result.innerHTML = "❌ Failed to reach server.";
    });
}