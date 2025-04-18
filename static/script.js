async function analyzeText() {
    const input = document.getElementById("userInput").value;

    const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input })
    });

    const data = await res.json();

    document.getElementById("predictedClass").innerText = data.predicted_class;
    document.getElementById("confidence").innerText = (data.confidence * 100).toFixed(2) + "%";
    document.getElementById("result").style.display = "block";
}

