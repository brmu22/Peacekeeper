async function analyzeText() {
    const input = document.getElementById("text_input").value;
    
    if (!input) {
        alert("Please enter some text to analyze.");
        return;
    }

    try {
        const formData = new FormData();
        formData.append("text_input", input);

        const res = await fetch("/", {
            method: "POST",
            headers: { 
                "Accept": "application/json" 
            },
            body: formData
        });

        if (!res.ok) {
            throw new Error(`HTTP error! Status: ${res.status}`);
        }

        const data = await res.json();

        // Hide input card, show results card
        document.getElementById("inputCard").classList.add("hidden");
        document.getElementById("resultsCard").classList.remove("hidden");
        
        // Update UI based on stress level
        updateResultsUI(data);
    } catch (error) {
        console.error("Error:", error);
        alert("An error occurred during analysis. Please try again.");
    }
}

function updateResultsUI(data) {
    const stressLevel = data.model_prediction.predicted_class;
    const confidence = data.model_prediction.confidence;
    const sentiment = data.basic_analysis.vader_scores.compound;
    
    // Update results card class
    document.getElementById("resultsCard").className = `card result-${stressLevel} p-6`;
    
    // Update stress level indicator
    const indicator = document.getElementById("stressLevelIndicator");
    const levelText = document.getElementById("stressLevelText");
    const details = document.getElementById("resultDetails");
    
    if (stressLevel === 'low') {
        indicator.className = 'w-3 h-3 rounded-full bg-green-500 mr-2';
        levelText.textContent = 'Low Stress';
        details.textContent = 'Your input indicates a low level of stress. Continue with your self-care practices.';
    } else if (stressLevel === 'medium') {
        indicator.className = 'w-3 h-3 rounded-full bg-yellow-400 mr-2';
        levelText.textContent = 'Medium Stress';
        details.textContent = 'We\'ve detected a moderate level of stress. It\'s important to be mindful of how you\'re feeling.';
    } else {
        indicator.className = 'w-3 h-3 rounded-full bg-red-500 mr-2';
        levelText.textContent = 'High Stress';
        details.textContent = 'Your input indicates a high level of stress. Consider taking a break and practicing stress-reduction techniques.';
    }
    
    // Update confidence
    document.getElementById("confidenceScore").textContent = `${(confidence * 100).toFixed(0)}%`;
    document.getElementById("confidenceBar").style.width = `${confidence * 100}%`;
    
    // Create projection chart and suggestions
    createStressProjectionChart(stressLevel, sentiment);
    displaySuggestions(stressLevel);
}

