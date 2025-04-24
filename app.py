from flask import Flask, render_template, request, jsonify
from main import predict_stress
import os
import json
import logging
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
model_config_path = "config.json"

# Enhanced UI template path
ENHANCED_UI_PATH = "enhanced_ui.html"

# Create enhanced UI template
def create_enhanced_ui():
    """Create enhanced UI template file"""
    with open(ENHANCED_UI_PATH, 'w', encoding='utf-8') as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PeaceKeeper - Stress Detection</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body class="bg-gray-100 min-h-screen">
    <header class="gradient-bg text-white p-6 shadow-lg">
        <div class="container mx-auto">
            <h1 class="text-3xl font-bold">PeaceKeeper</h1>
            <p class="text-lg">AI-Powered Stress Level Detection</p>
        </div>
    </header>

    <main class="container mx-auto py-10 px-4">
        <div class="max-w-4xl mx-auto">
            <!-- Input Card -->
            <div class="card bg-white p-6 mb-8" id="inputCard" {% if result %}style="display: none;"{% endif %}>
                <h2 class="text-2xl font-semibold mb-4 text-gray-800">Analyze Your Stress Level</h2>
                <form id="stressForm" method="POST">
                    <div class="mb-4">
                        <label for="text_input" class="block text-gray-700 font-medium mb-2">Share your thoughts or feelings</label>
                        <textarea 
                            id="text_input" 
                            name="text_input" 
                            class="w-full h-32 p-3 border border-gray-300 rounded-lg resize-none focus:border-blue-500"
                            placeholder="Describe how you're feeling right now..."
                            required
                        ></textarea>
                    </div>
                    <button 
                        type="submit" 
                        class="w-full bg-blue-500 hover:bg-blue-600 text-white font-medium py-2 px-4 rounded-lg transition duration-200"
                    >
                        Analyze
                    </button>
                </form>
            </div>

            <!-- Results Card -->
            <div id="resultsCard" class="card p-6 {% if result %}result-{{ result.model_prediction.predicted_class }}{% else %}result-medium hidden{% endif %}">
                <h2 class="text-2xl font-semibold mb-4 text-gray-800">Your Stress Analysis</h2>
                
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <!-- Main Stress Level Indicator -->
                    <div class="bg-white rounded-lg p-4 col-span-1">
                        <div class="flex items-center mb-2">
                            <div id="stressLevelIndicator" class="w-3 h-3 rounded-full 
                            {% if result and result.model_prediction.predicted_class == 'low' %}
                                bg-green-500
                            {% elif result and result.model_prediction.predicted_class == 'high' %}
                                bg-red-500
                            {% else %}
                                bg-yellow-400
                            {% endif %} mr-2"></div>
                            <h3 class="text-xl font-medium" id="stressLevelText">
                            {% if result %}
                                {{ result.model_prediction.predicted_class|capitalize }} Stress
                            {% else %}
                                Medium Stress
                            {% endif %}
                            </h3>
                        </div>
                        <p id="resultDetails" class="text-gray-700 text-sm">
                        {% if result %}
                            {% if result.model_prediction.predicted_class == 'low' %}
                                Your input indicates a low level of stress. Continue with your self-care practices.
                            {% elif result.model_prediction.predicted_class == 'medium' %}
                                We've detected a moderate level of stress. It's important to be mindful of how you're feeling.
                            {% else %}
                                Your input indicates a high level of stress. Consider taking a break and practicing stress-reduction techniques.
                            {% endif %}
                        {% else %}
                            Based on your input, we've detected a moderate level of stress. It's important to be mindful of how you're feeling.
                        {% endif %}
                        </p>
                        
                        <div class="mt-4">
                            <div class="flex justify-between mb-1 text-sm">
                                <span class="text-gray-700">Model Confidence</span>
                                <span id="confidenceScore" class="text-gray-700">
                                {% if result %}
                                    {{ (result.model_prediction.confidence * 100)|int }}%
                                {% else %}
                                    85%
                                {% endif %}
                                </span>
                            </div>
                            <div class="w-full bg-gray-200 rounded-full h-2">
                                <div id="confidenceBar" class="bg-blue-500 h-2 rounded-full" style="width: {% if result %}{{ (result.model_prediction.confidence * 100)|int }}%{% else %}85%{% endif %}"></div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Stress Analysis Details -->
                    <div class="bg-white rounded-lg p-4 col-span-1">
                        <h3 class="text-lg font-medium mb-2">Stress Indicators</h3>
                        <div class="space-y-3">
                            <div>
                                <div class="flex justify-between mb-1 text-sm">
                                    <span class="text-gray-700">Sentiment Analysis</span>
                                    <span id="sentimentScore" class="text-gray-700">
                                    {% if result %}
                                        {{ result.basic_analysis.vader_scores.compound|round(2) }}
                                    {% else %}
                                        -0.2
                                    {% endif %}
                                    </span>
                                </div>
                                <div class="w-full bg-gray-200 rounded-full h-2">
                                    {% if result %}
                                        {% set sentiment_width = ((result.basic_analysis.vader_scores.compound + 1) / 2) * 100 %}
                                        {% if result.basic_analysis.vader_scores.compound < -0.3 %}
                                            <div id="sentimentBar" class="bg-red-500 h-2 rounded-full" style="width: {{ sentiment_width }}%"></div>
                                        {% elif result.basic_analysis.vader_scores.compound < 0.1 %}
                                            <div id="sentimentBar" class="bg-yellow-400 h-2 rounded-full" style="width: {{ sentiment_width }}%"></div>
                                        {% else %}
                                            <div id="sentimentBar" class="bg-green-500 h-2 rounded-full" style="width: {{ sentiment_width }}%"></div>
                                        {% endif %}
                                    {% else %}
                                        <div id="sentimentBar" class="bg-yellow-400 h-2 rounded-full" style="width: 60%"></div>
                                    {% endif %}
                                </div>
                            </div>
                            <div>
                                <div class="flex justify-between mb-1 text-sm">
                                    <span class="text-gray-700">Keyword Analysis</span>
                                    <span id="keywordScore" class="text-gray-700">
                                    {% if result %}
                                        {% set top_level = None %}
                                        {% set top_count = 0 %}
                                        {% for level, count in result.basic_analysis.stress_keywords_found.items() %}
                                            {% if count > top_count %}
                                                {% set top_level = level %}
                                                {% set top_count = count %}
                                            {% endif %}
                                        {% endfor %}
                                        {% if top_level %}
                                            {{ top_level|capitalize }} ({{ top_count }})
                                        {% else %}
                                            None found
                                        {% endif %}
                                    {% else %}
                                        Medium (3)
                                    {% endif %}
                                    </span>
                                </div>
                                <div class="w-full bg-gray-200 rounded-full h-2">
                                    {% if result %}
                                        {% set top_level = None %}
                                        {% set top_count = 0 %}
                                        {% for level, count in result.basic_analysis.stress_keywords_found.items() %}
                                            {% if count > top_count %}
                                                {% set top_level = level %}
                                                {% set top_count = count %}
                                            {% endif %}
                                        {% endfor %}
                                        {% if top_level == 'high' %}
                                            <div id="keywordBar" class="bg-red-500 h-2 rounded-full" style="width: 85%"></div>
                                        {% elif top_level == 'medium' %}
                                            <div id="keywordBar" class="bg-yellow-400 h-2 rounded-full" style="width: 50%"></div>
                                        {% elif top_level == 'low' %}
                                            <div id="keywordBar" class="bg-green-500 h-2 rounded-full" style="width: 25%"></div>
                                        {% else %}
                                            <div id="keywordBar" class="bg-gray-300 h-2 rounded-full" style="width: 0%"></div>
                                        {% endif %}
                                    {% else %}
                                        <div id="keywordBar" class="bg-yellow-400 h-2 rounded-full" style="width: 50%"></div>
                                    {% endif %}
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Detected Keywords -->
                    <div class="bg-white rounded-lg p-4 col-span-1">
                        <h3 class="text-lg font-medium mb-2">Detected Keywords</h3>
                        <div id="keywordsContainer" class="flex flex-wrap gap-2">
                            {% if result %}
                                {% set found_any = false %}
                                {% for level, count in result.basic_analysis.stress_keywords_found.items() %}
                                    {% if count > 0 %}
                                        {% set found_any = true %}
                                        {% if level == 'low' %}
                                            <span class="bg-green-100 text-green-800 text-xs font-semibold px-2.5 py-0.5 rounded">{{ level }} ({{ count }})</span>
                                        {% elif level == 'medium' %}
                                            <span class="bg-yellow-100 text-yellow-800 text-xs font-semibold px-2.5 py-0.5 rounded">{{ level }} ({{ count }})</span>
                                        {% else %}
                                            <span class="bg-red-100 text-red-800 text-xs font-semibold px-2.5 py-0.5 rounded">{{ level }} ({{ count }})</span>
                                        {% endif %}
                                    {% endif %}
                                {% endfor %}
                                {% if not found_any %}
                                    <p class="text-gray-500 text-sm">No specific stress keywords detected</p>
                                {% endif %}
                            {% else %}
                                <span class="bg-yellow-100 text-yellow-800 text-xs font-semibold px-2.5 py-0.5 rounded">medium (3)</span>
                            {% endif %}
                        </div>
                    </div>
                </div>
                
                <!-- Projection Dashboard -->
                <div class="bg-white rounded-lg p-4 mb-6">
                    <h3 class="text-lg font-medium mb-2">Stress Projection</h3>
                    <p class="text-sm text-gray-600 mb-4">Based on patterns in your text, here's a projection of how your stress might develop:</p>
                    <div class="h-64">
                        <canvas id="stressProjectionChart"></canvas>
                    </div>
                </div>
                
                <!-- Stress Management Suggestions -->
                <div class="bg-white rounded-lg p-4 mb-6">
                    <h3 class="text-lg font-medium mb-4">Stress Management Suggestions</h3>
                    <div id="suggestionsContainer" class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <!-- Suggestions will be populated dynamically via JavaScript -->
                    </div>
                </div>
                
                <div class="mt-6">
                    <a href="/" class="block w-full bg-white hover:bg-gray-100 text-gray-800 font-medium py-2 px-4 rounded-lg border border-gray-300 transition duration-200 text-center">
                        Analyze New Text
                    </a>
                </div>
            </div>
        </div>
    </main>

    <footer class="bg-gray-800 text-white py-6 mt-10">
        <div class="container mx-auto px-4 text-center">
            <p>PeaceKeeper &copy; 2025 | AI-Powered Stress Detection System</p>
            <p class="text-gray-400 text-sm mt-2">This tool is for informational purposes only and not a substitute for professional mental health advice.</p>
        </div>
    </footer>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const stressForm = document.getElementById('stressForm');
            const inputCard = document.getElementById('inputCard');
            const resultsCard = document.getElementById('resultsCard');
            let projectionChart = null;
            
            // Stress management suggestions
            const lowStressSuggestions = [
                { title: "Maintain Balance", content: "Continue your current self-care practices to maintain your low stress levels.", icon: "🧘‍♀️" },
                { title: "Mindful Moments", content: "Take a few minutes each day to appreciate positive aspects of your life.", icon: "🌱" },
                { title: "Regular Exercise", content: "Stay active with regular physical activity that you enjoy.", icon: "🚶‍♂️" },
                { title: "Quality Sleep", content: "Maintain your healthy sleep schedule - aim for 7-9 hours per night.", icon: "😴" }
            ];
            
            const mediumStressSuggestions = [
                { title: "Deep Breathing", content: "Practice deep breathing exercises - inhale for 4 seconds, hold for 4, exhale for 6.", icon: "🫁" },
                { title: "Physical Activity", content: "Even a short 10-minute walk can reduce stress hormones.", icon: "🏃‍♀️" },
                { title: "Set Boundaries", content: "Learn to say no to additional responsibilities when you're feeling stressed.", icon: "🛑" },
                { title: "Connect with Others", content: "Share your feelings with someone you trust - social support is powerful.", icon: "👥" }
            ];
            
            const highStressSuggestions = [
                { title: "Seek Support", content: "Consider talking to a mental health professional about what you're experiencing.", icon: "🧠" },
                { title: "Progressive Relaxation", content: "Try tensing and then relaxing each muscle group, starting from your toes to your head.", icon: "💆‍♂️" },
                { title: "Break Tasks Down", content: "Divide overwhelming responsibilities into smaller, manageable steps.", icon: "📝" },
                { title: "Limit Stimulants", content: "Reduce caffeine and sugar intake, which can increase anxiety and stress.", icon: "☕" },
                { title: "Take a Break", content: "Give yourself permission to step away and take a mental health break.", icon: "⏸️" }
            ];
            
            // Check if we need to display results (for server-side rendering)
            if (document.getElementById('resultsCard') && !document.getElementById('resultsCard').classList.contains('hidden')) {
                // Create projection chart and suggestions for server-side rendered results
                createProjectionChart();
                displaySuggestions();
            }
            
            // Form submission
            if (stressForm) {
                stressForm.addEventListener('submit', function(e) {
                    e.preventDefault();
                    
                    const textInput = document.getElementById('text_input').value;
                    
                    // Make AJAX request to server
                    fetch('/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded',
                            'Accept': 'application/json'
                        },
                        body: new URLSearchParams({
                            'text_input': textInput
                        })
                    })
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`HTTP error! Status: ${response.status}`);
                        }
                        return response.json();
                    })
                    .then(data => {
                        // Update UI with results
                        updateResultsUI(data);
                        
                        // Show results card
                        inputCard.classList.add('hidden');
                        resultsCard.classList.remove('hidden');
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        alert('An error occurred. Please try again.');
                    });
                });
            }
            
            // Create and update stress projection chart
            function createProjectionChart(stressLevel, sentimentScore) {
                if (!document.getElementById('stressProjectionChart')) {
                    return;
                }
                
                const ctx = document.getElementById('stressProjectionChart').getContext('2d');
                
                // If data is not provided (server-side rendering), extract it from the page
                if (!stressLevel) {
                    const levelText = document.getElementById('stressLevelText').textContent.toLowerCase();
                    stressLevel = levelText.includes('low') ? 'low' : 
                                  levelText.includes('medium') ? 'medium' : 'high';
                }
                
                if (!sentimentScore) {
                    const sentimentElement = document.getElementById('sentimentScore');
                    if (sentimentElement) {
                        sentimentScore = parseFloat(sentimentElement.textContent);
                    } else {
                        sentimentScore = 0;
                    }
                }
                
                // Convert stress level to numeric value
                let stressValue;
                if (stressLevel === 'low') stressValue = 0.2;
                else if (stressLevel === 'medium') stressValue = 0.5;
                else stressValue = 0.8;
                
                // Generate projection data
                // This is a simplified model - in a real app, this would use more sophisticated algorithms
                const today = 0;
                let projectedData = [stressValue];
                
                // Create projection based on current stress and sentiment
                // Sentiment affects the trajectory
                const sentimentFactor = (sentimentScore + 1) / 2; // Convert from [-1,1] to [0,1]
                
                for (let i = 1; i <= 7; i++) {
                    // Higher stress tends to increase more without intervention
                    // Negative sentiment accelerates stress increase
                    const changeRate = stressValue * 0.1 * (1 - sentimentFactor);
                    
                    // Add some random variation
                    const randomFactor = (Math.random() * 0.1) - 0.05;
                    
                    // Calculate next value with a ceiling of 1.0
                    let nextValue = projectedData[i-1] + changeRate + randomFactor;
                    nextValue = Math.min(Math.max(nextValue, 0), 1); // Keep between 0 and 1
                    
                    projectedData.push(nextValue);
                }
                
                // Create labels for the next 7 days
                const labels = ['Today'];
                for (let i = 1; i <= 7; i++) {
                    labels.push(`Day ${i}`);
                }
                
                // Determine background colors based on projection
                const backgroundColors = projectedData.map(value => {
                    if (value < 0.3) return 'rgba(72, 187, 120, 0.2)'; // Green for low
                    if (value < 0.7) return 'rgba(237, 137, 54, 0.2)'; // Orange for medium
                    return 'rgba(229, 62, 62, 0.2)'; // Red for high
                });
                
                // Determine border colors based on projection
                const borderColors = projectedData.map(value => {
                    if (value < 0.3) return 'rgb(72, 187, 120)'; // Green for low
                    if (value < 0.7) return 'rgb(237, 137, 54)'; // Orange for medium
                    return 'rgb(229, 62, 62)'; // Red for high
                });
                
                // Create the chart
                if (projectionChart) {
                    projectionChart.destroy();
                }
                
                projectionChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Projected Stress Level',
                            data: projectedData,
                            backgroundColor: backgroundColors,
                            borderColor: borderColors,
                            borderWidth: 2,
                            tension: 0.3,
                            pointBackgroundColor: borderColors,
                            pointRadius: 5,
                            pointHoverRadius: 7,
                            fill: true
                        }]
                    },
                    options: {
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 1,
                                ticks: {
                                    callback: function(value) {
                                        if (value <= 0.3) return 'Low';
                                        if (value <= 0.7) return 'Medium';
                                        return 'High';
                                    }
                                }
                            }
                        },
                        plugins: {
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        const value = context.parsed.y;
                                        let label = '';
                                        
                                        if (value <= 0.3) label = 'Low Stress';
                                        else if (value <= 0.7) label = 'Medium Stress';
                                        else label = 'High Stress';
                                        
                                        return label + ': ' + Math.round(value * 100) + '%';
                                    }
                                }
                            }
                        },
                        responsive: true,
                        maintainAspectRatio: false
                    }
                });
                
                return projectionChart;
            }
            
            // Display stress management suggestions
            function displaySuggestions(stressLevel) {
                const suggestionsContainer = document.getElementById('suggestionsContainer');
                if (!suggestionsContainer) return;
                
                suggestionsContainer.innerHTML = '';
                
                // If stressLevel is not provided, extract it from the page (for server-side rendering)
                if (!stressLevel) {
                    const levelText = document.getElementById('stressLevelText').textContent.toLowerCase();
                    stressLevel = levelText.includes('low') ? 'low' : 
                                  levelText.includes('medium') ? 'medium' : 'high';
                }
                
                let suggestions;
                let borderColor;
                
                if (stressLevel === 'low') {
                    suggestions = lowStressSuggestions;
                    borderColor = 'border-green-500';
                } else if (stressLevel === 'medium') {
                    suggestions = mediumStressSuggestions;
                    borderColor = 'border-yellow-500';
                } else {
                    suggestions = highStressSuggestions;
                    borderColor = 'border-red-500';
                }
                
                // Randomly select suggestions to display (up to 4)
                const selectedSuggestions = [...suggestions].sort(() => 0.5 - Math.random()).slice(0, 4);
                
                selectedSuggestions.forEach(suggestion => {
                    const suggestionCard = document.createElement('div');
                    suggestionCard.className = `suggestion-card ${borderColor} bg-white p-3 rounded shadow-sm`;
                    
                    suggestionCard.innerHTML = `
                        <div class="flex items-start">
                            <div class="text-2xl mr-3">${suggestion.icon}</div>
                            <div>
                                <h4 class="font-medium">${suggestion.title}</h4>
                                <p class="text-sm text-gray-600">${suggestion.content}</p>
                            </div>
                        </div>
                    `;
                    
                    suggestionsContainer.appendChild(suggestionCard);
                });
            }
            
            // Update UI with results
            function updateResultsUI(data) {
                const stressLevel = data.model_prediction.predicted_class;
                const confidence = data.model_prediction.confidence;
                const sentiment = data.basic_analysis.vader_scores.compound;
                const keywordsFound = data.basic_analysis.stress_keywords_found;
                
                // Update stress level indicator
                const stressLevelIndicator = document.getElementById('stressLevelIndicator');
                const stressLevelText = document.getElementById('stressLevelText');
                const resultDetails = document.getElementById('resultDetails');
                
                // Update card class
                resultsCard.className = 'card result-' + stressLevel + ' p-6';
                
                if (stressLevel === 'low') {
                    stressLevelIndicator.className = 'w-3 h-3 rounded-full bg-green-500 mr-2';
                    stressLevelText.textContent = 'Low Stress';
                    resultDetails.textContent = 'Your input indicates a low level of stress. Continue with your self-care practices.';
                } else if (stressLevel === 'medium') {
                    stressLevelIndicator.className = 'w-3 h-3 rounded-full bg-yellow-400 mr-2';
                    stressLevelText.textContent = 'Medium Stress';
                    resultDetails.textContent = 'We\'ve detected a moderate level of stress. It\'s important to be mindful of how you\'re feeling.';
                } else {
                    stressLevelIndicator.className = 'w-3 h-3 rounded-full bg-red-500 mr-2';
                    stressLevelText.textContent = 'High Stress';
                    resultDetails.textContent = 'Your input indicates a high level of stress. Consider taking a break and practicing stress-reduction techniques.';
                }
                
                // Update sentiment score
                document.getElementById('sentimentScore').textContent = sentiment.toFixed(2);
                const sentimentBar = document.getElementById('sentimentBar');
                // Convert sentiment from [-1,1] to [0,100]
                const sentimentWidth = ((sentiment + 1) / 2) * 100;
                sentimentBar.style.width = sentimentWidth + '%';
                
                if (sentiment < -0.3) {
                    sentimentBar.className = 'bg-red-500 h-2 rounded-full';
                } else if (sentiment < 0.1) {
                    sentimentBar.className = 'bg-yellow-400 h-2 rounded-full';
                } else {
                    sentimentBar.className = 'bg-green-500 h-2 rounded-full';
                }
                
                // Update keyword score
                const keywordInfo = Object.entries(keywordsFound)
                    .filter(([_, count]) => count > 0)
                    .sort(([_, countA], [__, countB]) => countB - countA);
                
                if (keywordInfo.length > 0) {
                    const [topKeywordLevel, topKeywordCount] = keywordInfo[0];
                    document.getElementById('keywordScore').textContent = 
                        topKeywordLevel.charAt(0).toUpperCase() + topKeywordLevel.slice(1) + 
                        ' (' + topKeywordCount + ')';
                    
                    const keywordBar = document.getElementById('keywordBar');
                    if (topKeywordLevel === 'high') {
                        keywordBar.className = 'bg-red-500 h-2 rounded-full';
                        keywordBar.style.width = '85%';
                    } else if (topKeywordLevel === 'medium') {
                        keywordBar.className = 'bg-yellow-400 h-2 rounded-full';
                        keywordBar.style.width = '50%';
                    } else {
                        keywordBar.className = 'bg-green-500 h-2 rounded-full';
                        keywordBar.style.width = '25%';
                    }
                } else {
                    document.getElementById('keywordScore').textContent = 'None found';
                    document.getElementById('keywordBar').style.width = '0%';
                }
                
                // Update confidence score
                document.getElementById('confidenceScore').textContent = (confidence * 100).toFixed(0) + '%';
                document.getElementById('confidenceBar').style.width = (confidence * 100) + '%';
                
                // Display keywords
                displayKeywords(keywordsFound);
                
                // Create projection chart
                createProjectionChart(stressLevel, sentiment);
                
                // Display suggestions
                displaySuggestions(stressLevel);
            }
            
            // Display detected keywords
            function displayKeywords(keywordsFound) {
                const keywordsContainer = document.getElementById('keywordsContainer');
                if (!keywordsContainer) return;
                
                keywordsContainer.innerHTML = '';
                
                // If keywords is not provided, return (for server-side rendering we already handled this)
                if (!keywordsFound) return;
                
                // Check if keywords were found
                if (Object.values(keywordsFound).every(count => count === 0)) {
                    keywordsContainer.innerHTML = '<p class="text-gray-500 text-sm">No specific stress keywords detected</p>';
                    return;
                }
                
                // Create keyword badges
                for (const [level, count] of Object.entries(keywordsFound)) {
                    if (count === 0) continue;
                    
                    let bgColor, textColor;
                    if (level === 'low') {
                        bgColor = 'bg-green-100';
                        textColor = 'text-green-800';
                    } else if (level === 'medium') {
                        bgColor = 'bg-yellow-100';
                        textColor = 'text-yellow-800';
                    } else {
                        bgColor = 'bg-red-100';
                        textColor = 'text-red-800';
                    }
                    
                    const badge = document.createElement('span');
                    badge.className = `${bgColor} ${textColor} text-xs font-semibold px-2.5 py-0.5 rounded`;
                    badge.textContent = `${level} (${count})`;
                    keywordsContainer.appendChild(badge);
                }
            }
        });
    </script>
</body>
</html>""")
    logger.info(f"Created enhanced UI template at {ENHANCED_UI_PATH}")

# Make sure model directories exist
def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = ['data', 'processed_data', 'models', 'results', 'logs', 'static', 'templates']
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            logger.info(f"Created directory: {dir_name}")

# Save the HTML template
def save_template():
    """Save the HTML template to the templates directory."""
    template_path = os.path.join('templates', 'index.html')
    
    # Check if enhanced UI exists, otherwise create it
    if not os.path.exists(ENHANCED_UI_PATH):
        create_enhanced_ui()
    
    # Read the HTML content from the enhanced UI
    with open(ENHANCED_UI_PATH, 'r', encoding='utf-8') as f:
        template_content = f.read()
    
    # Ensure the templates directory exists
    os.makedirs('templates', exist_ok=True)
    
    # Save the template
    with open(template_path, 'w', encoding='utf-8') as file:
        file.write(template_content)
    
    logger.info(f"Saved template to {template_path}")

# Create the basic CSS file
def create_static_files():
    """Create necessary static files."""
    css_path = os.path.join('static', 'style.css')
    
    # Ensure the static directory exists
    os.makedirs('static', exist_ok=True)
    
    # Basic CSS
    css_content = """
    .gradient-bg {
        background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
    }
    .result-low {
        background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
    }
    .result-medium {
        background: linear-gradient(120deg, #f6d365 0%, #fda085 100%);
    }
    .result-high {
        background: linear-gradient(120deg, #ff9a9e 0%, #fecfef 100%);
    }
    .card {
        border-radius: 1rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    .card:hover {
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        transform: translateY(-4px);
    }
    textarea:focus, button:focus {
        outline: none;
        box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.5);
    }
    .suggestion-card {
        border-left: 4px solid;
        transition: all 0.3s ease;
    }
    .suggestion-card:hover {
        transform: translateX(4px);
    }
    """
    
    # Save CSS file
    with open(css_path, 'w') as file:
        file.write(css_content)
    
    logger.info(f"Created CSS file at {css_path}")

def generate_initial_data():
    """Generate initial data and train model if needed."""
    try:
        from create_test_data import create_test_data
        
        # Create test data
        test_data_path = os.path.join("data", "test_data.csv")
        if not os.path.exists(test_data_path):
            create_test_data(test_data_path)
            logger.info(f"Created test data at {test_data_path}")
        
        # Generate synthetic data for training if model doesn't exist
        if not os.path.exists(os.path.join("models", "stress_detection_model.h5")):
            logger.info("Model not found. Generating synthetic data and training model.")
            from generate_dataset import DatasetGenerator
            from text_preprocessing import DataPreprocessor
            
            # Generate synthetic data
            generator = DatasetGenerator(model_config_path)
            train_path, val_path, test_path = generator.generate_and_save_datasets(
                train_size=200, val_size=50, test_size=50
            )
            
            # Preprocess data
            preprocessor = DataPreprocessor(model_config_path)
            train_df, val_df = preprocessor.prepare_training_data('models')
            
            # Train model
            from main import train_model
            logger.info("Training new model with synthetic data")
            model, metrics = train_model(model_config_path, train_df, val_df)
            
            # Log training results
            logger.info(f"Model trained successfully. Validation accuracy: {metrics['accuracy']:.4f}")
    except Exception as e:
        logger.error(f"Error generating initial data: {str(e)}")
        logger.error(traceback.format_exc())

# Make sure model directories exist
def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = ['data', 'processed_data', 'models', 'results', 'logs', 'static', 'templates']
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            logger.info(f"Created directory: {dir_name}")

# Save the HTML template
def save_template():
    """Save the HTML template to the templates directory."""
    template_path = os.path.join('templates', 'index.html')
    
    # Read the HTML content from the enhanced UI
    with open('enhanced_ui.html', 'r', encoding='utf-8') as f:
        template_content = f.read()
    
    # Ensure the templates directory exists
    os.makedirs('templates', exist_ok=True)
    
    # Save the template
    with open(template_path, 'w', encoding='utf-8') as file:
        file.write(template_content)
    
    logger.info(f"Saved template to {template_path}")

# Basic CSS file
def create_static_files():
    """Create necessary static files."""
    css_path = os.path.join('static', 'style.css')
    js_path = os.path.join('static', 'script.js')
    
    # Ensure the static directory exists
    os.makedirs('static', exist_ok=True)
    
    # Basic CSS
    css_content = """
    .gradient-bg {
        background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
    }
    .result-low {
        background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
    }
    .result-medium {
        background: linear-gradient(120deg, #f6d365 0%, #fda085 100%);
    }
    .result-high {
        background: linear-gradient(120deg, #ff9a9e 0%, #fecfef 100%);
    }
    .card {
        border-radius: 1rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    """
    
    # Save 
    with open(css_path, 'w') as file:
        file.write(css_content)
    
    logger.info(f"Created CSS file at {css_path}")

def generate_initial_data():
    """Generate initial data and train model if needed."""
    try:
        from create_test_data import create_test_data
        
        # Create test data
        test_data_path = os.path.join("data", "test_data.csv")
        if not os.path.exists(test_data_path):
            create_test_data(test_data_path)
            logger.info(f"Created test data at {test_data_path}")
        
        # Generate synthetic data for training if model doesn't exist
        if not os.path.exists(os.path.join("models", "stress_detection_model.h5")):
            logger.info("Model not found. Generating synthetic data and training model.")
            from generate_dataset import DatasetGenerator
            from text_preprocessing import DataPreprocessor
            
            # Generate synthetic data
            generator = DatasetGenerator(model_config_path)
            train_path, val_path, test_path = generator.generate_and_save_datasets(
                train_size=200, val_size=50, test_size=50
            )
            
            # Preprocess data
            preprocessor = DataPreprocessor(model_config_path)
            train_df, val_df = preprocessor.prepare_training_data('models')
            
            # Train model
            from main import train_model
            logger.info("Training new model with synthetic data")
            model, metrics = train_model(model_config_path, train_df, val_df)
            
            # Log training results
            logger.info(f"Model trained successfully. Validation accuracy: {metrics['accuracy']:.4f}")
    except Exception as e:
        logger.error(f"Error generating initial data: {str(e)}")
        logger.error(traceback.format_exc())

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            user_input = request.form.get("text_input", "")
            if not user_input:
                return jsonify({"error": "No input provided"}), 400

            config_path = "config.json"  
            result = predict_stress(config_path, user_input)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": "An error occurred during prediction"}), 500

    return render_template("index.html", result=None)

