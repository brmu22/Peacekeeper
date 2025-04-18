from flask import Flask, render_template, request
from main import predict_stress

app = Flask(__name__)
model_config_path = "config.json"

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        user_input = request.form.get("text_input", "").strip()
        if not user_input:
            error = "Please enter some text."
        else:
            try:
                result = predict_stress(model_config_path, user_input)
            except Exception as e:
                error = f"Prediction error: {str(e)}"

    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True)
 


