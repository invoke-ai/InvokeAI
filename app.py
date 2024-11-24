from flask import Flask, send_from_directory, jsonify, request
import openai

app = Flask(__name__, static_folder='static')

openai.api_key = 'YOUR_API_KEY'

def base_369_framework():
    log = []
    for i in range(1, 370):
        log.append(f"Running iteration {i} of 369")
    return "\n".join(log)

def fibonacci_sequence():
    log = []
    a, b = 0, 1
    for i in range(10):
        a, b = b, a + b
        log.append(f"Fibonacci number {i}: {a}")
    return "\n".join(log)

def golden_ratio():
    phi = (1 + 5 ** 0.5) / 2
    return f"Golden Ratio (φ): "

def run_framework():
    output = []
    output.append("Framework execution started")
    output.append(base_369_framework())
    output.append(fibonacci_sequence())
    output.append(golden_ratio())
    output.append("Framework execution completed")
    return "\n".join(output)

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/run_framework')
def run_framework_route():
    return jsonify(run_framework())

@app.route('/ai_response')
def ai_response():
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Describe the significance of the golden ratio in nature.",
        max_tokens=150
    )
    return jsonify(response.choices[0].text.strip())

@app.route('/spirit_interact', methods=['POST'])
def spirit_interact():
    user_input = request.json.get('input')
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=user_input,
        max_tokens=150
    )
    return response.choices[0].text.strip()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
