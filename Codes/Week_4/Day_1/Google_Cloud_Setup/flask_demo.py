from flask import Flask, jsonify, request

app = Flask(__name__)

# Sample endpoint - GET request
@app.route('/hello', methods=['GET'])
def hello_world():
    return "Hello, Flask API!"

@app.route('/greet', methods=['POST'])
def greet_user():
    data = request.get_json()
    name = data.get("name", "Guest")  # Default to "Guest" if no name provided
    return jsonify({"message": f"Hello, {name}!"})

'''Check with this command curl -X POST http://34.68.111.218:8080/greet \
     -H "Content-Type: application/json" \
     -d '{"name": "Ashish"}'''

# Running the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
