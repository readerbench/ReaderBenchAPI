from flask import Flask
app = Flask(__name__)
 
@app.route("/api/v1/isalive")
def hello():
    return "Alive"

@app.route("/api/v1/isdead")
def goodbye():
    return "Not dead"
 
if __name__ == "__main__":
    app.run()
