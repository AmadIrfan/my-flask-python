from flask import Flask


app = Flask(__name__)


@app.route("/")
def main():
    return "this is my app"


@app.route("/<ent>")
def main1(ent):
    return f" you entered input : {ent}"


if __name__ == "__main__":
    app.run()
