from datetime import datetime
import pytz
from flask import Flask, jsonify
from flask_restful import Api

from endpoints.LoadTopicModel import LoadTopicModel
from endpoints.TrainTopicModel import TrainTopicModel, logger
from endpoints.utils import error_response

app = Flask(__name__)
api = Api(app)
pathRoot = "/topic_clf"




api.add_resource(TrainTopicModel, pathRoot + "/TrainTopicModel", endpoint="TrainTopicModel")
api.add_resource(LoadTopicModel, pathRoot + "/LoadTopicModel", endpoint="LoadTopicModel")

release_date = str(datetime.now(tz=pytz.timezone("Europe/Athens")))


@app.route(f'/{pathRoot}/health', methods=['GET'])
def health():
    logger.info("Health endpoint Triggered.")
    return jsonify({"topic_classification service, release_date": release_date})


@app.errorhandler(Exception)
def handle_exceptions(e):
    logger.exception(msg=str(e))
    msg = "Internal Error"
    return error_response(status_code=500, message=msg)


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5000)
