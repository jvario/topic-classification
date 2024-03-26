import logging

from flask import request
from flask_restful import Resource

from endpoints.topic_app_call import train_evaluate_model, load_model
from endpoints.utils import log_request, response_object

logger = logging.getLogger('Topic CLF MS')


class LoadTopicModel(Resource):
    """
    POST /topic_clf/LoadTopicModel?model_name=""
    Train topic classification model
    """

    @staticmethod
    def post():
        log_request(request)
        logger.info(f"Endpoint: POST /TrainTopicModel/TrainTopicModel Triggered with request {request}.")
        model_name = request.args.get('model_name')
        response, status_code = load_model(model_name)

        return response_object(status_code=status_code, doc=response)
