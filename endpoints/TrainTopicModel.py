import logging

from flask import request
from flask_restful import Resource

from endpoints.topic_app_call import train_evaluate_model
from endpoints.utils import log_request, response_object

logger = logging.getLogger('Topic CLF MS')


class TrainTopicModel(Resource):
    """
    POST /topic_clf/TrainTopicModel
    Train topic classification model
    """

    @staticmethod
    def post():
        log_request(request)
        logger.info(f"Endpoint: POST /TrainTopicModel/TrainTopicModel Triggered with request {request}.")

        response, status_code = train_evaluate_model()

        return response_object(status_code=status_code, doc=response)
