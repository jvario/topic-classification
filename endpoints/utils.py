import logging
from json import dumps

from flask import Response

logger = logging.getLogger('Topic CLF MS')

def log_request(request):
    msg = f"New request {str(request)} with headers:\n {request.headers} "
    if request.is_json:
        msg += f"and body {request.json}"
    logger.debug(msg)


def error_response(status_code, message, debug=None):
    logger.error(msg=message)
    if not debug:
        debug = message

    error = Error(code=status_code, message=str(message), debug=str(debug))

    return Response(dumps(error.to_dict()), mimetype="application/json", status=status_code)


def response_object(doc, status_code):
    if status_code == 200:
        return Response(dumps(doc), mimetype="application/json", status=status_code)
    else:
        return error_response(status_code, str(doc))