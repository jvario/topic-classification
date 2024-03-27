import logging
from json import dumps

from flask import Response

logger = logging.getLogger('Topic CLF MS')

def log_request(request):
    msg = f"New request {str(request)} with headers:\n {request.headers} "
    if request.is_json:
        msg += f"and body {request.json}"
    logger.debug(msg)



def response_object(doc, status_code):
    if status_code == 200:
        return Response(dumps(doc), mimetype="application/json", status=status_code)
    else:
        return Response(dumps({"error": doc}), mimetype="application/json", status=status_code)