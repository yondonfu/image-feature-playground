from fastapi import Request


def get_pipeline(request: Request):
    return request.app.pipeline
