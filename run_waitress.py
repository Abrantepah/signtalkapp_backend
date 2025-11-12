from waitress import serve
from signtalk_backend.wsgi import application

if __name__ == "__main__":
    serve(
        application,
        host="0.0.0.0",
        port=8000,
        threads=48,            # optimal for 14 cores / 20 logical processors
        connection_limit=5000, # maximum active TCP connections
        backlog=5000           # queue if many requests arrive at once
    )

