import uvicorn

from app.main import app

if __name__ == "__main__":
    from app.utils.category_classes import *

    uvicorn.run(app, host="0.0.0.0", port=8000, log_config="log.ini")
