This is our BCG INRIA workingspace

build package
```
pip install -e .
```

run api (requires google maps api key)
```
uvicorn inria.api.fast:app
```

run webserver
```
streamlit run inria/webinterface/webapp.py
```
