FROM python:3.11-slim

WORKDIR /app

# Allow choosing a requirements file at build time (default to requirements.docker.txt)
ARG REQUIREMENTS=requirements.docker.txt
COPY ${REQUIREMENTS} ./requirements.txt

# Configurable PyPI index and longer pip timeout for slow/proxied networks
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_DEFAULT_TIMEOUT=100

RUN pip install --no-cache-dir --index-url ${PIP_INDEX_URL} -r requirements.txt

COPY . .

# Do not hardcode the exposed port; runtime platforms (Render) may provide $PORT.
EXPOSE 8502

# Ensure Streamlit reads the runtime port by setting STREAMLIT_SERVER_PORT
# from $PORT before starting the server. Use sh -c so expansion happens at runtime.
CMD ["sh", "-c", "STREAMLIT_SERVER_PORT=${PORT:-8502} streamlit run app.py --server.address 0.0.0.0"]