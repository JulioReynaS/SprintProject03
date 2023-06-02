FROM python:3.8.13 as base

ARG UID=1000
ARG GID=1000

# Add a non-root user
RUN addgroup --gid $GID app
RUN adduser --disabled-login --geco '' --uid $UID --gid $GID app
USER app

# Append /home/app/.local/bin/ to PATH variable because
# gunicorn is installed there.
ENV PYTHONPATH=$PYTHONPATH:/src/
ENV PATH=$PATH:/home/app/.local/bin/

COPY --chown=app:app ./ /src
#COPY ./ /src

WORKDIR /src

RUN pip install --upgrade pip && pip install -r requirements.txt

FROM base as test
RUN ["pytest", "-v", "/src/tests"]

FROM base as build
ENTRYPOINT ["gunicorn", "--workers=8", "--bind", "0.0.0.0:5000", "--timeout", "300", "app:app"]
