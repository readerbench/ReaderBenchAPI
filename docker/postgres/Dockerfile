FROM postgres:12

COPY ./scripts/* /docker-entrypoint-initdb.d/
ENV POSTGRES_HOST_AUTH_METHOD=trust
