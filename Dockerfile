FROM rust:1.99-slim-bookworm AS build
WORKDIR /src
COPY . .
RUN cargo build --release -p vecengine-server

FROM debian:bookworm-slim
RUN useradd --system --uid 10001 vecengine && mkdir /data && chown vecengine /data
COPY --from=build /src/target/release/vecengine-server /usr/local/bin/vecengine-server
USER vecengine
ENV VECENGINE_DATA=/data VECENGINE_ADDR=0.0.0.0:8080
VOLUME /data
EXPOSE 8080
ENTRYPOINT ["vecengine-server"]
