FROM golang:1.23.1 AS builder-base

FROM debian:bookworm AS release-base

FROM builder-base AS builder
COPY . /app
WORKDIR /app
RUN go mod download
RUN go build -o /entrypoint

FROM release-base AS release
WORKDIR /
COPY --from=builder /entrypoint /entrypoint
COPY config.toml /config.toml
EXPOSE 5138
ENTRYPOINT ["/entrypoint"]
