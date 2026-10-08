FROM qdrant/qdrant:latest

EXPOSE 6333 6334

VOLUME ["/qdrant/storage"]

CMD ["./qdrant", "--config-path", "/qdrant/config/production.yaml"]
