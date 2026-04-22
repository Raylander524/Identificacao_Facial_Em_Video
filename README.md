# Identificação Facial em Vídeo

O sistema realiza a verificação se as pessoas dos vídeos estão nos arquivos.

## Requisitos

Para rodar o projeto, execute:

```bash
## Banco Vetorial

Este projeto usa Milvus para busca vetorial de rostos com:

- métrica de similaridade: `COSINE`
- limiar de aceite: `> 0.7`
- tipo de índice: `IVF_FLAT`

Para subir com Docker Compose (incluindo Milvus):

	docker compose up -d --build
```

Abra no navegador:

```bash
https://localhost:80
```
