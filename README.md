# Time Series ML Platform

Plataforma de ML/MLOps ponta a ponta para previsão do próximo valor de séries temporais horárias (dados sintéticos): Feast, MLflow, FastAPI, Evidently, Airflow e Streamlit.

**Estado atual: Fase 1 de 8 (fundação).** Veja o que existe e o que falta em [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Nada além da Fase 1 está implementado — este README só documenta o que funciona hoje.

## O que a Fase 1 entrega
- Configuração central tipada (`tsplatform.config`), sem segredos no código.
- Gerador sintético **determinístico** (tendência, sazonalidade diária e semanal, ruído, anomalias, padrões distintos por série).
- Contratos de dados (Pydantic + validação estrutural do DataFrame).
- Persistência no PostgreSQL com **upsert idempotente** e registro de identidade do dataset (hash).
- Logging JSON estruturado com contexto de correlação.
- Docker Compose: PostgreSQL (bancos/roles separados para app, MLflow, Airflow, Feast), Redis, jobs `ingest` e `tests`.

## Pré-requisitos
- Docker Desktop (Compose v2) **ou** Python 3.11 para rodar sem containers.
- ~1 GB livre de disco para imagens nesta fase (as fases seguintes exigirão bem mais; será documentado).

## Início rápido (Docker)
```powershell
copy .env.example .env          # edite as senhas "change-me-*"
docker compose up -d postgres redis
docker compose --profile jobs run --rm ingest     # gera e persiste os dados (pode repetir: é idempotente)
docker compose --profile test run --rm tests      # unitários + integração contra Postgres real
```
Conferir os dados:
```powershell
docker compose exec postgres psql -U postgres -d tsplatform -c "select count(*) from raw.observations;"
```
> O script `docker/postgres/init` só roda na **primeira** criação do volume. Para recriar bancos/senhas: `docker compose down -v` (apaga os dados).

## Sem Docker (desenvolvimento)
```bash
python3.11 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements/dev.txt
pytest                                                    # integração é ignorada sem TSP_TEST_DATABASE_URL
TSP_ENV_FILE=/dev/null python -m tsplatform.cli info --n-series 2   # identidade do dataset e segmentos, sem banco
```

## Variáveis principais
Todas em `.env.example`. Dados sintéticos: `TSP_GEN_SEED`, `TSP_GEN_N_SERIES`, `TSP_GEN_N_HOURS`, `TSP_GEN_START`, `TSP_GEN_ANOMALY_RATE`. Mesma configuração + mesma seed ⇒ mesmo `dataset_id`.

## Validação realizada (e limites)
| Item | Como foi verificado |
|---|---|
| 27 testes unitários | `pytest` em Python 3.11.15 com as versões de `requirements/dev.txt` |
| Teste de integração PostgreSQL | rodado contra um Postgres embutido (pgserver), via socket; `ingest` executado 2× ⇒ contagem estável |
| `docker-compose.yml` | só sintaxe YAML |
| `01-create-databases.sh` | só `bash -n` |
| Dockerfile / imagens | **não construídos** (Docker indisponível no ambiente de desenvolvimento) |

Se `docker compose up` falhar na sua máquina, veja o troubleshooting abaixo e me envie a saída.

## Troubleshooting (Fase 1)
| Sintoma | Causa provável | Ação |
|---|---|---|
| `password authentication failed` após mudar `.env` | roles criados na 1ª inicialização do volume | `docker compose down -v` e subir de novo |
| `exec ... no such file or directory` no postgres init | fim de linha CRLF no `.sh` | `.gitattributes` força LF; rode `git add --renormalize .` |
| `variable is not set` no compose | `.env` ausente | `copy .env.example .env` |
| `ModuleNotFoundError: tsplatform` fora do Docker | pacote fora do path | rode a partir da raiz com `pytest` (já configura `pythonpath`) |

## Próximos passos
Fase 2: features temporais sem vazamento, repositório Feast, point-in-time join e materialização no Redis.
