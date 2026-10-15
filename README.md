# Time Series ML Platform

Plataforma de ML/MLOps ponta a ponta para previsão do próximo valor de séries temporais horárias (dados sintéticos): Feast, MLflow, FastAPI, Evidently, Airflow e Streamlit.

**Estado atual: Fases 1–2 de 8 (fundação + Feature Store).** Veja o que existe e o que falta em [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Nada além da Fase 1 está implementado — este README só documenta o que funciona hoje.

## O que a Fase 1 entrega
- Configuração central tipada (`tsplatform.config`), sem segredos no código.
- Gerador sintético **determinístico** (tendência, sazonalidade diária e semanal, ruído, anomalias, padrões distintos por série).
- Contratos de dados (Pydantic + validação estrutural do DataFrame).
- Persistência no PostgreSQL com **upsert idempotente** e registro de identidade do dataset (hash).
- Logging JSON estruturado com contexto de correlação.
- Docker Compose: PostgreSQL (bancos/roles separados para app, MLflow, Airflow, Feast), Redis, jobs `ingest` e `tests`.

## O que a Fase 2 entrega
- Features temporais **sem vazamento** (`value_lag_{1,2,24,168}`, `rolling_{mean_24,std_24,mean_168}`, `hour_of_day`, `day_of_week`): a linha de *t* só usa dados `< t`.
- Repositório Feast (`feast_repo/`): entity `series`, feature view `series_features` (TTL 6 h), feature service `forecast_features_v1`; offline store PostgreSQL, online store Redis, registry SQL em banco próprio.
- Dataset de treino por **point-in-time join** com divisão cronológica e linhagem (`feature_set_id`, hash).
- Comandos: `build-features`, `feast-apply`, `feast-materialize`, `verify-features`.

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
Pipeline de features (Fase 2), na ordem:
```powershell
docker compose --profile jobs run --rm features python -m tsplatform.cli build-features
docker compose --profile jobs run --rm features python -m tsplatform.cli feast-apply
docker compose --profile jobs run --rm features python -m tsplatform.cli verify-features     # deve FALHAR (exit 1): ainda nada no Redis
docker compose --profile jobs run --rm features python -m tsplatform.cli feast-materialize
docker compose --profile jobs run --rm features python -m tsplatform.cli verify-features     # agora OK
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
| Testes unitários (38) | `pytest` em Python 3.11.15 com as versões de `requirements/dev.txt` |
| Integração PostgreSQL + Feast (11) | Postgres 16-compatível embutido (pgserver) e Redis 6.2.14 reais por TCP: apply, PIT join, TTL, materialização, online, consistência treino×serving |
| Fluxo CLI completo | `01-create-databases.sh` real → ingest → build-features → feast-apply → verify → materialize → verify, em ambiente limpo |
| `docker-compose.yml` | só sintaxe YAML |
| `01-create-databases.sh` | **executado de verdade** (psql) contra Postgres embutido |
| Dockerfile / imagens / Compose | **não construídos nem executados por mim** (Docker indisponível aqui); validados só por sintaxe |

Se `docker compose up` falhar na sua máquina, veja o troubleshooting abaixo e me envie a saída.

## Troubleshooting (Fase 1)
| Sintoma | Causa provável | Ação |
|---|---|---|
| `password authentication failed` após mudar `.env` | roles criados na 1ª inicialização do volume | `docker compose down -v` e subir de novo |
| `exec ... no such file or directory` no postgres init | fim de linha CRLF no `.sh` | `.gitattributes` força LF; rode `git add --renormalize .` |
| `variable is not set` no compose | `.env` ausente | `copy .env.example .env` |
| `psycopg.OperationalError ... SSL was required` | Feast exige TLS por padrão | `APP_DB_SSLMODE=disable` (local); em produção use `require` |
| `verify-features` falha após `feast-materialize` | Redis foi apagado mas o registry ainda diz "materializado" (incremental = no-op) | `feast-materialize --start 2025-06-30T00:00:00` (use um instante anterior ao último de `max(event_timestamp)` das features) |
| `unresolved environment variables` | variável do `feature_store.yaml` ausente | conferir `APP_DB_*`, `FEAST_DB_*`, `REDIS_*` no ambiente do container |
| `ModuleNotFoundError: tsplatform` fora do Docker | pacote fora do path | rode a partir da raiz com `pytest` (já configura `pythonpath`) |

## Próximos passos
Fase 3: baseline + RandomForest com divisão temporal, métricas (MAE, RMSE, WAPE), MLflow tracking/registry e promoção controlada para o alias `champion`.
