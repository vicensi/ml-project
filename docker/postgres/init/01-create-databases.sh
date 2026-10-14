#!/bin/bash
# Runs ONCE, when the postgres data volume is first initialised (docker-entrypoint-initdb.d).
# To re-run: docker compose down -v   (this DELETES all data).
#
# Isolation model: one server, one database + one login role per application, so Airflow,
# MLflow, Feast and the app can never collide on table names or share credentials.
set -euo pipefail

psql_admin() { psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" "$@"; }

create_role_and_db() {
  local role="$1" pass="$2" db="$3"
  psql_admin --dbname postgres -v pass="$pass" <<SQL
CREATE ROLE "$role" LOGIN PASSWORD :'pass';
CREATE DATABASE "$db" OWNER "$role";
REVOKE ALL ON DATABASE "$db" FROM PUBLIC;
SQL
}

create_role_and_db "$APP_DB_USER"     "$APP_DB_PASSWORD"     "$APP_DB_NAME"
create_role_and_db "$MLFLOW_DB_USER"  "$MLFLOW_DB_PASSWORD"  "$MLFLOW_DB_NAME"
create_role_and_db "$AIRFLOW_DB_USER" "$AIRFLOW_DB_PASSWORD" "$AIRFLOW_DB_NAME"
create_role_and_db "$FEAST_DB_USER"   "$FEAST_DB_PASSWORD"   "$FEAST_DB_NAME"

# integration-test database, owned by the app role (so tests never touch real data)
psql_admin --dbname postgres <<SQL
CREATE DATABASE "$TEST_DB_NAME" OWNER "$APP_DB_USER";
SQL

# schemas inside the application database (and its test twin)
for db in "$APP_DB_NAME" "$TEST_DB_NAME"; do
  psql_admin --dbname "$db" <<SQL
CREATE SCHEMA IF NOT EXISTS raw        AUTHORIZATION "$APP_DB_USER";
CREATE SCHEMA IF NOT EXISTS features   AUTHORIZATION "$APP_DB_USER";
CREATE SCHEMA IF NOT EXISTS serving    AUTHORIZATION "$APP_DB_USER";
CREATE SCHEMA IF NOT EXISTS monitoring AUTHORIZATION "$APP_DB_USER";
SQL
done
echo "databases and schemas created"
