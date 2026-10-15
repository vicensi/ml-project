"""Feast wiring. Importing this package does NOT import feast (lazy), so the lightweight base
image keeps working; use ``tsplatform.feature_store.client`` for real operations."""
