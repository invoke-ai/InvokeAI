---
title: Database
---

# Invoke's SQLite Database

Invoke uses a SQLite database to store image, workflow, model, and execution data.

We take great care to ensure your data is safe, by utilizing transactions and a database migration system.

Even so, when testing an prerelease version of the app, we strongly suggest either backing up your database or using an in-memory database. This ensures any prelease hiccups or databases schema changes will not cause problems for your data.

## Database Backup

Backing up your database is very simple. Invoke's data is stored in an `$INVOKEAI_ROOT` directory - where your `invoke.sh`/`invoke.bat` and `invokeai.yaml` files live.

To back up your database, copy the `invokeai.db` file from `$INVOKEAI_ROOT/databases/invokeai.db` to somewhere safe.

If anything comes up during prelease testing, you can simply copy your backup back into `$INVOKEAI_ROOT/databases/`.

## In-Memory Database

SQLite can run on an in-memory database. Your existing database is untouched when this mode is enabled, but your existing data won't be accessible.

This is very useful for testing, as there is no chance of a database change modifying your "physical" database.

To run Invoke with a memory database, edit your `invokeai.yaml` file, and add `use_memory_db: true` to the `Paths:` stanza:

```yaml
InvokeAI:
  Development:
    use_memory_db: true
```

Delete this line (or set it to `false`) to use your main database.
