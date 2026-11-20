# Credentials

Only the optional commercial backends need a token: the `http_api` generator and the `llm` prompt
writer. Everything else runs without any account.

## Where tokens live

1. Environment variable `ANIMALWORLD_<NAME>` (upper case), or the bare `<NAME>`.
2. `~/.config/animalworld/credentials.toml` (or `$XDG_CONFIG_HOME/animalworld/`, or
   `$ANIMALWORLD_CONFIG_DIR/`), one line per token: `name = "value"`.

```bash
animalworld credentials set my_image_api      # asks for the value without echo
echo "$TOKEN" | animalworld credentials set my_image_api --stdin
animalworld credentials list                  # names and source only; values are never shown
animalworld credentials unset my_image_api
animalworld credentials path
```

In `project.yaml` you write the NAME, never the value:

```yaml
generate:
  backend: http_api
  options:
    endpoint: https://api.example.com/v1/images
    token_credential: my_image_api
```

## What the package enforces (all covered by tests)

* The credentials file is written with mode 600; if group or others can read it, loading stops and
  tells you to `chmod 600`.
* The credentials file can never be inside a project folder.
* `project.yaml` is refused if a key that looks like a secret (`api_key`, `token`, `password`, ...)
  holds a literal value.
* A token is placed only in the request header of the backend that needs it. It is never written to
  the project folder, the metadata log, a plan (`--dry-run` shows only whether it is available), or an
  error message (`credentials.redact`).
* `animalworld credentials set` never takes the value as a command-line argument (it would end up in
  shell history and the process list).

## Advice

* Do not commit `credentials.toml`; `animalworld init` writes a `.gitignore` that excludes it.
* On shared machines prefer the file (mode 600) over environment variables in job scripts.
* The Anthropic provider of the `llm` writer uses the official SDK's own credential resolution when
  no `token_credential` is given (`ANTHROPIC_API_KEY`, or a profile from `ant auth login`).
* Literature APIs need no key. `literature.contact_email` is empty by default and is sent only if
  you fill it in. The User-Agent is the generic string `animalworld/<version>`.
