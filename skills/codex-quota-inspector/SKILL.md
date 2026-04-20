---
name: codex-quota-inspector
description: "Проверка remaining quota у Codex-аккаунтов из auth.json через ChatGPT endpoint /backend-api/wham/usage."
version: 0.1.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [codex, quota, auth.json, wham, usage, chatgpt]
---

# codex-quota-inspector

Используй этот skill, когда нужно проверить remaining quota у Codex-аккаунтов из локального `auth.json`.

## Что делает

Skill использует script `scripts/check_codex_quotas.py`, который:
- читает `auth.json`;
- находит записи с provider/type `codex`;
- извлекает `chatgpt_account_id` и `plan_type` из `id_token`;
- при необходимости делает refresh access token через `https://auth.openai.com/oauth/token`;
- запрашивает quota из `https://chatgpt.com/backend-api/wham/usage`;
- показывает окна quota `5h` и `7d`;
- по умолчанию печатает компактный человекочитаемый summary по аккаунтам;
- умеет печатать JSON.

## Когда применять

- пользователь просит посмотреть remaining quota у Codex-аккаунтов;
- есть локальный `auth.json` с несколькими аккаунтами;
- нужен переиспользуемый script, который потом можно коммитить в repo.

## Основной запуск

Для машинного разбора и последующего красивого summary запускай:

```bash
python3 ~/.hermes/skills/workflow/codex-quota-inspector/scripts/check_codex_quotas.py /path/to/auth.json --json
```

Если путь не указан, script пытается читать `./auth.json`.

## Человекочитаемый запуск

```bash
python3 ~/.hermes/skills/workflow/codex-quota-inspector/scripts/check_codex_quotas.py /path/to/auth.json
```

## Полезные флаги

```bash
--json                 # JSON output
--ascii                # ASCII fallback for borders/bars
--timeout 30           # HTTP timeout
--concurrency 8        # parallel workers
--force-refresh        # refresh token before quota request
--write-back           # сохранить обновлённые токены обратно в auth.json
--wham-url URL         # override для quota endpoint
--token-url URL        # override для refresh endpoint
```

## Как использовать в ответе пользователю

1. Запусти script с `--json`.
2. Разбери JSON.
3. Суммаризируй:
   - сколько аккаунтов всего;
   - сколько exhausted / low / ok;
   - для каждого аккаунта: `name/email`, `plan`, `5h`, `7d`, `reset`, `status`.
4. Если есть ошибки, показывай их отдельно.

## Интерпретация статусов

- `ok` — quota есть;
- `low` — хотя бы одно окно близко к исчерпанию;
- `exhausted` — хотя бы одно окно exhausted;
- `error` — не удалось refresh/query/parse.

## Заметки

- script старается не трогать файл, если не передан `--write-back`;
- если access token истёк, script пытается refresh и повторяет quota request;
- формат `auth.json` может отличаться, поэтому script поддерживает несколько layout'ов: один объект, список объектов, вложенные коллекции (`files`, `auths`, `entries`, `accounts`), а также Hermes-style `credential_pool.openai-codex`;
- на практике `chatgpt_account_id` и `chatgpt_plan_type` могут лежать не только в `id_token`, но и прямо в JWT payload у `access_token`; script сначала пытается читать оба варианта;
- для Hermes-style pool имя аккаунта часто удобнее брать из `label`, а provider определять по ключу пула (`openai-codex`) или по `base_url`.
