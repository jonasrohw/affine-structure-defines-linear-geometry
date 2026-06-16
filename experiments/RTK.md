# RTK - Rust Token Killer (Codex CLI)

**Usage**: Token-optimized CLI proxy for shell commands.

## Rule

Always prefix shell commands with `rtk`.

In this Codex workspace, set RTK tracking to the project-local DB so sandboxed commands can write analytics:

```bash
RTK_DB_PATH=.rtk/history.db rtk <command>
```

Examples:

```bash
RTK_DB_PATH=.rtk/history.db rtk git status
RTK_DB_PATH=.rtk/history.db rtk cargo test
RTK_DB_PATH=.rtk/history.db rtk npm run build
RTK_DB_PATH=.rtk/history.db rtk pytest -q
```

## Meta Commands

```bash
RTK_DB_PATH=.rtk/history.db rtk gain            # Token savings analytics
RTK_DB_PATH=.rtk/history.db rtk gain --history  # Recent command savings history
RTK_DB_PATH=.rtk/history.db rtk proxy <cmd>     # Run raw command without filtering
```

## Verification

```bash
rtk --version
RTK_DB_PATH=.rtk/history.db rtk gain
which rtk
```
