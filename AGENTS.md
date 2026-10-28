# AGENTS.md

AnyPS5 relinks PS5 executables into native Linux or Windows programs that run against reimplemented system libraries; it is not an emulator.

## Communication

- Communicate with the human user in Russian. Keep code, commands and technical identifiers in English.
- Before tools for a non-trivial task, confirm the task and next 2-3 actions in a visible message. State material risks or uncertainties.
- Report meaningful progress and changes of approach, normally after 3-5 tool calls. Do not replace updates with tool output or repeat generic status messages.
- End with a concise report of completed work, verification and unresolved issues. Never claim success without evidence.
- Write subagent tasks and reports in clear English using ASD-STE100 style. The orchestrator summarizes their findings for the user.

## Scope and authorization

- Follow the user's current task scope. Existing authorization for that task persists; skills do not grant tool permissions.
- Inspect existing tools and dependencies before proposing installation. Never install software or package managers without approval.
- Heavy builds, cache clearing and game launches require user authorization. Do not interrupt other experiments.
- `git push`, `git reset --hard` and `git checkout` require explicit approval.
- Do not change global FreeLLMAPI configuration. Project harness changes belong under `.opencode/` and require task authorization.
- Do not use computer-use or browser/UI automation without explicit approval in the current conversation.

## Python and Windows tools

- Every Python command must use `D:\Python\Python311\python.exe` explicitly. Prefer `-B` to avoid temporary bytecode files. Do not use bare `python`, `python3` or `py`.
- Existing tools are in `C:\msys64\ucrt64\bin`: `cmake.exe`, `ctest.exe`, `ninja.exe`, `gcc.exe`, `g++.exe`. Verify their paths; absence from PATH does not mean they are missing. CTest comes with CMake.
- Existing build directories may use the older toolchain under `S:\AnyPS5\tools\mingw64\bin`. Inspect the chosen build's CMake cache and retain its configured toolchain. Do not mix compiler/runtime libraries or change global PATH automatically.
- Follow [BUILD](docs/dev/BUILD.md) for project builds. A relinker-only change uses `-DANYPS5_RELINKER_ONLY=ON`.

## Skills

- At the start of a substantive task, inspect installed skill descriptions and use the pre-approved `find-skills` to assess candidates.
- `find-skills` may search and assess, but must not install or invoke a candidate skill without explicit user approval.
- For development, build, test and debugging tasks, prefer the project skill `anyps5-development` after approval. Do not load unrelated skills.
- For a partial fit, explain the limits and required adaptation before requesting approval. Review third-party skill instructions and scripts before approved execution; popularity is not a security review.
- The canonical project skill is `.opencode/skills/anyps5-development/SKILL.md`. `.agents/skills/anyps5-development` is a local junction for Codex discovery, not a second editable copy. Preserve the junction and do not commit its contents twice.

## Roles and harness

- `orchestrator` plans, coordinates, verifies results and communicates with the user. It maintains project instructions, authorized harness configuration and external journals; it does not make direct production-code changes.
- `investigator` investigates read-only. `reviewer` reviews read-only. `developer` is the only role that writes assigned production code.
- Use no more than three active agents, including the orchestrator. Delegate useful independent work, not trivial tasks. Do not allow conflicting edits.
- Verify subagent evidence before integration. An LLM report is not proof of a successful build, test or measurement.
- Models, modes and permissions are defined in `.opencode/agents/*.md`; do not duplicate mutable model tables here. Model-pool and decision configuration belongs in the corresponding `.opencode/` JSON files.
- Preserve working plugins, harness helpers, harness tests and local dependencies. A script's name alone does not establish that it is disposable.

## Persistent state and workspace hygiene

- Read `D:\AnyPS5-Logs\agent-state\PROJECT_STATE.md` and `D:\AnyPS5-Logs\agent-state\EXPERIMENTS.md` before work. These are the only canonical project journals; `opencode.json` references them.
- Before a new hypothesis, check prior experiments. Record the hypothesis, commit, parameters, result, evidence paths and limits of the conclusion. Do not repeat disproved experiments without new grounds.
- Only the orchestrator updates the journals. Update state after significant changes and before completion; preserve concurrent entries and distinguish facts from hypotheses and unverified claims.
- Do not create journal copies under `.opencode/`, duplicate instruction files, or place one-off scripts, reports and backups in the repository root.
- Keep local logs, reports, temporary fixtures and recoverable archives under `D:\AnyPS5-Logs\`. Do not store secrets or full logs in the journals. Investigation material for upstream belongs in the PR.
- Archive obsolete local material with its original relative paths and checksums before removing it from the worktree. Check references and preserve active harness files.

## Project rules

The rules are [CONTRIBUTING.md](CONTRIBUTING.md) and [CONVENTIONS](docs/dev/CONVENTIONS.md).

- No comments in code, except `#endif` and namespace ends.
- An export you cannot implement calls `NotImplemented_nid_no_patch(__func__)`; never return a made-up success.
- Implement the function for every input. Never special-case a value, ID or path seen in one title's trace. If general behaviour is unknown, throw and record it in `docs/dev/debt/<library>.md`.
- New third-party code is a submodule at `3rdparty/<name>`: add its `.gitmodules` entry with `git submodule add <url> 3rdparty/<name>` and build it from source with CMake. Never `find_package` or copy sources in.
- Known gaps go in `docs/dev/debt/<library>.md`; notes, reports and logs go in the pull request. Add no other project files for temporary investigation material.
- Commit subjects and PR titles are Conventional Commits.
- PR body: copy [the template](.github/pull_request_template.md), keep every line, tick every box, and fill `What`, `Tested`, `Depends on` (#N or none), `Unblocks` (title ID and discussion, or none), `GPUs tested` (AGC, shader recompiler, VideoOut; otherwise n/a) and `AI-assisted` (yes or no).
- Before an authorized push, run `D:\Python\Python311\python.exe -B tools/check_conventions.py --base origin/main`; before opening a PR, run `D:\Python\Python311\python.exe -B tools/pr_overlap.py --branch <branch>`.
- A speed claim shows `tools/trace_digest.py` summaries of `main` and the change from the same machine and scene. Synthetic measurements do not establish game FPS.
- AGC, shader recompiler and VideoOut changes run the same focused tests on `main` and the change with `ANYPS5_REQUIRE_VULKAN=1`, `ANYPS5_NO_SHADER_CACHE=1` and `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`, using `ctest --test-dir build -R <pattern> -V` on the same machine and GPU.
- Confirm neither validation log contains `was not found but was requested by env var VK_INSTANCE_LAYERS`, and compare logs for new `VUID-` errors. CTest can report `Passed` despite validation errors; skipped tests are not passes.
