# Records docs/demo/http.gif from a real run: `vhs docs/demo/http.tape` from the repo root.
Output docs/demo/http.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 440
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate` Enter
Sleep 2s
Type `python -m hyperparameter_optimizer serve --port 18315 >/dev/null 2>&1 &` Enter
Sleep 2s
Sleep 8.0s
Type `clear` Enter
Sleep 1s
Show
Type `curl -s localhost:18315/health; echo` Enter
Sleep 1.5s
Type `curl -s localhost:18315/optimize -d '{"benchmark": "sphere", "sampler": "tpe", "trials": 30, "seed": 0}' | jq '{benchmark, sampler, trials, best_value, simple_regret, best_configuration}'` Enter
Sleep 4s
Hide
Type `kill %1` Enter
Sleep 1s
