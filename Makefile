.PHONY: score score-local

# The README tables, in a clean container, from the committed forecasts in data/.
score:
	docker build -t windpfn .
	docker run --rm windpfn

score-local:
	windpfn-score
