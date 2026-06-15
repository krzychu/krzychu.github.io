all: countdown.js

countdown.js: countdown.ts
	tsc countdown.ts

clean:
	rm -f countdown.js
