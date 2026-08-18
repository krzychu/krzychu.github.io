all: slop.html

slop.html: template.html slop.md
	pandoc slop.md -o content.html
	m4 template.html > slop.html
	rm content.html

clean:
	rm -f slop.html
