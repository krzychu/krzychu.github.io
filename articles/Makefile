all: slop.html residuals_1_intro.html residuals_2_language_divergence.html

slop.html: template.html slop.md
	pandoc slop.md -o content.html
	m4 template.html > slop.html
	rm content.html

residuals_1_intro.html: template.html residuals_1_intro.md
	pandoc residuals_1_intro.md -o content.html
	m4 template.html > residuals_1_intro.html
	rm content.html

residuals_2_language_divergence.html: template.html residuals_2_language_divergence.md
	pandoc residuals_2_language_divergence.md -o content.html
	m4 template.html > residuals_2_language_divergence.html
	rm content.html

clean:
	rm -f slop.html residuals_1_intro.html
