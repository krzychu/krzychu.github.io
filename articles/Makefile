all: slop.html \
	residuals_1_intro.html \
	residuals_2_language_divergence.html \
	residuals_3_refusal_mitigation.html

slop.html: template.html slop.md
	pandoc slop.md -o content.html
	m4 template.html > slop.html
	rm content.html

residuals_1_intro.html: template.html residuals_1_intro.md
	pandoc --mathml residuals_1_intro.md -o content.html
	m4 template.html > residuals_1_intro.html
	rm content.html

residuals_2_language_divergence.html: template.html residuals_2_language_divergence.md
	pandoc --mathml residuals_2_language_divergence.md -o content.html
	m4 template.html > residuals_2_language_divergence.html
	rm content.html

residuals_3_refusal_mitigation.html: template.html residuals_3_refusal_mitigation.md
	pandoc --mathml residuals_3_refusal_mitigation.md -o content.html
	m4 template.html > residuals_3_refusal_mitigation.html
	rm content.html

clean:
	rm -f \
		slop.html \
		residuals_1_intro.html \
		residuals_2_language_divergence.html \
		residuals_3_refusal_mitigation.html 
