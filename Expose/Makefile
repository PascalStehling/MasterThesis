TARGET = thesis
LATEX = lualatex
FLAGS = -synctex=1 -shell-escape

.PHONY: all tidy clean

all: $(TARGET).pdf

$(TARGET).pdf: fiwthesis.cls $(TARGET).tex anlagen/ kapitel/ verzeichnisse/ quellen.bib
	$(LATEX) $(FLAGS) $(TARGET)
	biber $(TARGET)
	makeglossaries $(TARGET)
	$(LATEX) $(FLAGS) $(TARGET)
	$(LATEX) $(FLAGS) $(TARGET)
	$(LATEX) $(FLAGS) $(TARGET)

tidy:
	rm -rf $(TARGET).acn $(TARGET).acr $(TARGET).alg $(TARGET).bbl \
		  $(TARGET).glg $(TARGET).glo $(TARGET).gls $(TARGET).ist \
		  $(TARGET).aux $(TARGET).blg $(TARGET).idx $(TARGET).ilg \
		  $(TARGET).ind $(TARGET).lof $(TARGET).lot $(TARGET).out \
		  $(TARGET).lol $(TARGET).run.xml $(TARGET).slg $(TARGET).syg \
		  $(TARGET).syi $(TARGET).nlo $(TARGET).bcf $(TARGET).log \
		  $(TARGET).toc _minted-$(TARGET) $(TARGET).loa \
		  $(TARGET).synctex.gz

clean: tidy
	rm -f $(TARGET).pdf
