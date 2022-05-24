# Author Alphée GROSDIDIER
CC=pdflatex
CFLAGS=--synctex=1 -interaction=nonstopmode
TARGETS=Recherche_Documentaire_completion_automatique.pdf

all: ${TARGETS}

%.pdf: %.tex
	${CC} ${CFLAGS} $^
	biber $(basename $^)
	${CC} ${CFLAGS} $^
	${CC} ${CFLAGS} $^
	make clean

clean:
	rm -f *.log *.toc *.out *.aux *.bfc *.xml *.bak *.synctex.gz *.bcf *.blg *.bbl

mrproper: clean
	rm -f *.pdf
