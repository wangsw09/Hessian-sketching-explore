CURR_DIR = "C:\Users\Shuaiwen\AppData\Local\lxss\home\wangsw09\work\Proj-Hessian-sketch\slice\"
CURR_FILE = class_report
READER_PATH = /mnt/c/Program\ Files/SumatraPDF/SumatraPDF.exe

.PHONY: curr

clean_pre:
	rm -rf *.log *.out *.aux *.bbl *.blg *~ .*~

clean_post:
	rm -rf *.log *.out *.aux *.bbl *.blg *.snm *.nav *.toc

compile_pdf1:
	pdflatex $(CURR_FILE).tex

compile_pdf2:
	pdflatex $(CURR_FILE).tex

compile_pdf3:
	pdflatex $(CURR_FILE).tex

compile_bib:
	bibtex $(CURR_FILE).aux

open_sumatra:
	$(READER_PATH) "$(patsubst "%",%,$(CURR_DIR))$(CURR_FILE).pdf"

curr: clean_pre compile_pdf1 compile_bib compile_pdf2 compile_pdf3 clean_post open_sumatra

nobib: clean_pre compile_pdf1 compile_pdf2 clean_post open_sumatra

