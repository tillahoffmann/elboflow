.PHONY : tests

NOTEBOOKS = $(wildcard examples/*.ipynb)
NOTEBOOK_OUTPUTS = $(NOTEBOOKS:.ipynb=.html)

notebooks : $(NOTEBOOK_OUTPUTS)

$(NOTEBOOK_OUTPUTS) : %.html : %.ipynb
	jupyter nbconvert --execute $@ $<

tests :
	py.test -v --cov elboflow --cov-report html -rsx
