.PHONY : tests

NOTEBOOKS = $(wildcard examples/*.ipynb)
NOTEBOOK_OUTPUTS = $(NOTEBOOKS:.ipynb=.html)

examples : $(NOTEBOOK_OUTPUTS)

$(NOTEBOOK_OUTPUTS) : %.html : %.ipynb
	jupyter nbconvert --execute --ExecutePreprocessor.timeout=None --allow-errors $@ $<

tests :
	py.test -v --cov elboflow --cov-report html -rsx
