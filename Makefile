.PHONY : tests

all :
	echo "Configure your own targets here."

tests :
	py.test -v --cov elboflow --cov-report html
