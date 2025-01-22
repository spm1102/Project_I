clean:
	-del "build"

build:
	-mkdir "build"
	gcc src/*.c   -Iinc -o build/main -g -O0  

run:
	./build/main