clean:
	-del "build"

build:
	-mkdir "build"
	gcc src/*.c   -Iinc -pg -o build/main -g -O0  

run:
	./build/main