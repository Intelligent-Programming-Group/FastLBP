.PHONY: build
build: configure
	cmake --build build --config Release

.PHONY: configure
configure:
	cmake -B build -DCMAKE_BUILD_TYPE=Release -Wno-dev

.PHONY: test
test:
	ctest --test-dir build

.PHONY: clean
clean:
	rm -rf build
