#pragma once
#include <iostream>
#include <vector>
#include <functional>
#include <stdexcept>
#include <hls_vector.h>

#define TEST(name) { #name, name }

using TestFunc = std::function<void()>;

struct TestEntry {
	const char* name;
	TestFunc func;
};

class TestRunner {
public:
	void runAllTests(const std::vector<TestEntry>& tests) {
		int passed = 0;
		int failed = 0;

		for (TestEntry test : tests) {
			try {
				test.func();
				std::cout << "[PASS] Test " << test.name << "\n";
				passed++;
			}
			catch (const std::exception& e) {
				std::cout << "[FAIL] Test " << test.name << " : " << e.what() << "\n";
				failed++;
			}
		}

		std::cout << "\n=== Test Summary ===\n";
		std::cout << "Passed: " << passed << "\n";
		std::cout << "Failed: " << failed << "\n";
	}
};

namespace Assertions {

	template <typename T, int N>
	const T* arr(const T (&a)[N]) {
		return a;
	}

	void assertTrue(bool condition) {
		if (!condition) {
			throw std::runtime_error("Assertion failed");
		}
	}

	void assertFalse(bool condition) {
		if (condition) {
			throw std::runtime_error("Assertion failed");
		}
	}

	void assertEquals(int expected, int actual) {
		if (expected != actual) {
			throw std::runtime_error(
				"Assertion failed: expected=" + std::to_string(expected)
				+ ", actual=" + std::to_string(actual)
			);
		}
	}

	template <typename ET, typename AT>
	void assertArrayEqualsImpl(const ET& expected, const AT& actual, int size) {
		for (int i = 0; i < size; i++) {
			if (!(expected[i] == actual[i])) {
				throw std::runtime_error(
					"array contents differ at index [" + std::to_string(i) +
					"], expected:<" + std::to_string(expected[i]) +
					"> but was:<" + std::to_string(actual[i]) + ">"
				);
			}
		}
	}

	template <typename T, size_t N>
	void assertArrayEquals(const std::initializer_list<T>& expected, const hls::vector<T, N>& actual) {
		if (expected.size() != N) {
			throw std::runtime_error("array size mismatch");
		}
		assertArrayEqualsImpl(expected.begin(), actual, N);
	}

	template<typename V>
	static void print_vector(const V& v) {
		const int N = v.size();
		for (int i = 0; i < N; i++) {
			std::cout << v[i];
			if (i != N - 1) std::cout << ", ";
		}
		std::cout << std::endl;
	}
};
