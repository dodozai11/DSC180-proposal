from solution import fizzbuzz_list


def test_r1_numbers_as_strings():
    assert fizzbuzz_list(2) == ["1", "2"]
    assert len(fizzbuzz_list(7)) == 7
    assert fizzbuzz_list(7)[6] == "7"


def test_r2_fizz():
    assert fizzbuzz_list(3)[2] == "Fizz"
    assert fizzbuzz_list(9)[8] == "Fizz"


def test_r3_buzz():
    assert fizzbuzz_list(5)[4] == "Buzz"
    assert fizzbuzz_list(10)[9] == "Buzz"


def test_r4_fizzbuzz():
    assert fizzbuzz_list(15)[14] == "FizzBuzz"
    assert fizzbuzz_list(30)[29] == "FizzBuzz"


def test_r5_non_positive_n():
    assert fizzbuzz_list(0) == []
    assert fizzbuzz_list(-3) == []
