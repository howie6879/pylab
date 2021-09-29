import pytest

from bowling import Frame, Game


# def test_one_throw():
#     game = Game()
#     game.add(5)
#     assert game.score == 5
#     assert game.current_frame == 1


def test_two_throws_no_mark():
    game = Game()
    game.add(5)
    game.add(4)
    assert game.score == 9
    assert game.current_frame == 2


def test_four_throws_no_mark():
    game = Game()
    game.add(5)
    game.add(4)
    game.add(7)
    game.add(2)

    assert game.score == 18
    assert game.score_for_frame(1) == 9
    assert game.score_for_frame(2) == 18
    assert game.current_frame == 3


def test_simple_spare():
    game = Game()
    game.add(3)
    game.add(7)
    game.add(3)
    assert game.score_for_frame(1) == 13
    assert game.current_frame == 2


def test_simple_frame_after_spare():
    game = Game()
    game.add(3)
    game.add(7)
    game.add(3)
    game.add(2)
    assert game.score_for_frame(1) == 13
    assert game.score_for_frame(2) == 18
    assert game.score == 18
    assert game.current_frame == 3


def test_simple_strike():
    game = Game()
    game.add(10)
    game.add(3)
    game.add(6)
    assert game.score_for_frame(1) == 19
    assert game.score_for_frame(2) == 28
    assert game.current_frame == 3


def test_perfect_game():
    game = Game()
    for _ in range(12):
        game.add(10)
    assert game.score == 300
    assert game.current_frame == 11


def test_end_of_array():
    game = Game()
    for _ in range(9):
        game.add(0)
        game.add(0)
    game.add(2)
    game.add(8)
    game.add(10)
    assert game.score == 20


def test_sample_game():
    game = Game()
    game.add(1)
    game.add(4)
    game.add(4)
    game.add(5)
    game.add(6)
    game.add(4)
    game.add(5)
    game.add(5)
    game.add(10)
    game.add(0)
    game.add(1)
    game.add(7)
    game.add(3)
    game.add(6)
    game.add(4)
    game.add(10)
    game.add(2)
    game.add(8)
    game.add(6)
    assert game.score == 133


def test_heart_break():
    game = Game()
    for _ in range(11):
        game.add(10)
    game.add(9)
    assert game.score == 299


def test_tenth_frame_spare():
    game = Game()
    for _ in range(9):
        game.add(10)
    game.add(9)
    game.add(1)
    game.add(1)
    assert game.score == 270


def test_frame():
    frame = Frame()
    frame.add(5)
    assert frame.score == 5
