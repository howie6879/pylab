class Game:
    def __init__(self):
        self._current_frame, self._current_throw, self._score = 1, 0, 0
        self._is_first_throw = True
        self._throws = {}

    @property
    def current_frame(self) -> int:
        return self._current_frame

    @property
    def score(self) -> int:
        return self.score_for_frame(self._current_frame - 1)

    def add(self, pins: int):
        self._throws[self._current_throw] = pins
        self._current_throw += 1
        self._score += pins
        self.djust_current_frame(pins)

    def djust_current_frame(self, pins: int):
        if self._is_first_throw:
            if pins == 10:
                self._current_frame += 1
            else:
                self._is_first_throw = False
        else:
            self._is_first_throw = True
            self._current_frame += 1

        if self._current_frame > 11:
            self._current_frame = 11

    def score_for_frame(self, the_frame: int) -> int:
        ball, score = 0, 0
        for _ in range(the_frame):
            first_throw = self._throws[ball]
            ball += 1
            if first_throw == 10:
                score += 10 + self._throws[ball] + self._throws[ball + 1]
            else:
                second_throw = self._throws[ball]
                ball += 1
                frame_score = first_throw + second_throw
                if frame_score == 10:
                    score += frame_score + self._throws[ball]
                else:
                    score += frame_score

        return score


class Frame:
    def __init__(self):
        self._score = 0

    def add(self, pins: int):
        self._score += pins

    @property
    def score(self) -> int:
        return self._score
