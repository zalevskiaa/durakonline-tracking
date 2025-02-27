import numpy as np


STABLE_LIFETIME = 3


class BBox:
    def __init__(self, box: list[int, int, int, int]):
        self.box: list[int, int, int, int] = box

    def iou(self, other: "BBox") -> float:
        x1, y1, x2, y2 = self.box
        ox1, oy1, ox2, oy2 = other.box

        inter_x1 = max(x1, ox1)
        inter_y1 = max(y1, oy1)
        inter_x2 = min(x2, ox2)
        inter_y2 = min(y2, oy2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        area_self = (x2 - x1) * (y2 - y1)
        area_other = (ox2 - ox1) * (oy2 - oy1)

        union_area = area_self + area_other - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    @property
    def x1(self):
        return self.box[0]

    @property
    def y1(self):
        return self.box[1]

    @property
    def x2(self):
        return self.box[2]

    @property
    def y2(self):
        return self.box[3]


class Card:
    rank_keys = {r: k for r, k in zip(
        ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'],
        range(2, 16)
    )}
    suit_keys = {r: k for r, k in zip(
        ['s', 'c', 'd', 'h'],
        range(4)
    )}

    def card_key(self):
        return 100 * Card.rank_keys[self.rank] + Card.suit_keys[self.suit]

    def __init__(self, rank: str, suit: str):
        self.rank = rank
        self.suit = suit

    def __eq__(self, other: "Card"):
        return self.rank == other.rank \
            and self.suit == other.suit

    def __lt__(self, other: "Card"):
        return self.card_key() < other.card_key()

    def __str__(self):
        return f'{self.rank}{self.suit}'

    def __hash__(self):
        return hash(self.rank + self.suit)


class Corner:
    def __init__(self, box: BBox, card: Card):
        self.box = box
        self.card = card

        self.lifetime = 0

    def __str__(self):
        return f'{str(self.card)} ({self.lifetime})'


class Player:
    def __init__(self, box: BBox, istake: bool):
        self.reset(box, istake)

    def reset(self, box: BBox, istake: bool):
        self.box = box
        self.istake = istake
        self.istake_lifetime = 0

        self.stable_istake = False

    def update(self, box: BBox, take: bool):
        self.box = box
        self.istake_lifetime = \
            min(self.istake_lifetime + 1, STABLE_LIFETIME) \
            if self.istake == take else 0

        self.istake = take

        if self.istake_lifetime >= STABLE_LIFETIME:
            self.stable_istake = self.istake

    def get_stable_take(self):
        return self.stable_istake


def match_bboxes(
        boxes: list[BBox],
        next_boxes: list[BBox],
        iou_threshold=0.7
        ) -> dict[int, int]:

    n = len(boxes)
    m = len(next_boxes)

    if n == 0 or m == 0:
        return {}

    iou_mat = np.array([
        [cb.iou(nb) for nb in next_boxes]
        for cb in boxes
    ], dtype=np.float32)

    iou_mat_as = iou_mat.argsort(axis=1)[:, ::-1]

    # print(iou_mat)
    # print(iou_mat_as)

    nexts_taken = np.zeros((m), dtype=bool)
    matched_pairs = {}
    unmatched_curr = set(range(n))
    unmatched_next = set(range(m))
    for cand_cur in range(n):
        for j in range(m):
            cand_next = iou_mat_as[cand_cur, j]
            iou = iou_mat[cand_cur, cand_next]
            # print('iou', iou)

            if iou < iou_threshold:
                # print('break')
                break
            if not nexts_taken[cand_next]:
                nexts_taken[cand_next] = True
                matched_pairs[cand_cur] = cand_next

                unmatched_curr.discard(cand_cur)
                unmatched_next.discard(cand_next)

    return matched_pairs, unmatched_curr, unmatched_next


def argmax(elements: list, key: callable):
    b, bi = None, None
    for i, e in enumerate(elements):
        if b is None or key(e) > key(b):
            bi, b = i, e
    return bi


def argmin(elements: list, key: callable):
    return argmax(elements, lambda e: -key(e))


class CardsTracker:
    def __init__(self):
        self.current_corners: list[Corner] = []
        self.cards_table_static: list[Card] = []
        self.new_corners_empty_lifetime = 0

    def _match_corners(self,
                       next_corners: list[Corner],
                       iou_threshold: float
                       ) -> dict[int, int]:
        """
        update actual corners, remove disappeared corners, add new corners
        """
        n = len(self.current_corners)
        m = len(next_corners)

        if n == 0 or m == 0:
            return {}

        iou_mat = np.array([
            [cc.box.iou(nc.box) for nc in next_corners]
            for cc in self.current_corners
        ], dtype=np.float32)

        iou_mat_as = iou_mat.argsort(axis=1)[:, ::-1]

        # print(iou_mat)
        # print(iou_mat_as)

        nexts_taken = np.zeros((m), dtype=bool)
        matched_pairs = {}
        for cand_cur in range(n):
            for j in range(m):
                cand_next = iou_mat_as[cand_cur, j]
                iou = iou_mat[cand_cur, cand_next]
                # print('iou', iou)

                if iou < iou_threshold:
                    # print('break')
                    break
                if not nexts_taken[cand_next]:
                    nexts_taken[cand_next] = True
                    matched_pairs[cand_cur] = cand_next
        return matched_pairs

    def _update_current_corners(self,
                                next_corners: list[Corner],
                                iou_threshold=0.9
                                ) -> None:
        matched_pairs = self._match_corners(next_corners, iou_threshold)

        # print(matched_pairs)
        # for i, j in matched_pairs.items():
        #     print(self.current_corners[i], next_corners[j])

        # global frames_count
        # frames_count += 1
        # if frames_count > 10:
        #     raise Exception('kek')

        new_current_corners = []
        for i, cur_corner in enumerate(self.current_corners):
            if i in matched_pairs:
                next_corner = next_corners[matched_pairs[i]]

                # print(cur_corner.card, next_corner.card)

                if cur_corner.card == next_corner.card:
                    next_corner.lifetime = min(cur_corner.lifetime + 1,
                                               STABLE_LIFETIME)
                    # print('lifetime: ', next_corner.lifetime)

                new_current_corners.append(next_corner)

        unmatched_nexts = \
            set(range(len(next_corners))) - set(matched_pairs.values())
        for j in unmatched_nexts:
            next_corner = next_corners[j]
            new_current_corners.append(next_corner)

        self.current_corners = new_current_corners

    def _update_table_static(self):
        candidate_table_static = []
        for corner in self.current_corners:
            if corner.lifetime >= STABLE_LIFETIME:
                candidate_table_static.append(corner.card)
        if len(candidate_table_static) >= len(self.cards_table_static):
            self.cards_table_static = candidate_table_static

    def update(self, new_corners, *, hardcode_geom):
        self._update_current_corners(new_corners)
        self._update_table_static()

        self.new_corners_empty_lifetime = (
            min(self.new_corners_empty_lifetime + 1, STABLE_LIFETIME)
            if not new_corners else 0
        )

    def get_table(self) -> list[Card]:
        return self.cards_table_static

    def is_table_disappeared(self) -> bool:
        return self.new_corners_empty_lifetime >= STABLE_LIFETIME \
                and self.cards_table_static


class PlayersTracker:
    def __init__(self, players_num):
        self.players_num = players_num
        self.players: list[Player] = []

    def update(self, next_players: list[Player]):
        if len(next_players) != self.players_num:
            return

        next_p0_index = argmax(next_players, lambda p: p.box.y1)
        next_p0 = next_players.pop(next_p0_index)

        next_players = [next_p0] + sorted(next_players, key=lambda p: p.box.x1)

        if len(self.players) == len(next_players):
            for p, next_p in zip(self.players, next_players):
                p.update(next_p.box, next_p.istake)

        elif len(self.players) == 0:
            self.players = next_players

        else:
            raise ValueError

    def get_who_take_ids(self):
        result = []
        for i, p in enumerate(self.players):
            if p.get_stable_take():
                result.append(i)
        return result


def brs_to_corners(boxes, ranks, suits):
    return [Corner(BBox(b), Card(r, s))
            for b, r, s in zip(boxes, ranks, suits)]


def bt_to_players(
        players_boxes: list[list[int]],
        players_take: list[str],
        ) -> list[Player]:

    result = []
    for box, is_take in zip(players_boxes, players_take):
        x1, y1, x2, y2 = box
        is_take = (is_take == '1')
        p = Player(BBox([x1, y1, x2, y2]), is_take)
        result.append(p)
    return result


class GameState:
    def __init__(self, players_num) -> None:
        self.players_num = players_num

        self.all_ranks: set[str] = {'6', '7', '8', '9', '10',
                                    'J', 'Q', 'K', 'A'}
        self.all_suits: set[str] = {'s', 'c', 'd', 'h'}
        self.all_cards: set[Card] = {Card(r, s)
                                     for r in self.all_ranks
                                     for s in self.all_suits}
        self.all_cards_sorted = sorted(list(self.all_cards))

        self.cards_hist: list[list[Card]] = []
        self.takes_ids_hist: list[list[int]] = []

        self.cards_state: dict[Card, int] = {}
        self.CARDSTATE_OUT = -1
        assert self.CARDSTATE_OUT not in range(self.players_num)

        self.unseen_cards = self.all_cards.copy()

    def reset(self):
        self.cards_hist.clear()
        self.takes_ids_hist.clear()
        self.cards_state.clear()
        self.unseen_cards = self.all_cards.copy()

    def add(self, cards: list[Card], takes_ids: list[int]):
        self.cards_hist.append(cards.copy())
        self.takes_ids_hist.append(takes_ids.copy())

        if len(takes_ids) not in [0, 1]:
            # im sure that players detection error in this case
            # only 1 take-player is possible max
            #
            # todo: something
            return

        cardstate = takes_ids[0] if takes_ids else self.CARDSTATE_OUT
        for card in cards:
            self.cards_state[card] = cardstate

        self.unseen_cards -= set(cards)

    def get_cards_by_cardstate(self):
        result: dict[int, list[Card]] = {}
        all_cardstates = list(range(self.players_num)) + [self.CARDSTATE_OUT]

        for cardstate in all_cardstates:
            result[cardstate] = []

        for card, cardstate in self.cards_state.items():
            result[cardstate].append(card)

        for cardstate in all_cardstates:
            result[cardstate].sort()

        return result

    def get_state(self) -> dict[Card, int]:
        return self.cards_state

    def get_unseen_cards_sorted(self):
        return [card for card in self.all_cards_sorted
                if card in self.unseen_cards]


class GameTracker:
    def __init__(self) -> None:
        # self.cards_hist: list[list[Card]] = []
        # self.takes_ids_hist: list[list[int]] = []

        players_num = 3

        self.cards_tracker = CardsTracker()
        self.players_tracker = PlayersTracker(players_num=players_num)
        self.game_state = GameState(players_num=players_num)

    def _update_if_table_disappeared(self):
        if self.cards_tracker.is_table_disappeared():
            self.game_state.add(self.cards_tracker.get_table(),
                                self.players_tracker.get_who_take_ids())
            # todo: this is a maybe potentially dangerous place
            #       because we track cards and players independently
            #       idk, do something
            #       at least, this place seems unbeautiful to me :/

            self.cards_tracker.cards_table_static.clear()

    def update(
            self,
            boxes: list[list[int, int, int, int]],
            ranks: list[str],
            suits: list[str],
            players_boxes: list[list[int]],
            players_take: list[str],
            *,
            hardcode_geom: dict):

        y_table_bottom = hardcode_geom['y_table_bottom']

        new_corners = brs_to_corners(boxes, ranks, suits)
        new_corners = [
            c for c in new_corners
            if c.box.box[1] < y_table_bottom and c.box.box[3] < y_table_bottom
        ]

        self.cards_tracker.update(new_corners, hardcode_geom=hardcode_geom)

        new_players = bt_to_players(players_boxes, players_take)
        self.players_tracker.update(new_players)

        self._update_if_table_disappeared()

    def get_state(self):
        result = {
            'cards_table_static': self.cards_tracker.get_table(),
            'takes_players_ids': self.players_tracker.get_who_take_ids()
        }
        return result

    @staticmethod
    def cards_word_wrap(cards: list, num: int) -> list[str]:
        wrapped_cards = [' '.join(map(str, cards[i:i+num]))
                         for i in range(0, len(cards), num)]
        return wrapped_cards

    def get_summary(self):
        lines = []

        # lines.append('cur corners')
        # for c in self.cards_tracker.current_corners:
        #     line = f'{c}'
        #     lines.append(line)

        lines.append('')
        lines.append('cards_hist')
        for ch_elem, th_elem in zip(self.game_state.cards_hist,
                                    self.game_state.takes_ids_hist):
            line = ' '.join([str(card) for card in ch_elem]) + \
                ' | ' + str(th_elem)

            lines.append(line)

        lines.append('')
        lines.append('table')
        lines.append(' '.join(map(str, self.cards_tracker.get_table())))

        lines.append('')
        lines.append('cards by state')

        cards_by_cardstate = self.game_state.get_cards_by_cardstate()
        for cardstate, cards in cards_by_cardstate.items():
            cards_wwrap = self.cards_word_wrap(cards, 12)

            for i, row in enumerate(cards_wwrap):
                prefix = f'{cardstate: 2d}: ' if i == 0 else '    '
                line = prefix + row
                lines.append(line)

        lines.append('')
        lines.append('unseen cards')
        unseen_cards = self.game_state.get_unseen_cards_sorted()
        unseen_cards = self.cards_word_wrap(unseen_cards, 12)
        lines.extend(unseen_cards)

        return '\n'.join(lines)

    def start_new_game(self):
        self.game_state.reset()


def main():
    print(Card.rank_keys)
    assert Card('A', 'd') > Card('K', 'h')


if __name__ == '__main__':
    main()
