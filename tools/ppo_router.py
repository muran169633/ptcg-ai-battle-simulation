"""Stateless, identity-free features for an observable-state PPO router."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch


PUBLIC_ROUTER_FEATURE_VERSION = "ptcg-public-router-features-v1"
DENSE_DIM = 32
HASH_DIM = 128
PUBLIC_ROUTER_FEATURE_DIM = DENSE_DIM + HASH_DIM
# Short aliases retained for local callers that prefer the module context.
ROUTER_FEATURE_VERSION = PUBLIC_ROUTER_FEATURE_VERSION
ROUTER_FEATURE_DIM = PUBLIC_ROUTER_FEATURE_DIM

PUBLIC_HISTORY_V2_FEATURE_VERSION = "ptcg-public-opponent-history-v2"
PUBLIC_HISTORY_V2_DENSE_FEATURE_NAMES = (
    "opponent_event_count",
    "opponent_event_token_count",
    "distinct_opponent_event_count",
    "distinct_opponent_event_type_count",
    "observed_turns_with_opponent_events",
    "opponent_card_reference_event_count",
    "opponent_attack_event_count",
    "opponent_state_change_event_count",
)
PUBLIC_HISTORY_V2_DENSE_FEATURE_BOUNDS = (
    256.0,
    4096.0,
    256.0,
    64.0,
    60.0,
    256.0,
    128.0,
    256.0,
)
PUBLIC_HISTORY_V2_DENSE_DIM = len(PUBLIC_HISTORY_V2_DENSE_FEATURE_NAMES)
PUBLIC_HISTORY_V2_HASH_DIM = 128
PUBLIC_HISTORY_V2_FEATURE_DIM = (
    PUBLIC_HISTORY_V2_DENSE_DIM + PUBLIC_HISTORY_V2_HASH_DIM
)

# V2 is intentionally narrower than LOG_HASH_FIELDS. In particular, terminal
# ``result`` is not accepted because the engine encodes it as an absolute seat.
# Identity metadata, serials, model/deck identifiers, and unknown fields are
# excluded by using this positive whitelist.
PUBLIC_HISTORY_V2_EVENT_FIELDS = (
    "type",
    "hasBasicPokemon",
    "cardId",
    "fromArea",
    "toArea",
    "cardIdActive",
    "cardIdBench",
    "cardIdBefore",
    "cardIdAfter",
    "cardIdTarget",
    "attackId",
    "value",
    "putDamageCounter",
    "isRecover",
    "head",
    "reason",
)

# Order is part of the feature contract.  Every dense value is clipped to
# [0, bound] and divided by its bound before being emitted.
DENSE_FEATURE_NAMES = (
    "turn",
    "turn_action_count",
    "self_is_first",
    "supporter_played",
    "stadium_played",
    "energy_attached",
    "retreated",
    "stadium_present",
    "self_deck_count",
    "self_hand_count",
    "self_active_count",
    "self_bench_count",
    "self_discard_count",
    "self_prize_count",
    "opponent_deck_count",
    "opponent_hand_count",
    "opponent_active_count",
    "opponent_bench_count",
    "opponent_discard_count",
    "opponent_prize_count",
    "self_active_hp_fraction",
    "self_active_damage_fraction",
    "self_active_energy_count",
    "opponent_active_hp_fraction",
    "opponent_active_damage_fraction",
    "opponent_active_energy_count",
    "select_type",
    "select_context",
    "select_min_count",
    "select_max_count",
    "select_option_count",
    "recent_log_count",
)
DENSE_FEATURE_BOUNDS = (
    60.0,
    30.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    60.0,
    30.0,
    2.0,
    8.0,
    60.0,
    6.0,
    60.0,
    30.0,
    2.0,
    8.0,
    60.0,
    6.0,
    1.0,
    1.0,
    12.0,
    1.0,
    1.0,
    12.0,
    64.0,
    64.0,
    60.0,
    60.0,
    128.0,
    64.0,
)

# These are the only scalar keys allowed into the signed-hash block.
CURRENT_HASH_FIELDS = (
    "turn",
    "turnActionCount",
    "supporterPlayed",
    "stadiumPlayed",
    "energyAttached",
    "retreated",
)
PLAYER_HASH_FIELDS = (
    "benchMax",
    "deckCount",
    "handCount",
    "poisoned",
    "burned",
    "asleep",
    "paralyzed",
    "confused",
)
CARD_HASH_FIELDS = (
    "id",
    "hp",
    "maxHp",
    "appearThisTurn",
)
SELECT_HASH_FIELDS = (
    "type",
    "context",
    "minCount",
    "maxCount",
    "remainDamageCounter",
    "remainEnergyCost",
)
OPTION_HASH_FIELDS = (
    "type",
    "number",
    "area",
    "index",
    "toolIndex",
    "energyIndex",
    "count",
    "inPlayArea",
    "inPlayIndex",
    "attackId",
    "cardId",
    "specialConditionType",
)
LOG_HASH_FIELDS = (
    "type",
    "hasBasicPokemon",
    "cardId",
    "fromArea",
    "toArea",
    "cardIdActive",
    "cardIdBench",
    "cardIdBefore",
    "cardIdAfter",
    "cardIdTarget",
    "attackId",
    "value",
    "putDamageCounter",
    "isRecover",
    "head",
    "result",
    "reason",
)

_HASH_KEY = b"ptcg-public-router-v1"
_HASH_PERSON = b"ptcg-router-v1"
_HISTORY_V2_HASH_KEY = b"ptcg-history-v2"
_HISTORY_V2_HASH_PERSON = b"ptcg-hist-v2"
_MAX_ZONE_CARDS = {
    "hand": 60,
    "active": 2,
    "bench": 8,
    "discard": 60,
    "prize": 6,
}
_MAX_LOGS = 64
_MAX_OPTIONS = 128
_MAX_LOOKING_CARDS = 60
_MAX_STADIUM_CARDS = 2
_MAX_SELECT_DECK_CARDS = 60
_MAX_ATTACHMENTS = 16


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return value
    return ()


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    number = _finite_number(value)
    if number is None:
        return None
    return int(number)


def _bounded(value: Any, upper: float) -> float:
    number = _finite_number(value)
    if number is None:
        return 0.0
    return min(max(number, 0.0), upper) / upper


def _flag(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    number = _finite_number(value)
    return float(number != 0.0) if number is not None else 0.0


def _relative_player(player_index: Any, your_index: int | None) -> str:
    index = _integer(player_index)
    if index is None or your_index not in (0, 1):
        return "unknown"
    if index == your_index:
        return "self"
    if index == 1 - your_index:
        return "opponent"
    return "other"


def _visible_cards(zone: Any, limit: int) -> Sequence[Any]:
    return _sequence(zone)[:limit]


def _active_summary(player: Mapping[str, Any]) -> tuple[float, float, float]:
    active = _visible_cards(player.get("active"), 2)
    card = next((item for item in active if isinstance(item, Mapping)), None)
    if card is None:
        return 0.0, 0.0, 0.0
    hp = _finite_number(card.get("hp"))
    max_hp = _finite_number(card.get("maxHp"))
    hp_value = max(hp or 0.0, 0.0)
    max_hp_value = max(max_hp or 0.0, 0.0)
    if max_hp_value > 0.0:
        hp_fraction = min(hp_value / max_hp_value, 1.0)
        damage_fraction = min(
            max(max_hp_value - hp_value, 0.0) / max_hp_value,
            1.0,
        )
    else:
        hp_fraction = 0.0
        damage_fraction = 0.0
    energy_count = len(_visible_cards(card.get("energies"), _MAX_ATTACHMENTS))
    return hp_fraction, damage_fraction, _bounded(energy_count, 12.0)


def _player_views(
    current: Mapping[str, Any],
    your_index: int | None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    players = _sequence(current.get("players"))
    if your_index not in (0, 1) or len(players) < 2:
        return {}, {}
    return _mapping(players[your_index]), _mapping(players[1 - your_index])


def _dense_features(
    current: Mapping[str, Any],
    select: Mapping[str, Any],
    logs: Sequence[Any],
    your_index: int | None,
) -> list[float]:
    self_player, opponent_player = _player_views(current, your_index)
    self_active = _active_summary(self_player)
    opponent_active = _active_summary(opponent_player)
    first_player = _integer(current.get("firstPlayer"))
    self_is_first = float(
        your_index in (0, 1) and first_player == your_index
    )
    stadium_present = float(
        any(
            card is not None
            for card in _visible_cards(
                current.get("stadium"),
                _MAX_STADIUM_CARDS,
            )
        )
    )
    values = [
        _bounded(current.get("turn"), 60.0),
        _bounded(current.get("turnActionCount"), 30.0),
        self_is_first,
        _flag(current.get("supporterPlayed")),
        _flag(current.get("stadiumPlayed")),
        _flag(current.get("energyAttached")),
        _flag(current.get("retreated")),
        stadium_present,
        _bounded(self_player.get("deckCount"), 60.0),
        _bounded(self_player.get("handCount"), 30.0),
        _bounded(len(_visible_cards(self_player.get("active"), 2)), 2.0),
        _bounded(len(_visible_cards(self_player.get("bench"), 8)), 8.0),
        _bounded(len(_visible_cards(self_player.get("discard"), 60)), 60.0),
        _bounded(len(_visible_cards(self_player.get("prize"), 6)), 6.0),
        _bounded(opponent_player.get("deckCount"), 60.0),
        _bounded(opponent_player.get("handCount"), 30.0),
        _bounded(
            len(_visible_cards(opponent_player.get("active"), 2)),
            2.0,
        ),
        _bounded(
            len(_visible_cards(opponent_player.get("bench"), 8)),
            8.0,
        ),
        _bounded(
            len(_visible_cards(opponent_player.get("discard"), 60)),
            60.0,
        ),
        _bounded(
            len(_visible_cards(opponent_player.get("prize"), 6)),
            6.0,
        ),
        *self_active,
        *opponent_active,
        _bounded(select.get("type"), 64.0),
        _bounded(select.get("context"), 64.0),
        _bounded(select.get("minCount"), 60.0),
        _bounded(select.get("maxCount"), 60.0),
        _bounded(len(_visible_cards(select.get("option"), 128)), 128.0),
        _bounded(len(logs[-_MAX_LOGS:]), 64.0),
    ]
    if len(values) != DENSE_DIM:
        raise RuntimeError("Dense router feature contract is inconsistent")
    return values


def _canonical_scalar(value: Any) -> str | None:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return str(int(value)) if value.is_integer() else format(value, ".12g")
    return None


def _append_scalar_tokens(
    tokens: list[str],
    prefix: str,
    source: Mapping[str, Any],
    fields: Sequence[str],
) -> None:
    for field in fields:
        rendered = _canonical_scalar(source.get(field))
        if rendered is not None:
            tokens.append(f"{prefix}:{field}={rendered}")


def _append_card_tokens(
    tokens: list[str],
    prefix: str,
    card: Any,
) -> None:
    if card is None:
        tokens.append(f"{prefix}:hidden")
        return
    value = _mapping(card)
    if not value:
        return
    _append_scalar_tokens(tokens, prefix, value, CARD_HASH_FIELDS)
    for field, limit in (
        ("energies", _MAX_ATTACHMENTS),
        ("energyCards", _MAX_ATTACHMENTS),
        ("tools", 8),
        ("preEvolution", 4),
    ):
        for item in _visible_cards(value.get(field), limit):
            if field == "energies":
                rendered = _canonical_scalar(item)
            else:
                rendered = _canonical_scalar(_mapping(item).get("id"))
            if rendered is not None:
                tokens.append(f"{prefix}:{field}={rendered}")


def _append_zone_tokens(
    tokens: list[str],
    relation: str,
    zone: str,
    cards: Any,
) -> None:
    values = _visible_cards(cards, _MAX_ZONE_CARDS[zone])
    tokens.append(f"player:{relation}:{zone}:count={len(values)}")
    for position, card in enumerate(values):
        _append_card_tokens(
            tokens,
            f"player:{relation}:{zone}:{position}",
            card,
        )


def _append_player_tokens(
    tokens: list[str],
    relation: str,
    player: Mapping[str, Any],
    *,
    include_hand: bool,
) -> None:
    _append_scalar_tokens(
        tokens,
        f"player:{relation}",
        player,
        PLAYER_HASH_FIELDS,
    )
    zones = ("hand", "active", "bench", "discard", "prize")
    for zone in zones:
        if zone == "hand" and not include_hand:
            continue
        _append_zone_tokens(tokens, relation, zone, player.get(zone))


def _hash_tokens(
    current: Mapping[str, Any],
    select: Mapping[str, Any],
    logs: Sequence[Any],
    your_index: int | None,
) -> list[str]:
    tokens: list[str] = []
    _append_scalar_tokens(tokens, "current", current, CURRENT_HASH_FIELDS)
    tokens.append(
        "current:first="
        + _relative_player(current.get("firstPlayer"), your_index)
    )

    self_player, opponent_player = _player_views(current, your_index)
    _append_player_tokens(tokens, "self", self_player, include_hand=True)
    # Opponent hand content is hidden by contract.  Ignore it even if a caller
    # accidentally supplies a populated list.
    _append_player_tokens(
        tokens,
        "opponent",
        opponent_player,
        include_hand=False,
    )

    for position, card in enumerate(
        _visible_cards(current.get("stadium"), _MAX_STADIUM_CARDS)
    ):
        _append_card_tokens(tokens, f"current:stadium:{position}", card)
    for position, card in enumerate(
        _visible_cards(current.get("looking"), _MAX_LOOKING_CARDS)
    ):
        _append_card_tokens(tokens, f"current:looking:{position}", card)

    _append_scalar_tokens(tokens, "select", select, SELECT_HASH_FIELDS)
    for role in ("contextCard", "effect"):
        card = select.get(role)
        if isinstance(card, Mapping):
            _append_card_tokens(tokens, f"select:{role}", card)
    for position, card in enumerate(
        _visible_cards(select.get("deck"), _MAX_SELECT_DECK_CARDS)
    ):
        _append_card_tokens(tokens, f"select:deck:{position}", card)
    for position, option_value in enumerate(
        _visible_cards(select.get("option"), _MAX_OPTIONS)
    ):
        option = _mapping(option_value)
        prefix = f"select:option:{position}"
        _append_scalar_tokens(tokens, prefix, option, OPTION_HASH_FIELDS)
        if "playerIndex" in option:
            tokens.append(
                f"{prefix}:player="
                + _relative_player(option.get("playerIndex"), your_index)
            )

    recent_logs = logs[-_MAX_LOGS:]
    for recent_position, event_value in enumerate(reversed(recent_logs)):
        event = _mapping(event_value)
        if not event:
            continue
        prefix = f"log:{recent_position}"
        _append_scalar_tokens(tokens, prefix, event, LOG_HASH_FIELDS)
        if "playerIndex" in event:
            tokens.append(
                f"{prefix}:player="
                + _relative_player(event.get("playerIndex"), your_index)
            )
    return tokens


def _signed_hash_features(tokens: Sequence[str]) -> list[float]:
    values = [0.0] * HASH_DIM
    if not tokens:
        return values
    for token in tokens:
        digest = hashlib.blake2b(
            token.encode("utf-8"),
            digest_size=16,
            key=_HASH_KEY,
            person=_HASH_PERSON,
        ).digest()
        bucket = int.from_bytes(digest[:8], "little") % HASH_DIM
        sign = 1.0 if digest[8] & 1 else -1.0
        values[bucket] += sign
    scale = math.sqrt(float(len(tokens)))
    return [value / scale for value in values]


def public_router_features(observation: dict[str, Any]) -> torch.Tensor:
    """Return one finite ``torch.float32`` vector with shape ``(160,)``.

    Only explicitly whitelisted fields below ``current``, ``select`` and
    ``logs`` are inspected.  Absolute player indices are used solely to map
    values to ``self`` or ``opponent``; identity metadata is never encoded.
    The function has no history, cache, RNG use, or input mutation.
    """

    root = _mapping(observation)
    current = _mapping(root.get("current"))
    select = _mapping(root.get("select"))
    logs = _sequence(root.get("logs"))
    your_index = _integer(current.get("yourIndex"))
    if your_index not in (0, 1):
        your_index = None

    dense = _dense_features(current, select, logs, your_index)
    tokens = _hash_tokens(current, select, logs, your_index)
    result = torch.tensor(
        dense + _signed_hash_features(tokens),
        dtype=torch.float32,
    )
    if result.shape != (PUBLIC_ROUTER_FEATURE_DIM,):
        raise RuntimeError("Router feature shape contract is inconsistent")
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _history_v2_event_payload(
    event_value: Any,
    opponent_seat: int,
) -> tuple[str, ...] | None:
    """Return a canonical payload only for an explicit opponent event."""

    event = _mapping(event_value)
    if _integer(event.get("playerIndex")) != opponent_seat:
        return None
    tokens: list[str] = []
    for field in PUBLIC_HISTORY_V2_EVENT_FIELDS:
        rendered = _canonical_scalar(event.get(field))
        if rendered is not None:
            tokens.append(f"{field}={rendered}")
    # An explicitly attributed public event is still an event when this engine
    # version supplies none of the whitelisted scalar details.
    return tuple(tokens) if tokens else ("event",)


class PublicOpponentHistoryV2:
    """Causal accumulator for one battle from one fixed observer seat.

    Lifecycle is explicit:

    * construct one instance at ``BattleStart``;
    * call :meth:`observe` on successive observations for that battle;
    * discard it at ``BattleFinish``;
    * when an environment/process slot is reused, either construct a new
      instance or call :meth:`reset` before the next battle.

    ``turn`` regression triggers a defensive automatic reset. It protects
    against an accidentally reused instance when a new game starts at a lower
    turn, but it does not replace the caller's explicit cross-game reset
    contract because two games can begin at the same turn.

    Only log entries whose absolute ``playerIndex`` equals the opponent seat
    are accumulated. The observer's own events, unknown/unattributed events,
    identity metadata, and non-whitelisted fields never affect the state.
    """

    def __init__(self, observer_seat: int) -> None:
        self._generation = -1
        self._turn_regression_resets = 0
        self._observer_seat = 0
        self.reset(observer_seat=observer_seat, reason="battle_start")

    @staticmethod
    def _validate_seat(observer_seat: int) -> int:
        if isinstance(observer_seat, bool) or observer_seat not in (0, 1):
            raise ValueError("observer_seat must be absolute seat 0 or 1")
        return int(observer_seat)

    def reset(
        self,
        *,
        observer_seat: int | None = None,
        reason: str = "explicit_new_game",
    ) -> None:
        """Clear all battle-local state before a new process-slot game."""

        if observer_seat is not None:
            self._observer_seat = self._validate_seat(observer_seat)
        self._generation += 1
        self._last_reset_reason = str(reason)
        self._last_turn: int | None = None
        self._last_observation: dict[str, Any] | None = None
        self._event_count = 0
        self._event_token_count = 0
        self._distinct_events: set[tuple[str, ...]] = set()
        self._distinct_event_types: set[str] = set()
        self._turns_with_events: set[int] = set()
        self._card_reference_event_count = 0
        self._attack_event_count = 0
        self._state_change_event_count = 0
        self._hash_values = [0.0] * PUBLIC_HISTORY_V2_HASH_DIM

    @property
    def observer_seat(self) -> int:
        return self._observer_seat

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def last_reset_reason(self) -> str:
        return self._last_reset_reason

    @property
    def turn_regression_resets(self) -> int:
        return self._turn_regression_resets

    @property
    def event_count(self) -> int:
        return self._event_count

    def _add_hash_token(self, token: str) -> None:
        digest = hashlib.blake2b(
            token.encode("utf-8"),
            digest_size=16,
            key=_HISTORY_V2_HASH_KEY,
            person=_HISTORY_V2_HASH_PERSON,
        ).digest()
        bucket = (
            int.from_bytes(digest[:8], "little")
            % PUBLIC_HISTORY_V2_HASH_DIM
        )
        sign = 1.0 if digest[8] & 1 else -1.0
        self._hash_values[bucket] += sign

    def _append_event(
        self,
        payload: tuple[str, ...],
        observed_turn: int | None,
    ) -> None:
        self._event_count += 1
        self._event_token_count += len(payload)
        self._distinct_events.add(payload)
        if observed_turn is not None:
            self._turns_with_events.add(observed_turn)
        token_fields = {token.split("=", 1)[0] for token in payload}
        for token in payload:
            self._add_hash_token(token)
            if token.startswith("type="):
                self._distinct_event_types.add(token)
        if token_fields & {
            "cardId",
            "cardIdActive",
            "cardIdBench",
            "cardIdBefore",
            "cardIdAfter",
            "cardIdTarget",
        }:
            self._card_reference_event_count += 1
        if "attackId" in token_fields:
            self._attack_event_count += 1
        if token_fields & {
            "fromArea",
            "toArea",
            "cardIdBefore",
            "cardIdAfter",
            "cardIdTarget",
            "value",
            "putDamageCounter",
            "isRecover",
        }:
            self._state_change_event_count += 1

    def observe(self, observation: dict[str, Any]) -> int:
        """Causally consume newly visible opponent events.

        The official API defines ``logs`` as events since the previous
        selection. A submitted agent is invoked only for its own selections,
        so observations whose ``current.yourIndex`` is not the fixed observer
        seat must not enter the history even when an offline two-policy runner
        can inspect them. Call this exactly once for each observation actually
        delivered to the observer agent. Repeating the same dictionary object
        is idempotent as a defensive convenience. Returns the number of
        opponent events appended by this call.
        """

        if observation is self._last_observation:
            return 0
        root = _mapping(observation)
        current = _mapping(root.get("current"))
        if _integer(current.get("yourIndex")) != self._observer_seat:
            return 0
        turn = _integer(current.get("turn"))
        if (
            turn is not None
            and self._last_turn is not None
            and turn < self._last_turn
        ):
            self._turn_regression_resets += 1
            self.reset(reason="turn_regression")

        opponent_seat = 1 - self._observer_seat
        current_events = tuple(
            payload
            for event in _sequence(root.get("logs"))
            if (
                payload := _history_v2_event_payload(
                    event,
                    opponent_seat,
                )
            )
            is not None
        )
        for payload in current_events:
            self._append_event(payload, turn)
        self._last_observation = observation
        if turn is not None:
            self._last_turn = turn
        return len(current_events)

    def features(self) -> torch.Tensor:
        """Return the current finite opponent-history vector."""

        dense_raw = (
            self._event_count,
            self._event_token_count,
            len(self._distinct_events),
            len(self._distinct_event_types),
            len(self._turns_with_events),
            self._card_reference_event_count,
            self._attack_event_count,
            self._state_change_event_count,
        )
        dense = [
            _bounded(value, bound)
            for value, bound in zip(
                dense_raw,
                PUBLIC_HISTORY_V2_DENSE_FEATURE_BOUNDS,
                strict=True,
            )
        ]
        scale = math.sqrt(float(max(self._event_token_count, 1)))
        hashed = [value / scale for value in self._hash_values]
        result = torch.tensor(dense + hashed, dtype=torch.float32)
        if result.shape != (PUBLIC_HISTORY_V2_FEATURE_DIM,):
            raise RuntimeError(
                "Public opponent history v2 feature contract is inconsistent"
            )
        return torch.nan_to_num(
            result,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def observe_and_features(
        self,
        observation: dict[str, Any],
    ) -> torch.Tensor:
        """Consume one causal observation and return the updated feature."""

        self.observe(observation)
        return self.features()
