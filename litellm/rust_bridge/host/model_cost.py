"""The Python side of the model cost table that Rust asks for: the row type, how a kept registration merges and what
Python derives from the table (all belong to the Python host), and the registration lookup and row merge (shims until
lookup moves to Rust).

Contract for every function here:
- Rust calls it synchronously with the GIL held and no ``NativeModelCostMap`` lock held, since ``resolve`` reaches
  ``find_key`` through ``get_model_info``.
- Values cross as Python objects. Rows and registrations can hold values JSON cannot (``datetime`` in router
  ``model_info``), so nothing here is converted to JSON.
- Exceptions propagate to the ``register_model`` caller unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from litellm.litellm_core_utils.get_model_cost_map import AliasedCostEntry


@dataclass(frozen=True, slots=True)
class Landing:
    key: str
    base: Mapping[str, object]
    matched_builtin: bool


def entry(fields: dict[str, object], names: tuple[str, ...], registered_fields: frozenset[str]) -> AliasedCostEntry:
    """The row object every alias name of one model shares in ``litellm.model_cost``."""
    return AliasedCostEntry(fields, names, registered_fields)


def resolve(
    rows: Mapping[str, Mapping[str, object]],
    key: str,
    registration: Mapping[str, object],
    custom_llm_provider: str | None,
) -> Landing | None:
    """Where a ``register_model`` entry lands in ``rows`` and the row it starts from. ``None`` when ``rows`` is not
    the published table, because Python lookups read only ``litellm.model_cost``, so the caller retries on the new
    table.

    Shim: this is lookup logic, not host state. It stays here until ``get_model_info`` moves to Rust, when its body
    becomes a native call. ``model_cost_registration_row`` asks the same question through ``registration_base``,
    so there is one resolver for both callers."""
    import litellm
    from litellm.utils import registration_base

    if rows is not litellm.model_cost:
        return None
    base = registration_base(key, registration, custom_llm_provider)
    return Landing(base.key, base.entry, base.matched_builtin)


def merged(
    current: Mapping[str, object] | None, base: Mapping[str, object], registration: Mapping[str, object]
) -> AliasedCostEntry:
    """The row a registration writes over ``current``, the row now under its landing key.

    Shim: the merge rule belongs in core, but rows hold Python objects JSON cannot, so it stays here until core has a
    merge generic over the value type."""
    from litellm.utils import model_cost_registration_write

    return model_cost_registration_write(current, base, registration)


def kept(previous: dict[str, object] | None, registration: Mapping[str, object]) -> dict[str, object]:
    """The registration ``register_model(persist_across_reloads=True)`` keeps for a key: ``registration`` merged
    into the one kept before it, as main merged them."""
    from litellm.utils import update_dictionary

    return update_dictionary(dict(previous or {}), registration)


def changed() -> None:
    """Drops what Python derives from ``litellm.model_cost`` after a write landed in it. Rust calls it after every
    entry, because the next entry's lookup reads through the ``get_model_info`` cache."""
    from litellm.utils import model_cost_changed

    model_cost_changed()
