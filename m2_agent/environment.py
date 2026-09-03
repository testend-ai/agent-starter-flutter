"""Stateful environment / mock backend (tau2 Layer-1 pattern).

Implements the tau2 environment pieces: a ``DB`` base with a deterministic
SHA-256 ``get_hash()``, ``@is_tool`` + ``ToolKitBase`` metaclass tool
registration, and an ``Environment`` that is the only place state mutates.
For the current intent-only golden CSV the world is the telephony/policy
domain; per-intent ``tool_spec`` extensions later plug into the same
interface without touching tasks.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any, Callable, Literal

from pydantic import BaseModel

from schemas import ToolCall

Requestor = Literal["agent", "user"]


def canonical_json(payload: Any) -> str:
    """Canonical JSON: fixed key order + separators so hashes are reproducible."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class DB(BaseModel):
    """Base state container; hash snapshots the end-of-call world state."""

    def get_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.model_dump()).encode("utf-8")).hexdigest()


class TelephonyDB(DB):
    """Minimal telephony/policy world for pure-intent tasks.

    ``detected_intents``/``actions`` are written only through tools, so the end
    state doubles as an audit trail of what the conversation actually did.
    """

    detected_intents: list[str] = []
    actions: list[dict[str, Any]] = []
    transferred_to_human: bool = False
    resolution_confirmed_by_caller: bool = False


def _json_schema_for(func: Callable[..., Any]) -> dict[str, Any]:
    """Tiny docstring+typehint to OpenAI tool-schema extractor.

    Avoids the docstring-parser dependency: reads the first docstring line as
    description and simple annotations for parameter types. Skips ``self`` so
    registered (unbound) methods produce clean parameter schemas.
    """
    sig = inspect.signature(func)
    props: dict[str, Any] = {}
    required: list[str] = []
    type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
    for pname, param in sig.parameters.items():
        if pname == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else str
        json_type = type_map.get(annotation, "string")
        prop: dict[str, Any] = {"type": json_type}
        props[pname] = prop
        if param.default is inspect.Parameter.empty:
            required.append(pname)
        else:
            prop["description"] = f"default: {param.default!r}"
    doc = inspect.getdoc(func) or func.__name__.replace("_", " ")
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc.splitlines()[0],
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


class ToolKitType(type):
    """Metaclass that collects ``@is_tool`` methods into ``cls._func_tools``."""

    def __new__(mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any]) -> "ToolKitType":
        cls = super().__new__(mcs, name, bases, namespace)
        registry: dict[str, tuple[Callable[..., Any], bool]] = {}
        for attr in vars(cls).values():
            if callable(attr) and getattr(attr, "_is_tool", False):
                registry[attr.__name__] = (attr, getattr(attr, "_mutates_state", False))
        # inherit parent tools so subclassing a toolkit extends it
        for base in bases:
            inherited = getattr(base, "_func_tools", {})
            for tool_name, entry in inherited.items():
                registry.setdefault(tool_name, entry)
        cls._func_tools = registry  # type: ignore[attr-defined]
        return cls


def is_tool(mutates_state: bool = False) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a method as an environment tool (tau2 ``@is_tool(ToolType)``)."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func._is_tool = True  # type: ignore[attr-defined]
        func._mutates_state = mutates_state  # type: ignore[attr-defined]
        return func

    return decorator


class ToolKitBase(metaclass=ToolKitType):
    """Base class for assistant/user toolkits bound to one Environment DB."""

    _func_tools: dict[str, tuple[Callable[..., Any], bool]]

    def __init__(self, db: DB) -> None:
        self._db = db

    @classmethod
    def tool_names(cls) -> list[str]:
        return sorted(cls._func_tools.keys())

    @classmethod
    def tool_schemas(cls) -> list[dict[str, Any]]:
        return [_json_schema_for(fn) for fn, _ in cls._func_tools.values()]

    def call(self, name: str, **kwargs: Any) -> Any:
        if name not in self._func_tools:
            raise KeyError(f"unknown tool '{name}'")
        fn, _ = self._func_tools[name]
        return fn(self, **kwargs)

    def mutates(self, name: str) -> bool:
        return self._func_tools[name][1]


class AssistantToolkit(ToolKitBase):
    """Agent-side tools — the ONLY way M2 touches state (How we benchmark.md 2)."""

    @is_tool(mutates_state=True)
    def detect_intent(self, intent_code: str) -> str:
        """Record the intent code the agent concluded the caller wants."""
        assert isinstance(self._db, TelephonyDB)
        self._db.detected_intents.append(intent_code.upper())
        self._db.actions.append({"tool": "detect_intent", "intent": intent_code.upper()})
        return f"intent {intent_code.upper()} recorded"

    @is_tool(mutates_state=True)
    def transfer_to_human(self, reason: str) -> str:
        """Escalate the call to a human agent."""
        assert isinstance(self._db, TelephonyDB)
        self._db.transferred_to_human = True
        self._db.actions.append({"tool": "transfer_to_human", "reason": reason})
        return "transfer queued"

    @is_tool()
    def lookup_policy(self, topic: str) -> str:
        """Look up bank policy text for a topic (proof-of-funds, cards, fees...)."""
        policies = {
            "proof_of_funds": "An official account confirmation letter can be issued within 3 business days after identity verification.",
            "virtual_card": "A temporary virtual card number can be generated in the banking app for online purchases.",
            "marketing_opt_out": "Marketing consent can be withdrawn at any time; processing takes up to 48 hours.",
        }
        normalized = topic.lower().strip().replace("-", "_").replace(" ", "_")
        return policies.get(normalized, f"No policy found for topic '{topic}'.")


class UserToolkit(ToolKitBase):
    """Caller-side tools (tau2 dual-control): M1 acts on the world, not the agent."""

    @is_tool(mutates_state=True)
    def confirm_resolution(self, accepted: bool) -> str:
        """Caller confirms whether the agent's answer resolved their need."""
        assert isinstance(self._db, TelephonyDB)
        if accepted:
            self._db.resolution_confirmed_by_caller = True
        self._db.actions.append({"tool": "confirm_resolution", "accepted": accepted})
        return "caller accepted" if accepted else "caller not satisfied"

    @is_tool(mutates_state=True)
    def hang_up(self) -> str:
        """Caller ends the call."""
        self._db.actions.append({"tool": "hang_up"})
        return "call ended"


DEFAULT_POLICY = (
    "You are M2, the phone agent of a German bank/insurer. Be concise, professional and "
    "helpful. Use the provided tools when you conclude what the caller needs. Confirm "
    "outcomes clearly before ending the call."
)


class Environment:
    """Holds policy + both toolkits + DB; executes tool calls atomically."""

    def __init__(
        self,
        db: DB | None = None,
        policy_text: str = DEFAULT_POLICY,
        assistant_tools: ToolKitBase | None = None,
        user_tools: ToolKitBase | None = None,
    ) -> None:
        self.db = db or TelephonyDB()
        self.policy_text = policy_text
        self.assistant_tools = assistant_tools or AssistantToolkit(self.db)
        self.user_tools = user_tools or UserToolkit(self.db)

    def toolkit_for(self, requestor: Requestor) -> ToolKitBase:
        return self.assistant_tools if requestor == "agent" else self.user_tools

    def make_tool_call(self, call: ToolCall, requestor: Requestor) -> tuple[Any, bool]:
        """Execute one tool call; returns (result, error). Raises on unknown tools."""
        toolkit = self.toolkit_for(requestor)
        try:
            result = toolkit.call(call.name, **call.arguments)
            return result, False
        except Exception as exc:  # noqa: BLE001 — surface any tool failure to the caller
            return f"Error: {exc}", True

    def get_db_hash(self) -> str:
        return self.db.get_hash()

    def snapshot(self) -> dict[str, Any]:
        return self.db.model_dump()

    def set_state(self, snapshot: dict[str, Any]) -> None:
        self.db = type(self.db)(**snapshot)
