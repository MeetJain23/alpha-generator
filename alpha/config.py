"""Credentials and deployment settings, from the environment and nowhere else.

No secret is ever read from a file this repository tracks, and no secret has a
default. Those two rules do most of the work: a default is how a placeholder
reaches production, and a tracked file is how a key reaches a public
repository.

Failing loudly
--------------
A missing key raises at startup, naming the variable. The alternative is
passing ``None`` into a request and meeting it again as a 401 an hour later,
in a log line that says the vendor rejected the request rather than that
nobody set ``NDL_API_KEY``. The second is a worse bug than the first because
it points at the wrong system.

Keys never reach a log line
---------------------------
This is the part that goes wrong quietly. Structured logging encourages
attaching a whole config object to a record, and a dataclass renders its
fields on ``repr``, so a key ends up in a log file that is shipped somewhere,
rotated somewhere, and readable by more people than the key ever should be.

``Secret`` therefore wraps every credential and refuses to render itself.
``repr``, ``str`` and f-string interpolation all give a redaction. The value
comes out only through an explicit ``.reveal()`` call, which is greppable, and
which is the point: an audit for where a key is used is a search for one
method name rather than a reading of every format string.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Final

REDACTION: Final[str] = "<redacted>"


class MissingCredential(RuntimeError):
    """A required environment variable is not set."""


@dataclass(frozen=True, slots=True)
class Secret:
    """A credential that will not render itself.

    Holds the value, and hands it over only when asked in so many words.
    Every accidental path to a string, which is to say every path a logger or
    a traceback takes, gives the redaction instead.
    """

    name: str
    _value: str = field(repr=False)

    def reveal(self) -> str:
        """The actual value. Call this at the point of use, never earlier."""
        return self._value

    def __repr__(self) -> str:
        return f"Secret({self.name}={REDACTION})"

    def __str__(self) -> str:
        return REDACTION

    def __format__(self, spec: str) -> str:
        # Without this an f-string would fall through to __str__ for an empty
        # spec but to the underlying value for some others.
        return REDACTION

    def __len__(self) -> int:
        """Length without disclosure, so a caller can check for emptiness."""
        return len(self._value)

    def __bool__(self) -> bool:
        return bool(self._value)


def require(name: str) -> Secret:
    """Read a required credential, or raise naming the variable.

    No default, ever. A default for a credential is a placeholder that works
    in development, reaches production, and fails there.
    """
    value = os.environ.get(name)
    if not value:
        raise MissingCredential(
            f"environment variable {name} is not set. It holds a credential, "
            f"so it has no default and cannot be read from a file in this "
            f"repository. Set it in the environment, and see .env.example for "
            f"the variables this system expects."
        )
    return Secret(name, value)


def optional_setting(name: str, default: str) -> str:
    """A non-secret setting that may sensibly have a default.

    Deliberately separate from ``require``. Keeping the two functions distinct
    means a credential cannot acquire a default by someone reaching for the
    convenient call, and a reader can tell which is which without checking the
    argument list.
    """
    return os.environ.get(name) or default


@dataclass(frozen=True, slots=True)
class DataCredentials:
    """Vendor credentials. Every field is a Secret, so none of them render.

    Built by ``load`` rather than at import, because importing this module
    must not fail on a machine that only runs the tests. Nothing in the test
    suite touches a vendor.
    """

    nasdaq_data_link: Secret

    def __repr__(self) -> str:
        return f"DataCredentials(nasdaq_data_link={REDACTION})"


def load_data_credentials() -> DataCredentials:
    """Read every vendor credential, failing on the first one missing."""
    return DataCredentials(nasdaq_data_link=require("NDL_API_KEY"))
