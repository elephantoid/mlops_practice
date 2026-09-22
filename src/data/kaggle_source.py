"""Acquisition for track source data, with an error taxonomy that names the real fault.

Kaggle exposes two download modes that fail differently, and conflating them produces a
misdiagnosis that costs real time:

* A **dataset** (``mlg-ulb/creditcardfraud``) needs only an API token.
* A **competition** (``home-credit-default-risk``) needs the token **and** browser
  acceptance of that competition's rules. Without the acceptance the API returns
  **403 Forbidden** -- indistinguishable, at the status-code level, from a bad key.

That is the entire reason :class:`KaggleAuthError` and :class:`KaggleConsentError` are
separate types. A 403 on a competition, when the token has already been proven against a
dataset, is a *consent* failure, and the only useful thing to tell the operator is the URL
where consent is given. A single ``KaggleError`` would send them to regenerate a token
that was never the problem.

This module is deliberately network-free to construct and to test. Downloads go through
:func:`_run_kaggle`, which is the single seam the tests mock, and every decision the module
makes -- which mode, whether the cache is warm, which fallback to take, what to record as
``source_used`` -- is observable without a token, an archive, or a network call.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

from src.data.tracks import SourceSpec

logger = logging.getLogger(__name__)

KAGGLE_CONFIG_PATH = Path.home() / ".kaggle" / "kaggle.json"

# Shown verbatim to whoever hits the consent wall. A competition slug maps to its rules
# page by construction, so the URL is derived rather than hardcoded per competition.
RULES_URL_TEMPLATE = "https://www.kaggle.com/c/{slug}/rules"

DOWNLOAD_TIMEOUT_SECONDS = 1800


class KaggleSourceError(RuntimeError):
    """Base class, so a caller may catch both modes when it genuinely does not care."""


class KaggleAuthError(KaggleSourceError):
    """No usable credential.

    Either ``~/.kaggle/kaggle.json`` is absent, or the token in it was rejected. Both are
    fixed the same way: regenerate the token at kaggle.com/settings.
    """


class KaggleConsentError(KaggleSourceError):
    """Credential is fine; the competition's rules have not been accepted.

    Carries the rules URL because that is the only actionable part. This is the failure
    that masquerades as an auth error, and the reason the fraud dataset is fetched first --
    a token already proven against a dataset cannot be the cause of a competition 403.
    """

    def __init__(self, slug: str, detail: str = "") -> None:
        self.slug = slug
        self.rules_url = RULES_URL_TEMPLATE.format(slug=slug)
        message = (
            f"Kaggle returned 403 for competition {slug!r}, and the API token is valid. "
            f"This is a rules-acceptance failure, not an authentication failure: "
            f"open {self.rules_url} and accept the competition rules, then retry. "
            f"There is no API equivalent -- rule acceptance is a consent action."
        )
        if detail:
            message = f"{message} Underlying output: {detail}"
        super().__init__(message)


@dataclass(frozen=True)
class Acquisition:
    """What an acquisition actually did.

    ``source_used`` is the point of this object. When a primary source is unavailable and
    a fallback runs, the resulting parquet must carry which one produced it -- otherwise a
    model trained on the UCI fallback is indistinguishable, downstream, from one trained
    on Home Credit, and the two are different datasets with different columns.
    """

    path: Path
    source_used: str
    from_cache: bool
    is_fallback: bool


def credentials_available(config_path: Path = KAGGLE_CONFIG_PATH) -> bool:
    """Whether a Kaggle token is present.

    Checked before invoking the CLI so a missing credential is reported as the plain
    thing it is, rather than as whatever the CLI happens to print when it cannot
    authenticate.
    """
    return config_path.is_file()


def _run_kaggle(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Invoke the Kaggle CLI. The single seam the tests mock.

    Kept as a thin wrapper with no logic of its own so that mocking it in tests replaces
    exactly the network call and nothing else -- the mode dispatch, cache check, error
    classification and extraction all still execute for real under test.
    """
    return subprocess.run(
        ["kaggle", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=DOWNLOAD_TIMEOUT_SECONDS,
        check=False,
    )


def _classify_failure(result: subprocess.CompletedProcess[str], spec: SourceSpec) -> Exception:
    """Turn a failed CLI invocation into the error that names the actual fault."""
    output = f"{result.stdout}\n{result.stderr}".strip()
    lowered = output.lower()

    forbidden = "403" in output or "forbidden" in lowered
    if forbidden and spec.requires_rule_acceptance:
        return KaggleConsentError(spec.source_ref, detail=output[:400])

    if forbidden or "401" in output or "unauthorized" in lowered:
        return KaggleAuthError(
            f"Kaggle rejected the credential for {spec.source_ref!r}. Regenerate the token "
            f"at https://www.kaggle.com/settings ('Create New Token') and save it to "
            f"{KAGGLE_CONFIG_PATH} with mode 600. Underlying output: {output[:400]}"
        )

    return KaggleSourceError(
        f"Kaggle download failed for {spec.source_ref!r} (exit {result.returncode}): {output[:400]}"
    )


def _cli_args(spec: SourceSpec, destination: Path) -> list[str]:
    """Argument vector for this source's mode.

    The two subcommands are not interchangeable: `datasets download -d` on a competition
    slug fails with a confusing not-found rather than the 403 that would have told the
    operator what was actually wrong.
    """
    if spec.source_kind == "kaggle_competition":
        return ["competitions", "download", "-c", spec.source_ref, "-p", str(destination)]
    if spec.source_kind == "kaggle_dataset":
        return ["datasets", "download", "-d", spec.source_ref, "-p", str(destination)]
    raise ValueError(f"{spec.source_kind!r} is not a Kaggle source kind")


def _extract(archive: Path, destination: Path, members: tuple[str, ...]) -> None:
    """Unpack, taking only the members the track declares.

    Home Credit's archive carries eight relational tables at roughly 166 MB; W1 models the
    application table alone. Extracting everything would cost disk and time for tables
    nothing reads.
    """
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        wanted = list(members) if members else sorted(names)

        missing = [m for m in wanted if m not in names]
        if missing:
            raise KaggleSourceError(
                f"{archive.name} does not contain {missing}; it holds {sorted(names)[:10]}"
            )

        for member in wanted:
            bundle.extract(member, destination)


def is_cached(spec: SourceSpec, destination: Path) -> bool:
    """Whether every declared member is already on disk.

    Membership, not mere directory existence: a half-extracted archive left by an
    interrupted run must not read as a cache hit and skip the repair.
    """
    if spec.archive_members:
        return all((destination / member).is_file() for member in spec.archive_members)
    return (destination / spec.primary_table).is_file()


def acquire(
    spec: SourceSpec,
    destination: Path,
    *,
    allow_fallback: bool = True,
    config_path: Path = KAGGLE_CONFIG_PATH,
) -> Acquisition:
    """Make a track's raw data available, returning what was actually used.

    Never re-downloads: a warm cache short-circuits before any CLI invocation, so repeated
    ingest runs cost nothing and a rate limit cannot break a rerun.

    On failure, falls back when the spec declares one and ``allow_fallback`` permits it,
    logging a WARNING that names both the failure and the substitution. The substitution is
    recorded in the returned :class:`Acquisition` rather than left implicit, because for
    the credit track the fallback is a *different dataset*, not the same data by another
    route.
    """
    destination.mkdir(parents=True, exist_ok=True)

    if is_cached(spec, destination):
        logger.info("Cache hit for %s; skipping download", spec.source_ref)
        return Acquisition(
            path=destination / spec.primary_table,
            source_used=spec.source_ref,
            from_cache=True,
            is_fallback=False,
        )

    try:
        if spec.source_kind in ("kaggle_competition", "kaggle_dataset"):
            if not credentials_available(config_path):
                raise KaggleAuthError(
                    f"No Kaggle credential at {config_path}. Create one at "
                    f"https://www.kaggle.com/settings ('Create New Token'), save it there, "
                    f"and chmod 600 it."
                )

            result = _run_kaggle(_cli_args(spec, destination), cwd=destination)
            if result.returncode != 0:
                raise _classify_failure(result, spec)

            archives = sorted(destination.glob("*.zip"))
            if archives:
                _extract(archives[0], destination, spec.archive_members)
        else:
            raise KaggleSourceError(
                f"{spec.source_kind!r} is not handled by this module; "
                f"auth-free sources are fetched by their own loader"
            )

    except KaggleSourceError as exc:
        if not (allow_fallback and spec.fallback is not None):
            raise

        logger.warning(
            "Primary source %r unavailable (%s: %s). Falling back to %r. %s",
            spec.source_ref,
            type(exc).__name__,
            str(exc)[:200],
            spec.fallback.source_ref,
            (
                "NOTE: this fallback is a DIFFERENT dataset, not the same data by another "
                "route -- downstream schema and feature contracts change with it."
                if not spec.equivalent_to_primary
                else "The fallback carries equivalent data."
            ),
        )
        return Acquisition(
            path=destination / spec.fallback.primary_table,
            source_used=spec.fallback.source_ref,
            from_cache=False,
            is_fallback=True,
        )

    return Acquisition(
        path=destination / spec.primary_table,
        source_used=spec.source_ref,
        from_cache=False,
        is_fallback=False,
    )


# Sources that need no credential. Fetched by id through their own library rather than
# the Kaggle CLI, but they resolve through the SAME entrypoint as the primary -- which is
# what makes a typo'd id fail at resolution instead of surfacing as a confusing download
# error hours later, or worse, as a silent no-op.
AUTH_FREE_KINDS = frozenset({"openml", "url"})


def resolve(spec: SourceSpec) -> str:
    """Validate that a source reference is well-formed, returning its canonical id.

    Every source -- primary and fallback, Kaggle and auth-free -- goes through here, so a
    malformed reference fails the same way regardless of which one it is. Before this
    existed, a fallback was only ever checked for being non-``None``: a ``source_ref`` of
    ``"424777"`` instead of ``"42477"`` passed every test and would have failed only at
    the moment the fallback was actually needed, which is precisely the moment there is no
    time to debug it.

    Raises ``ValueError`` naming the offending reference and what the kind expects.
    """
    ref = spec.source_ref.strip()
    if not ref:
        raise ValueError(f"{spec.source_kind} source has an empty source_ref")

    if spec.source_kind == "openml":
        # OpenML data ids are integers. A slug or a typo'd non-numeric id is caught here
        # rather than by fetch_openml raising something less specific over the network.
        if not ref.isdigit():
            raise ValueError(
                f"OpenML source_ref must be a numeric data id; got {ref!r}. "
                f"(Credit fallback is 42477, fraud fallback is 1597.)"
            )
        return ref

    if spec.source_kind == "kaggle_dataset":
        # owner/dataset-name. A competition slug here is the common mistake and it fails
        # as a confusing not-found rather than as the 403 that would explain itself.
        if ref.count("/") != 1 or any(not part for part in ref.split("/")):
            raise ValueError(
                f"Kaggle dataset source_ref must be 'owner/dataset'; got {ref!r}. "
                f"A bare slug is a competition -- use source_kind='kaggle_competition'."
            )
        return ref

    if spec.source_kind == "kaggle_competition":
        if "/" in ref:
            raise ValueError(
                f"Kaggle competition source_ref must be a bare slug; got {ref!r}. "
                f"An 'owner/name' reference is a dataset -- use source_kind='kaggle_dataset'."
            )
        return ref

    if spec.source_kind == "url":
        if not ref.startswith(("http://", "https://")):
            raise ValueError(f"url source_ref must be an absolute URL; got {ref!r}")
        return ref

    raise ValueError(f"unknown source_kind {spec.source_kind!r} for {ref!r}")


def resolve_chain(spec: SourceSpec) -> list[str]:
    """Resolve a source and every fallback beneath it, primary first.

    Used to validate a whole registry up front: a fallback that cannot resolve is a
    fallback that does not exist, and discovering that at the moment of failure defeats
    the point of having one.
    """
    chain = [resolve(spec)]
    current = spec.fallback
    while current is not None:
        chain.append(resolve(current))
        current = current.fallback
    return chain


def describe_blocker(spec: SourceSpec, config_path: Path = KAGGLE_CONFIG_PATH) -> str | None:
    """Human-readable description of what stands between here and this source.

    Returns ``None`` when nothing does. Exists so an operator can be told what to do
    before a long download is attempted and fails, rather than after.
    """
    if not credentials_available(config_path):
        return (
            f"No Kaggle token at {config_path}. Create one at "
            f"https://www.kaggle.com/settings ('Create New Token'), save it there, chmod 600."
        )
    if spec.requires_rule_acceptance:
        return (
            f"Competition {spec.source_ref!r} additionally requires rules acceptance at "
            f"{RULES_URL_TEMPLATE.format(slug=spec.source_ref)} -- no API equivalent exists. "
            f"A 403 here means consent, not credentials."
        )
    return None


def kaggle_cli_available() -> bool:
    """Whether the ``kaggle`` executable is on PATH."""
    return shutil.which("kaggle") is not None or bool(os.environ.get("KAGGLE_CLI_PATH"))
