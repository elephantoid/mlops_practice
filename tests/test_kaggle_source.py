"""Tests for the Kaggle acquisition layer.

Every test here is mock-driven: no token, no archive, no network. That is deliberate and
it is the reason this module could be written while data acquisition is blocked -- the
decisions worth testing (which mode, is the cache warm, which error is this really, what
gets recorded as source_used) are all pure logic around a single subprocess seam.

The error taxonomy is the point. A 403 from Kaggle is ambiguous at the status-code level:
on a dataset it means the token is bad, on a competition it usually means the rules were
never accepted. Sending someone to regenerate a working token because the real problem was
consent is exactly the failure this module exists to prevent.
"""

from __future__ import annotations

import dataclasses
import subprocess
import zipfile
from pathlib import Path

import pytest

from src.data import kaggle_source
from src.data.kaggle_source import (
    Acquisition,
    KaggleAuthError,
    KaggleConsentError,
    KaggleSourceError,
    acquire,
    describe_blocker,
    is_cached,
)
from src.data.tracks import SourceSpec

COMPETITION = SourceSpec(
    source_kind="kaggle_competition",
    source_ref="home-credit-default-risk",
    primary_table="application_train.csv",
    archive_members=("application_train.csv",),
    fallback=SourceSpec(
        source_kind="openml",
        source_ref="42477",
        primary_table="default-of-credit-card-clients",
    ),
    equivalent_to_primary=False,
)

DATASET = SourceSpec(
    source_kind="kaggle_dataset",
    source_ref="mlg-ulb/creditcardfraud",
    primary_table="creditcard.csv",
    archive_members=("creditcard.csv",),
)


def _result(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["kaggle"], returncode=returncode, stdout=stdout, stderr=stderr
    )


@pytest.fixture
def token(tmp_path):
    """A present-and-valid-looking credential, so auth is never the reason a test fails."""
    path = tmp_path / "kaggle.json"
    path.write_text('{"username": "u", "key": "k"}')
    return path


def test_competition_403_is_a_consent_error_naming_the_rules_url(tmp_path, token, monkeypatch):
    """The headline case. A 403 on a competition, with a valid token, means consent."""
    monkeypatch.setattr(
        kaggle_source,
        "_run_kaggle",
        lambda *a, **k: _result(1, stderr="403 Client Error: Forbidden"),
    )

    with pytest.raises(KaggleConsentError) as excinfo:
        acquire(COMPETITION, tmp_path / "raw", allow_fallback=False, config_path=token)

    error = excinfo.value
    assert error.rules_url == "https://www.kaggle.com/c/home-credit-default-risk/rules"
    message = str(error)
    assert error.rules_url in message, (
        "the actionable URL must be in the message, not just an attribute"
    )
    assert "not an authentication failure" in message, (
        "the message must actively steer away from the wrong fix"
    )


def test_dataset_403_is_an_auth_error_not_a_consent_error(tmp_path, token, monkeypatch):
    """A dataset has no rules to accept, so the same status code means the opposite thing."""
    monkeypatch.setattr(
        kaggle_source,
        "_run_kaggle",
        lambda *a, **k: _result(1, stderr="403 Client Error: Forbidden"),
    )

    with pytest.raises(KaggleAuthError) as excinfo:
        acquire(DATASET, tmp_path / "raw", allow_fallback=False, config_path=token)

    message = str(excinfo.value)
    assert "kaggle.com/settings" in message
    assert "rules" not in message.lower(), "a dataset failure must not send anyone to a rules page"


def test_missing_credential_is_reported_before_the_cli_runs(tmp_path, monkeypatch):
    """Whatever the CLI prints when it cannot authenticate is not what we want to show."""
    called = []
    monkeypatch.setattr(kaggle_source, "_run_kaggle", lambda *a, **k: called.append(1))

    with pytest.raises(KaggleAuthError, match="No Kaggle credential"):
        acquire(
            DATASET,
            tmp_path / "raw",
            allow_fallback=False,
            config_path=tmp_path / "absent.json",
        )

    assert not called, "the CLI must not be invoked without a credential"


def test_cache_hit_performs_no_download(tmp_path, token, monkeypatch):
    """A warm cache must short-circuit before any CLI invocation.

    Re-downloading a 166 MB archive on every ingest run is slow, and a rate limit would
    then break reruns that should have cost nothing.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "application_train.csv").write_text("SK_ID_CURR,TARGET\n1,0\n")

    def explode(*a, **k):
        raise AssertionError("download attempted despite a warm cache")

    monkeypatch.setattr(kaggle_source, "_run_kaggle", explode)

    result = acquire(COMPETITION, raw, config_path=token)

    assert result.from_cache is True
    assert result.source_used == "home-credit-default-risk"
    assert result.is_fallback is False


def test_partial_extraction_is_not_a_cache_hit(tmp_path):
    """An interrupted run leaves some members behind; that must not read as complete."""
    raw = tmp_path / "raw"
    raw.mkdir()
    spec = SourceSpec(
        source_kind="kaggle_competition",
        source_ref="c",
        primary_table="a.csv",
        archive_members=("a.csv", "b.csv"),
    )
    (raw / "a.csv").write_text("x")

    assert is_cached(spec, raw) is False

    (raw / "b.csv").write_text("y")
    assert is_cached(spec, raw) is True


def test_fallback_records_source_used_and_flags_the_substitution(
    tmp_path, token, monkeypatch, caplog
):
    """A fallback must be visible in the record, not just in a log line.

    For credit the fallback is a different dataset -- UCI Taiwan, 30k x 24 against Home
    Credit's 307k x 122 -- so a model trained on it is not comparable to one trained on
    the primary. Anything downstream needs to be able to tell which it got.
    """
    monkeypatch.setattr(
        kaggle_source, "_run_kaggle", lambda *a, **k: _result(1, stderr="403 Forbidden")
    )

    with caplog.at_level("WARNING"):
        result = acquire(COMPETITION, tmp_path / "raw", config_path=token)

    assert result.is_fallback is True
    assert result.source_used == "42477", "the fallback's identity must be recorded"
    assert result.source_used != COMPETITION.source_ref

    warning = caplog.text
    assert "DIFFERENT dataset" in warning, (
        "a non-equivalent substitution must say so, or it reads as a routine retry"
    )


def test_fallback_is_suppressible(tmp_path, token, monkeypatch):
    """Ingest must be able to demand the primary and fail loudly instead of substituting."""
    monkeypatch.setattr(
        kaggle_source, "_run_kaggle", lambda *a, **k: _result(1, stderr="403 Forbidden")
    )

    with pytest.raises(KaggleConsentError):
        acquire(COMPETITION, tmp_path / "raw", allow_fallback=False, config_path=token)


def test_competition_and_dataset_use_different_subcommands():
    """`datasets download -d` on a competition slug fails with a confusing not-found."""
    competition = kaggle_source._cli_args(COMPETITION, "/tmp")
    dataset = kaggle_source._cli_args(DATASET, "/tmp")

    assert competition[:2] == ["competitions", "download"]
    assert "-c" in competition
    assert dataset[:2] == ["datasets", "download"]
    assert "-d" in dataset


def test_selective_extraction_takes_only_declared_members(tmp_path, token, monkeypatch):
    """Home Credit ships eight tables; W1 models one."""
    raw = tmp_path / "raw"
    raw.mkdir()
    archive = raw / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("application_train.csv", "SK_ID_CURR,TARGET\n1,0\n")
        bundle.writestr("bureau.csv", "SK_ID_BUREAU\n9\n")
        bundle.writestr("previous_application.csv", "SK_ID_PREV\n7\n")

    monkeypatch.setattr(kaggle_source, "_run_kaggle", lambda *a, **k: _result(0))

    acquire(COMPETITION, raw, config_path=token)

    assert (raw / "application_train.csv").is_file()
    assert not (raw / "bureau.csv").exists(), "undeclared members must not be extracted"
    assert not (raw / "previous_application.csv").exists()


def test_extraction_reports_a_missing_member_by_name(tmp_path, token, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    with zipfile.ZipFile(raw / "bundle.zip", "w") as bundle:
        bundle.writestr("something_else.csv", "x\n")

    monkeypatch.setattr(kaggle_source, "_run_kaggle", lambda *a, **k: _result(0))

    with pytest.raises(KaggleSourceError, match="application_train.csv"):
        acquire(COMPETITION, raw, allow_fallback=False, config_path=token)


def test_non_403_failure_is_neither_auth_nor_consent(tmp_path, token, monkeypatch):
    """A network timeout must not be reported as a credential or a consent problem."""
    monkeypatch.setattr(
        kaggle_source,
        "_run_kaggle",
        lambda *a, **k: _result(1, stderr="Connection timed out after 30s"),
    )

    with pytest.raises(KaggleSourceError) as excinfo:
        acquire(DATASET, tmp_path / "raw", allow_fallback=False, config_path=token)

    assert not isinstance(excinfo.value, (KaggleAuthError, KaggleConsentError))
    assert "timed out" in str(excinfo.value)


def test_describe_blocker_names_the_token_then_the_rules(tmp_path, token):
    """An operator should learn what stands in the way before a long download, not after."""
    no_token = describe_blocker(COMPETITION, config_path=tmp_path / "absent.json")
    assert no_token is not None and "kaggle.com/settings" in no_token

    with_token = describe_blocker(COMPETITION, config_path=token)
    assert with_token is not None
    assert "rules" in with_token
    assert "403 here means consent" in with_token

    # A dataset with a token in place has nothing blocking it.
    assert describe_blocker(DATASET, config_path=token) is None


def test_acquisition_is_immutable():
    """The acquisition record is evidence; it must not be edited after the fact."""
    record = Acquisition(
        path=Path("x"),
        source_used="s",
        from_cache=False,
        is_fallback=False,
    )
    # FrozenInstanceError specifically -- a blind Exception would pass on a typo'd
    # attribute name too, which is the opposite of what this asserts.
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.source_used = "something-else"  # type: ignore[misc]


# --- Fallback resolution (plan Step 3 acceptance) -------------------------------------
#
# "the fallback resolves through the same resolver as the primary (mock permitted) so a
# typo'd OpenML id fails". Before this, a fallback was only ever asserted non-None, so a
# source_ref of "424777" passed every test and would have failed only at the moment the
# fallback was needed -- which is the moment there is no time to debug it.


def test_fallback_resolves_through_the_same_resolver_as_the_primary():
    from src.data.kaggle_source import resolve, resolve_chain
    from src.data.tracks import get_track

    track = get_track("credit")
    chain = resolve_chain(track.source)

    assert chain == ["home-credit-default-risk", "42477"]
    # Same function, both ends -- not a parallel code path that could diverge.
    assert resolve(track.source) == chain[0]
    assert resolve(track.source.fallback) == chain[1]


def test_a_typod_openml_id_fails_resolution():
    """The exact case the plan's acceptance names."""
    from src.data.kaggle_source import resolve

    typod = SourceSpec(
        source_kind="openml",
        source_ref="four-two-four-seven-seven",
        primary_table="whatever",
    )

    with pytest.raises(ValueError, match="numeric data id"):
        resolve(typod)


def test_resolution_catches_a_competition_slug_used_as_a_dataset():
    """The common mistake: it fails as a confusing not-found rather than an honest 403."""
    from src.data.kaggle_source import resolve

    wrong = SourceSpec(
        source_kind="kaggle_dataset",
        source_ref="home-credit-default-risk",
        primary_table="application_train.csv",
    )

    with pytest.raises(ValueError, match="owner/dataset"):
        resolve(wrong)


def test_resolution_catches_a_dataset_reference_used_as_a_competition():
    from src.data.kaggle_source import resolve

    wrong = SourceSpec(
        source_kind="kaggle_competition",
        source_ref="mlg-ulb/creditcardfraud",
        primary_table="creditcard.csv",
    )

    with pytest.raises(ValueError, match="bare slug"):
        resolve(wrong)


def test_every_registered_track_resolves_end_to_end():
    """A registry-wide check, so a bad reference cannot be committed unnoticed."""
    from src.data.kaggle_source import resolve_chain
    from src.data.tracks import get_track, registered_track_names

    for name in registered_track_names():
        chain = resolve_chain(get_track(name).source)
        assert len(chain) >= 2, f"{name} has no resolvable fallback"
        assert all(chain), f"{name} produced an empty reference"
