"""Tests for the Track seam.

The load-bearing test here is :func:`test_serving_entrypoint_does_not_import_pandera`.
The rest check the registry's shape; that one checks the property the seam exists for.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.data.tracks import TRACKS, SchemaSpec, get_track, registered_track_names
from src.features.specs import FEATURE_SPECS, FeatureSpec, get_feature_spec


def test_serving_entrypoint_does_not_import_pandera():
    """Importing the API must not pull pandera into the process.

    This is the executable form of the placement decision. Composition alone does not
    sever the edge: Python imports at module granularity, so if the pandera-bearing
    SchemaSpec and the feature lists shared a module, importing the API for its feature
    contract would drag pandera -- and the whole validation stack -- into the serving
    image, where a 0.5 GB registry budget is already contested.

    Run in a subprocess because pytest has almost certainly imported pandera already via
    a sibling test module; checking sys.modules in-process would pass for the wrong
    reason and keep passing after a regression.
    """
    probe = (
        "import sys; import src.api.main; "
        "leaked = sorted(m for m in sys.modules if m == 'pandera' or m.startswith('pandera.')); "
        "print('LEAKED' if leaked else 'CLEAN')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=120,
        # Asserted explicitly below so a failed probe reports its stdout and stderr.
        # check=True would raise CalledProcessError and throw that diagnostic away.
        check=False,
    )

    assert result.returncode == 0, (
        f"importing src.api.main failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "CLEAN" in result.stdout, (
        "src.api.main transitively imports pandera. The data -> features dependency "
        "direction is inverted somewhere: main.py must reach feature contracts through "
        "src.features.specs, never through src.data.tracks."
    )


def test_every_registered_track_has_a_source_and_a_fallback():
    """A track without a fallback is one auth stall away from stalling the deploy."""
    assert TRACKS, "no tracks registered"

    for name, track in TRACKS.items():
        assert track.name == name, f"{name!r} is registered under a mismatched key"
        assert track.source.primary_table, f"{name!r} has an empty primary_table"
        assert track.source.fallback is not None, (
            f"{name!r} has no fallback source; fallbacks are code, not prose"
        )
        assert track.source.fallback.primary_table, f"{name!r} fallback has an empty primary_table"


def test_both_tracks_are_registered_and_resolve_to_separate_state():
    """The fraud track landed in W2 Step 8, and it must not share anything with credit.

    Registration alone is not the property worth asserting -- a registry entry that
    resolved to the same parquet, the same cache directory or the same registered model as
    credit would read as two tracks while being one. Every path a track owns is derived
    from its name, and this is what checks that the derivation actually separates them:
    DoD (7)'s "drift in one track retrains only that track" is false the moment any of
    these collide.
    """
    fraud = get_track("fraud")
    credit = get_track("credit")

    assert fraud.name == "fraud"
    assert fraud.schema.module_path == "src.data.fraud"

    assert fraud.processed_path != credit.processed_path
    assert fraud.raw_dir != credit.raw_dir
    assert fraud.model_name != credit.model_name
    assert fraud.experiment_name != credit.experiment_name
    assert fraud.schema.manifest_path != credit.schema.manifest_path


def test_the_two_tracks_share_no_column_at_all():
    """The no-join ruling, executable.

    ``AGENTS.md`` rules that these datasets are never joined: no shared key, no shared
    entity, no shared time base, no shared feature space. Prose cannot stop someone
    proposing a join later; an overlap of even one column name is where such a proposal
    would start, and a shared name would also make the interleaved prediction log
    ambiguous about which track a record came from.
    """
    credit = get_feature_spec("credit")
    fraud = get_feature_spec("fraud")

    credit_columns = {credit.id_column, credit.target_column, *credit.feature_columns}
    fraud_columns = {fraud.id_column, fraud.target_column, *fraud.feature_columns}

    assert not credit_columns & fraud_columns, (
        f"the tracks share {sorted(credit_columns & fraud_columns)}; they are ruled"
        " un-joinable and a shared column name is where a join starts"
    )


def test_fraud_needs_no_rule_acceptance_which_is_why_it_is_fetched_first():
    """A dataset needs a token; a competition needs a token *and* a browser consent action.

    The ordering argument depends on this asymmetry. Fetching fraud first proves the
    credential in isolation, so a later 403 on the credit competition is diagnosable as
    missing consent rather than a bad key -- which is the misdiagnosis the two exception
    types exist to prevent.
    """
    fraud = get_track("fraud").source

    assert not fraud.requires_rule_acceptance
    assert get_track("credit").source.requires_rule_acceptance


def test_fraud_fallback_is_equivalent_but_lands_under_its_own_filename():
    """OpenML 1597 is the same ULB extract by another route, unlike the credit fallback.

    The filename is the load-bearing part. Were the fallback to write ``creditcard.csv``,
    a fallback artefact on disk would be indistinguishable from a primary one afterwards,
    and ``is_cached`` would read a fallback fetch as a primary cache hit on the next run.
    """
    fraud = get_track("fraud").source

    assert fraud.equivalent_to_primary is True
    assert fraud.fallback is not None
    assert fraud.fallback.source_ref == "1597"
    assert fraud.fallback.primary_table != fraud.primary_table


def test_fraud_source_and_fallback_both_resolve():
    """A fallback that cannot resolve is a fallback that does not exist.

    Resolved through the real resolver rather than checked for being non-``None``: a
    ``source_ref`` of ``"15977"`` would pass every structural check and fail only at the
    moment the fallback was actually needed, which is the moment there is no time to debug
    it. The dataset/competition distinction is also enforced here -- a bare slug under
    ``kaggle_dataset`` raises.
    """
    from src.data.kaggle_source import resolve_chain

    assert resolve_chain(get_track("fraud").source) == ["mlg-ulb/creditcardfraud", "1597"]


def test_fraud_models_twenty_nine_numeric_columns_and_no_categoricals():
    """Zero categoricals is the fraud shape, and 29 modeled columns is the count.

    31 source columns, minus ``Class`` (the target) and minus ``Time`` (dropped on
    judgement: no live caller can produce seconds-since-this-extract, it is monotonic in
    row order, and it would register as drifted every run). The id column is derived rather
    than taken from the source, so it does not come out of the 31.
    """
    spec = get_feature_spec("fraud")

    assert spec.categorical_features == ()
    assert spec.derived_features == (), "no sentinels, so nothing to derive"
    assert dict(spec.sentinels) == {}
    assert len(spec.feature_columns) == 29
    assert "Amount" in spec.feature_columns
    assert "Time" not in spec.feature_columns, "Time is validated but not modeled"
    assert spec.target_column not in spec.feature_columns


def test_fraud_component_names_are_left_unmapped_rather_than_invented():
    """V1..V28 are PCA components whose loadings were never published.

    ``display_name`` falling back to the raw column is what makes an honest reason code
    possible: "V14" says exactly as much as is known. A plausible-sounding invented label
    would say more than is known, which is worse than saying nothing, and DoD (3) is about
    reasons a human can act on rather than sentences that read well.
    """
    spec = get_feature_spec("fraud")

    assert spec.display_name("Amount") == "Transaction amount"
    for component in ("V1", "V14", "V28"):
        assert spec.display_name(component) == component


def test_fraud_schema_and_feature_contract_agree_on_the_component_set():
    """Both generate V1..V28 from a range; nothing but this stops them diverging.

    They are separate lists on purpose -- one is what ingest *validates*, the other is what
    the model *consumes*, and those are legitimately different sets (``Time`` is in the
    first and not the second). A component added to one and not the other would mean a
    column the schema accepts and the model never sees, or worse, one the model expects and
    nothing validates.
    """
    from src.data import fraud

    modeled = set(get_feature_spec("fraud").feature_columns) - {"Amount"}

    assert set(fraud.V_COLUMNS) == modeled
    assert len(fraud.V_COLUMNS) == 28
    assert set(fraud.FraudSchema.columns) >= set(fraud.V_COLUMNS) | {"Time", "Amount", "Class"}


def test_fraud_derived_id_is_absent_from_the_source_manifest():
    """The manifest describes the raw frame, and the id column is not in it.

    ``assert_fingerprint`` runs *before* ``clean``, so a manifest listing
    ``TransactionIndex`` would demand a column the source has never shipped and fail every
    single ingest. The inverse mistake is equally available -- deriving the key in a way
    that leaves it out of the parquet -- and ``ingest``'s keep list is what covers that.
    """
    schema = get_track("fraud").schema
    names = [line for line in schema.manifest_path.read_text().splitlines() if line.strip()]

    assert get_feature_spec("fraud").id_column not in names
    assert "Time" in names, "Time is a source column even though it is not modeled"


def test_fraud_manifest_matches_the_real_header():
    """A fingerprint's entire value is byte-faithfulness to the source.

    31 plausible unique names that are not the ones the file carries would make the
    upstream-change detector assert something other than what the data is. Read from the
    extracted CSV when ingest has run, and otherwise straight out of the cached archive,
    because the zip is the artefact that actually persists between runs.
    """
    import csv
    import io
    import zipfile

    schema = get_track("fraud").schema
    names = [line for line in schema.manifest_path.read_text().splitlines() if line.strip()]

    assert len(names) == 31, f"expected 31 column names, found {len(names)}"
    assert len(set(names)) == len(names), "duplicate column names in the manifest"
    assert names[0] == "Time"
    assert names[-1] == "Class"

    raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw" / "fraud"
    csv_path = raw_dir / "creditcard.csv"
    archive = raw_dir / "creditcardfraud.zip"

    if csv_path.is_file():
        header_line = csv_path.open(encoding="utf-8").readline()
    elif archive.is_file():
        with zipfile.ZipFile(archive) as bundle, bundle.open("creditcard.csv") as handle:
            header_line = io.TextIOWrapper(handle, encoding="utf-8").readline()
    else:
        pytest.skip("no fraud archive present; data/ is gitignored and CI has no copy")

    # csv.reader rather than a split on commas: this header is quoted ("Time","V1",...)
    # where Home Credit's is not, and a naive split would compare 'Time' against '"Time"'.
    header = next(csv.reader([header_line]))

    assert names == header, (
        "manifest does not match the real header; first divergence at index "
        f"{next((i for i, (a, b) in enumerate(zip(names, header)) if a != b), len(header))}"
    )


def test_unknown_track_error_names_the_registered_tracks():
    with pytest.raises(KeyError) as excinfo:
        get_track("telco")

    assert "telco" in str(excinfo.value)
    assert "registered tracks" in str(excinfo.value)


def test_model_names_are_track_derived():
    """Two registered models, not one shared name.

    Independent registry entries are what let one track retrain without touching the
    other, which is the property DoD (7) demonstrates.
    """
    credit = get_track("credit")
    assert credit.model_name == "riskwatch_credit"
    assert credit.experiment_name == "riskwatch_credit"

    names = {get_track(n).model_name for n in registered_track_names()}
    assert len(names) == len(TRACKS), "two tracks share a registered-model name"


def test_credit_source_requires_rule_acceptance():
    """Home Credit is a competition, so it needs a browser action no API call performs.

    The fallback is auth-free, which is the point of having one.
    """
    source = get_track("credit").source
    assert source.requires_rule_acceptance
    assert source.fallback is not None
    assert not source.fallback.requires_rule_acceptance


def test_credit_fallback_is_flagged_as_a_different_dataset():
    """The credit fallback is UCI Taiwan, not Home Credit by another route.

    Taking it means rewriting the schema and feature spec. The flag is what stops a
    deadline substitution being recorded as an equivalent swap.
    """
    assert get_track("credit").source.equivalent_to_primary is False


def test_feature_columns_are_pinned_and_exclude_non_features():
    spec = get_feature_spec("credit")

    assert spec.feature_columns == tuple(spec.numeric_features) + tuple(spec.categorical_features)
    assert spec.id_column not in spec.feature_columns
    assert spec.target_column not in spec.feature_columns
    assert spec.non_feature_columns == (spec.id_column, spec.target_column)


def test_display_names_fall_back_to_the_raw_column():
    """Reason codes need human sentences; unmapped columns degrade, they do not crash."""
    spec = get_feature_spec("credit")

    assert spec.display_name("DAYS_BIRTH") == "Age (years)"
    assert spec.display_name("NOT_A_REAL_COLUMN") == "NOT_A_REAL_COLUMN"


def test_positive_label_is_the_already_encoded_integer():
    """Both riskwatch tracks ship integer 0/1 targets.

    The Telco pipeline hardcoded {"Yes": 1, "No": 0} and raised on anything else, so this
    contract is the first thing the credit track would otherwise have broken on.
    """
    assert get_feature_spec("credit").positive_label == 1


def test_feature_spec_rejects_a_column_claimed_as_both_numeric_and_categorical():
    with pytest.raises(ValueError, match="both numeric and categorical"):
        FeatureSpec(
            id_column="id",
            target_column="y",
            positive_label=1,
            numeric_features=("a", "b"),
            categorical_features=("b",),
        )


def test_feature_spec_rejects_a_target_listed_as_a_feature():
    """Leaking the target into the feature list is a silent, total leak."""
    with pytest.raises(ValueError, match="non-feature column"):
        FeatureSpec(
            id_column="id",
            target_column="y",
            positive_label=1,
            numeric_features=("a", "y"),
            categorical_features=(),
        )


def test_feature_spec_rejects_an_empty_feature_set():
    with pytest.raises(ValueError, match="no features at all"):
        FeatureSpec(
            id_column="id",
            target_column="y",
            positive_label=1,
            numeric_features=(),
            categorical_features=(),
        )


def test_registries_agree_on_which_tracks_exist():
    """A track registered without a feature spec would fail at construction, not lookup."""
    assert set(TRACKS) <= set(FEATURE_SPECS)


def test_training_and_serving_agree_on_the_registry_name():
    """The name train.py registers under must equal the name serving looks up.

    These live in different modules for a real reason -- importing train.py into the API
    would pull sklearn and LightGBM into the serving image -- so the two derive the name
    independently and nothing but this test stops them drifting apart.

    A disagreement is invisible until deploy: training writes riskwatch_credit, serving
    asks for something else, and the failure surfaces as "model not found" against a
    registry that plainly contains a model.
    """
    from src.models.train import experiment_name_for, model_name_for

    for name in registered_track_names():
        track = get_track(name)
        assert model_name_for(name) == track.model_name
        assert experiment_name_for(name) == track.experiment_name


def test_api_default_model_uri_matches_the_registered_name():
    """The serving default must resolve to the name training actually registers."""
    from src.api.main import model_uri_for

    track = get_track("credit")
    assert model_uri_for("credit") == f"models:/{track.model_name}@production"


def test_has_manifest_distinguishes_declaration_from_existence():
    """A declared manifest path is not a manifest.

    This distinction was worth encoding: the credit manifest could not be authored until
    the archive landed, because reproducing 122 exact column names from memory would be
    fabrication and the fingerprint's whole value is byte-faithfulness. A property that
    answered True for a declared-but-absent file would have told the Step 4 executor it
    was ready.

    The manifest now exists, written from the archive. The invariant under test is the
    distinction itself, not the current answer -- so the absent case is exercised with a
    path that genuinely is not there.
    """
    schema = get_track("credit").schema

    assert schema.manifest_declared is True
    assert schema.has_manifest is True, "written from application_train.csv"

    absent = SchemaSpec(model=None, manifest_path=Path("/nonexistent/columns.txt"))
    assert absent.manifest_declared is True, "declared"
    assert absent.has_manifest is False, "but not on disk -- the distinction that matters"

    undeclared = SchemaSpec(model=None, manifest_path=None)
    assert undeclared.manifest_declared is False
    assert undeclared.has_manifest is False


def test_credit_manifest_matches_the_real_column_count():
    """The manifest is a fingerprint, so its contents must match the source exactly.

    122 is the published width of application_train.csv. A manifest that drifted from it
    would make the upstream-change detector assert something other than what the data is.
    """
    schema = get_track("credit").schema
    if not schema.has_manifest:
        pytest.skip("manifest not written yet; requires the Home Credit archive")

    names = [line for line in schema.manifest_path.read_text().splitlines() if line.strip()]

    assert len(names) == 122, f"expected 122 column names, found {len(names)}"
    assert len(set(names)) == len(names), "duplicate column names in the manifest"
    assert names[0] == "SK_ID_CURR"
    assert "TARGET" in names

    # Every modeled column must appear in the manifest, or the feature contract and the
    # structural check disagree about what the dataset contains.
    spec = get_feature_spec("credit")
    missing = [c for c in spec.feature_columns if c not in names]
    assert not missing, f"modeled columns absent from the manifest: {missing}"

    # Against the real header when it is available. Without this the manifest could be
    # well-formed and wrong -- 122 plausible unique names that are not the ones the file
    # actually carries -- and a fingerprint whose whole value is byte-faithfulness would
    # be asserting something other than what the data does. Skipped rather than failed
    # when the archive is absent, since data/ is gitignored and CI has no copy.
    raw = Path(__file__).resolve().parents[1] / "data" / "raw" / "credit" / "application_train.csv"
    if not raw.is_file():
        return

    with raw.open(encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")

    assert names == header, (
        "manifest does not match the real header; "
        f"first divergence at index {next((i for i, (a, b) in enumerate(zip(names, header)) if a != b), len(header))}"
    )


def test_schema_model_is_declared_unbuilt_rather_than_silently_absent():
    """get_track('credit') must not read as fully wired while validation is missing."""
    assert get_track("credit").schema.model is None


def test_export_default_uri_names_a_registered_model():
    """The baked-export default must resolve to a model training actually registers.

    export.py cannot import train.py (that would pull sklearn and LightGBM into anything
    touching export), so the name is derived in both places. Nothing but this test stops
    them disagreeing -- and a stale singular "riskwatch" here fails as RESOURCE_DOES_NOT_
    EXIST, which reads as an empty registry rather than a wrong constant.
    """
    from src.models.export import DEFAULT_MODEL_URI

    registered = {get_track(n).model_name for n in registered_track_names()}
    name = DEFAULT_MODEL_URI.removeprefix("models:/").split("@")[0]

    assert name in registered, f"{name!r} is not registered; known: {sorted(registered)}"


def test_resolve_track_round_trips_every_registered_model_name():
    """The label an export bakes in must be the track name the API compares against.

    ``resolve_track`` parses the registered model name rather than taking a ``--track``
    argument, so the label can never disagree with the artifact that was downloaded. What it
    *can* disagree with is the naming convention, and then a correctly exported artifact gets
    stamped with a track the serving side has never heard of -- which fails as
    "``MODEL_TRACK`` says X, we are serving Y", pointing at the deployment rather than at the
    prefix. This asserts the round trip for every track there is, so the convention breaking is
    what fails.
    """
    import pytest

    from src.models.export import resolve_track

    for track_name in registered_track_names():
        model = get_track(track_name).model_name
        assert resolve_track(f"models:/{model}@production") == track_name
        assert resolve_track(f"models:/{model}/3") == track_name

    # A local path names no registered model, so there is nothing to derive and guessing is the
    # failure mode being removed. The version form above is included because it takes a
    # different branch and would otherwise be unparsed.
    with pytest.raises(ValueError, match="only a models:/ URI"):
        resolve_track("/app/model")
    with pytest.raises(ValueError, match="does not follow"):
        resolve_track("models:/churnwatch@production")


def test_training_and_drift_resolve_the_same_processed_snapshot():
    """Train and drift must read the same file, or the model and its drift reference
    describe different data and the retrain trigger measures against the wrong baseline."""
    from src.models.train import processed_path_for
    from src.monitoring.drift import reference_path

    for name in registered_track_names():
        track = get_track(name)
        assert processed_path_for(name) == track.processed_path
        assert reference_path(name) == track.processed_path


def test_deployment_configs_name_registered_models():
    """docker-compose.yml and .env.example are the documented ways to configure serving.

    Neither is exercised by any test that stubs load_model, so a stale name there breaks
    every containerized path while the suite stays green -- which is exactly what happened.
    """
    import re

    registered = {get_track(n).model_name for n in registered_track_names()}
    root = Path(__file__).resolve().parents[1]

    for filename in ("docker-compose.yml", ".env.example"):
        text = (root / filename).read_text()
        for uri in re.findall(r"models:/([A-Za-z0-9_\-]+)@", text):
            assert uri in registered, f"{filename} names unregistered model {uri!r}"
