import pytest

from vyapar_ai.catalog import generate_catalog
from vyapar_ai.retrieval import SimilarityIndex
from vyapar_ai.risk import RiskFeatures, RiskModel


@pytest.fixture(scope="module")
def catalog():
    return generate_catalog(1500, seed=1)


def test_snapshot_uses_subcategory_when_data_is_dense(catalog):
    idx = SimilarityIndex(catalog)
    snap = idx.market_snapshot("pink printed cotton kurti", "kurti", "women_ethnic",
                               fabric="cotton", pattern="printed", color="pink")
    assert snap.level == "subcategory"
    assert snap.support >= 8
    assert 200 < snap.median_price < 360
    assert snap.band_low < snap.median_price < snap.band_high


def test_snapshot_falls_back_when_subcategory_missing(catalog):
    no_dupatta = [l for l in catalog if l.subcategory != "dupatta"]
    idx = SimilarityIndex(no_dupatta)
    snap = idx.market_snapshot("red floral dupatta", "dupatta", "women_ethnic")
    assert snap.level in ("category", "global")


def test_risk_model_learns_size_chart_lever(catalog):
    model = RiskModel().fit(catalog)
    f = RiskFeatures(subcategory="kurti", category="women_ethnic", image_count=2)
    ret, rto = model.predict(f)
    assert 0.1 < ret < 0.45 and 0.05 < rto < 0.4
    levers = {l["lever"]: l for l in model.lever_impacts(f)}
    assert "add_size_chart" in levers
    assert levers["add_size_chart"]["return_rate_after"] < ret


def test_higher_cod_share_raises_rto(catalog):
    model = RiskModel().fit(catalog)
    low = model.predict(RiskFeatures("kurti", "women_ethnic", cod_share=0.5))[1]
    high = model.predict(RiskFeatures("kurti", "women_ethnic", cod_share=0.9))[1]
    assert high > low


def test_risk_model_falls_back_when_lightgbm_cannot_load(monkeypatch, catalog):
    import builtins
    from sklearn.ensemble import HistGradientBoostingRegressor
    real_import = builtins.__import__

    def broken(name, *args, **kwargs):
        if name == "lightgbm":
            raise OSError("Library not loaded: @rpath/libomp.dylib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken)
    with pytest.warns(UserWarning, match="LightGBM unavailable"):
        model = RiskModel()
    assert isinstance(model.ret_model, HistGradientBoostingRegressor)
    model.fit(catalog)
    ret, rto = model.predict(RiskFeatures("kurti", "women_ethnic"))
    assert 0.05 < ret < 0.5 and 0.05 < rto < 0.5
