from severity_estimation.severity.risk_threshold import RiskThreshold


def test_risk_threshold():
    rt = RiskThreshold([0.2, 0.7, 0.9], ["low", "medium", "high", "catastrophic"])
    assert rt.range_membership(0, 0.8) == ["low", "medium", "high"]
    assert rt.range_membership(0.1, 0.5) == ["low", "medium"]
    assert rt.range_membership(0.5, 0.6) == ["medium"]
    assert rt.membership(0.1) == "low"
    assert rt.membership(0.5) == "medium"
    assert rt.membership(0.8) == "high"
    assert rt.membership(0.99) == "catastrophic"
    assert rt.membership(1.0) == "catastrophic"
