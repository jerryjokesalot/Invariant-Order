"""Tests for the Design class (observable designer / mathematical contract)."""

import numpy as np
import pytest
from invariant_order.design import Design, DesignReport


class TestDesign:
    def test_basic_construction(self):
        d = Design(nuisance="multiplicative", order=1)
        assert d.nuisance == "multiplicative"
        assert d.order == 1

    def test_invalid_nuisance(self):
        with pytest.raises(ValueError):
            Design(nuisance="quadratic")

    def test_invalid_order(self):
        with pytest.raises(ValueError):
            Design(order=0)

    def test_analyze_without_signal(self):
        d = Design(nuisance="multiplicative", order=1)
        report = d.analyze()
        assert isinstance(report, DesignReport)
        assert report.nuisance_group == "Multiplicative scale"
        assert report.order == 1
        assert report.working_space == "log"
        assert report.transformed is None
        assert report.signal_length == 0

    def test_analyze_with_signal(self):
        signal = np.random.exponential(1.0, 100)
        d = Design(nuisance="multiplicative", order=1)
        report = d.analyze(signal)
        assert report.signal_length == 100
        assert report.transformed_length == 99
        assert report.transformed is not None
        assert len(report.transformed) == 99

    def test_suppressed_order_1(self):
        report = Design("multiplicative", 1).analyze()
        assert any("constant gain" in s for s in report.suppressed)

    def test_suppressed_order_2(self):
        report = Design("multiplicative", 2).analyze()
        assert any("linear" in s for s in report.suppressed)
        assert len(report.suppressed) >= 2

    def test_suppressed_order_3(self):
        report = Design("multiplicative", 3).analyze()
        assert any("quadratic" in s for s in report.suppressed)
        assert len(report.suppressed) >= 3

    def test_preserved_populated(self):
        report = Design("multiplicative", 1).analyze()
        assert len(report.preserved) >= 3
        assert any("variability" in p.lower() for p in report.preserved)

    def test_identifiability_warnings(self):
        report = Design("multiplicative", 1).analyze()
        assert len(report.identifiability_warnings) >= 1
        assert any("cannot be distinguished" in w for w in report.identifiability_warnings)

    def test_symmetry_breakers(self):
        report = Design("multiplicative", 1).analyze()
        assert len(report.symmetry_breakers) >= 2
        assert any("reference" in b.lower() or "calibration" in b.lower()
                    for b in report.symmetry_breakers)

    def test_frequency_response_formula(self):
        report = Design("multiplicative", 2).analyze()
        assert "2" in report.frequency_response_formula
        assert report.low_freq_attenuation == "O(ω^2)"

    def test_additive_design(self):
        report = Design("additive", 1).analyze()
        assert report.nuisance_group == "Additive translation"
        assert report.working_space == "linear"
        assert any("offset" in s for s in report.suppressed)

    def test_str_output(self):
        report = Design("multiplicative", 1).analyze()
        text = str(report)
        assert "INVARIANT OBSERVABLE DESIGN CONTRACT" in text
        assert "SUPPRESSED" in text
        assert "PRESERVED" in text
        assert "IDENTIFIABILITY" in text
        assert "SYMMETRY BREAKERS" in text

    def test_str_with_signal(self):
        signal = np.random.exponential(1.0, 200)
        report = Design("multiplicative", 1).analyze(signal)
        text = str(report)
        assert "200" in text
        assert "199" in text

    def test_frequency_response(self):
        d = Design("multiplicative", 2)
        omegas, mags = d.frequency_response()
        assert len(omegas) == 200
        assert len(mags) == 200
        assert mags[0] < mags[-1]

    def test_compare_orders(self):
        d = Design("multiplicative", 1)
        omegas, results = d.compare_orders(max_order=3)
        assert len(results) == 3
        assert 1 in results and 2 in results and 3 in results
        # Higher order = more suppression at low freq
        assert results[3][0] < results[2][0] < results[1][0]

    def test_higher_order_design(self):
        report = Design("multiplicative", 5).analyze()
        assert report.order == 5
        assert "5" in report.frequency_response_formula
        assert len(report.suppressed) >= 1
