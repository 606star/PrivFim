import csv

from experiments.prepare_large_data import convert_raw_files


def test_census_income_conversion_separates_weight_and_label(tmp_path):
    raw_path = tmp_path / "census-income.data"
    rows = []
    for value in (0, 10, 20):
        row = [str(index) for index in range(42)]
        for index in (0, 5, 16, 17, 18, 30, 39):
            row[index] = str(value)
        row[24] = "999999"  # instance weight is intentionally ignored.
        row[41] = "- 50000." if value < 10 else "50000+."
        rows.append(row)
    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)

    output_path = tmp_path / "CensusIncomeKDD.csv"
    metadata = convert_raw_files((raw_path,), output_path, quantile_bins=3)

    assert metadata["rows"] == 3
    assert metadata["attributes"] == 40
    assert metadata["includes_income_label"] is False
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        converted = list(reader)
    assert header == [str(index) for index in range(40)]
    assert all(len(row) == 40 for row in converted)
    assert all(row[24] != "999999" for row in converted)
    with (tmp_path / "CensusIncomeKDD_target.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        target_rows = list(csv.reader(handle))
    assert target_rows == [["0"], ["0"], ["1"], ["1"]]
