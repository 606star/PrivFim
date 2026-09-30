import csv

from privfim.data import load_vertical_csv


def test_attribute_ratios_use_largest_remainder_allocation(tmp_path):
    csv_path = tmp_path / "sixteen_attributes.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(range(16))
        writer.writerow(range(16))

    dataset = load_vertical_csv(
        csv_path=csv_path,
        num_clients=4,
        attribute_ratios=(2.0, 1.0, 1.0, 1.0),
    )

    assert [len(partition) for partition in dataset.partitions] == [7, 3, 3, 3]
    assert tuple(attr for part in dataset.partitions for attr in part) == tuple(range(16))


def test_partition_seed_randomizes_ownership_reproducibly(tmp_path):
    csv_path = tmp_path / "attributes.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(range(12))
        writer.writerow(range(12))

    first = load_vertical_csv(csv_path, num_clients=3, partition_seed=17)
    second = load_vertical_csv(csv_path, num_clients=3, partition_seed=17)
    different = load_vertical_csv(csv_path, num_clients=3, partition_seed=18)

    assert first.partitions == second.partitions
    assert first.partitions != different.partitions
    assert sorted(attr for part in first.partitions for attr in part) == list(range(12))


def test_affinity_partition_preserves_capacity_and_is_reproducible(tmp_path):
    csv_path = tmp_path / "correlated.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(range(6))
        for value in range(20):
            writer.writerow(
                [
                    value % 2,
                    value % 2,
                    (value // 2) % 2,
                    (value // 2) % 2,
                    value % 3,
                    value % 5,
                ]
            )

    first = load_vertical_csv(
        csv_path,
        num_clients=3,
        attribute_ratios=(2, 1, 1),
        partition_seed=17,
        partition_strategy="affinity",
    )
    second = load_vertical_csv(
        csv_path,
        num_clients=3,
        attribute_ratios=(2, 1, 1),
        partition_seed=17,
        partition_strategy="affinity",
    )
    assert first.partitions == second.partitions
    assert [len(part) for part in first.partitions] == [3, 2, 1]
    assert sorted(attr for part in first.partitions for attr in part) == list(range(6))


def test_row_sampling_is_reproducible_and_domains_use_full_file(tmp_path):
    csv_path = tmp_path / "biased_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([0, 1])
        writer.writerows([[0, 0]] * 50 + [[1, 1]] * 50)

    first = load_vertical_csv(
        csv_path, num_clients=2, max_rows=20, row_sampling_seed=17
    )
    second = load_vertical_csv(
        csv_path, num_clients=2, max_rows=20, row_sampling_seed=17
    )

    assert first.data.shape == (20, 2)
    assert (first.data == second.data).all()
    assert (first.data[:, 0] == 1).any()
    assert first.domains == ((0, 1), (0, 1))
