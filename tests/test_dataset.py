from polaris.dataset import DatasetInfo, Dataset


def test_dataset_info_serialization(test_dataset_info):
    serialized = test_dataset_info.serialize()
    recovered = DatasetInfo.deserialize(serialized)
    assert recovered == test_dataset_info


def test_modality_coverage(test_data, test_dataset_info):
    assert any(c not in test_dataset_info.modalities for c in test_data.columns)
    Dataset(test_data, test_dataset_info)
    assert all(c in test_dataset_info.modalities for c in test_data.columns)
