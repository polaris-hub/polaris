from polaris.hub.client import PolarisHubClient

client = PolarisHubClient()
# Try loading a dataset
dataset = client.get_dataset(owner="polaris", slug="posebusters-v1")
print(f"Dataset: {dataset.name}, rows: {len(dataset.table)}")
