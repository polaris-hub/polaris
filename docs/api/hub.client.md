::: polaris.hub.settings.PolarisHubSettings
    options:
        filters: ["!^_"]

---


::: polaris.hub.client.PolarisHubClient
    options: 
        merge_init_into_class: true
        filters: ["!^_", "!create_authorization_url", "!fetch_token", "!request", "!token"]
---
