# deeplearn24

## Start MLFlow

```
sudo docker compose up mlflow -d
```

## Open MLFlow (via VPN)

[http://cicero.nb.no:5400](http://cicero.nb.no:5400)


## Log to MLFlow (via VPN)

```python
import mlflow

mlflow.set_tracking_uri("http://cicero.nb.no:5400")
mlflow.set_experiment("check-cicero-connection")

with mlflow.start_run():
    mlflow.log_metric("foo", 1)
    mlflow.log_metric("bar", 2)
```