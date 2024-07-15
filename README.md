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

## Source for book
https://bdh-rd.bne.es/viewer.vm?lang=en&id=0000194858&page=1


## Install PDFImages to extract images from the book PDF
```
sudo apt install poppler-utils
```