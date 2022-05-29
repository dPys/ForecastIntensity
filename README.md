# ForecastIntensity

![Elastic-Net Forecaster](RealVsPredicted.png)

Install
-------
```
docker build -t forecastintensity .
```

Usage
-----
To begin generating predictions, run the command below, using a local directory to mount of your choice.
```
docker run -it --rm -p 80:80 -v {/path/to/a/local/working/directory/to/store/predictions}:/working forecastintensity
```
