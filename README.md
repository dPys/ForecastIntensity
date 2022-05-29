# ForecastIntensity

![Elastic-Net Forecaster](RealVsPredicted.png)

Install
-------
```
docker build -t forecastintensity .
```

Usage
-----
To generate predictions, 
```
docker run -it --rm -p 80:80 -v forecastintensity
```
