import pyro
import pyro.distributions as dist
import torch
from pyro.infer import Predictive


def my_pgm():
    # Weather: 0 for sunny, 1 for rainy
    weather = pyro.sample("weather", dist.Bernoulli(0.7))  # 70% chance of sunny

    # Temperature: Normal distribution, mean depends on weather
    mean_temp = torch.tensor(25.0) + (weather * torch.tensor(-10.0))  # Cooler if rainy
    temperature = pyro.sample("temperature", dist.Normal(mean_temp, 5.0))

    # Park: Decision to go to the park, depends on both weather and temperature
    # We model this as a logistic regression for simplicity
    logistic_regression = torch.sigmoid(torch.tensor(5.0) - weather * 3.0 + (temperature - 20.0) / 5.0)
    park = pyro.sample("park", dist.Bernoulli(logistic_regression))

    return weather, temperature, park


if __name__ == '__main__':
    # Sampling from the model
    predictive = Predictive(my_pgm, num_samples=1000)
    samples = predictive()

    # Inspect the samples
    # Input: {k: (type(v), v.shape) for k, v in samples.items()}
    # Output: {'weather': (<class 'torch.Tensor'>, torch.Size([1000])), 'temperature': (<class 'torch.Tensor'>, torch.Size([1000])), 'park': (<class 'torch.Tensor'>, torch.Size([1000]))}

    weather_samples = samples["weather"]
    temperature_samples = samples["temperature"]
    park_samples = samples["park"]