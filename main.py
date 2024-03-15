import pyro
import pyro.distributions as dist
import torch
from pyro.infer import Predictive
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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


def plot_pgm():
    # Create the graph
    G = nx.DiGraph()

    # Add nodes
    G.add_node("weather", name="Weather")
    G.add_node("temperature", name="Temperature")
    G.add_node("park", name="Go To Park")

    # Add edges
    G.add_edge("weather", "temperature")
    G.add_edge("weather", "park")
    G.add_edge("temperature", "park")

    # Draw the graph
    pos = nx.spring_layout(G)
    labels = {node: G.nodes[node]["name"] for node in G.nodes()}
    nx.draw(G, pos, with_labels=False, node_color="lightblue", node_size=1000, arrowsize=12, font_size=15, font_weight="bold")
    nx.draw_networkx_labels(G, pos, labels=labels)

    # Save the graph
    plt.savefig("pgm.pdf")


if __name__ == '__main__':
    # Sampling from the model
    predictive = Predictive(my_pgm, num_samples=1000)
    samples = predictive()

    # Inspect the samples
    weather_samples = samples["weather"]
    temperature_samples = samples["temperature"]
    park_samples = samples["park"]
    assert len(weather_samples) == len(temperature_samples) == len(park_samples), "Number of samples must be the same"

    # Print some values
    n_values = 10
    idx = np.random.choice(len(weather_samples), n_values)
    print(f"Weather: {weather_samples[idx]}")  # e.g. tensor([1., 0., 1., 1., 0., 1., 0., 1., 0., 1.])
    print(f"Temperature: {temperature_samples[idx]}")  # e.g. tensor([24.9238, 25.0549, 20.1630, 15.2604, 24.7132, 13.7514, 25.5131, 16.5955, 31.1766,  9.1974])
    print(f"Go To Park: {['Yes' if psample > 0 else 'No' for psample in park_samples[idx]]}")  # e.g. ['Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes']

    # Plot and save the PGM
    plot_pgm()