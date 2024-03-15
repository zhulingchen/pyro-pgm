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


def plot_pgm(filename):
    # Create the graph
    G = nx.DiGraph()

    # Add nodes
    G.add_node("weather", name="Weather")
    G.add_node("temperature", name="Temperature")
    G.add_node("park", name="Go To Park?")

    # Add edges
    G.add_edge("weather", "temperature")
    G.add_edge("weather", "park")
    G.add_edge("temperature", "park")

    # Draw the graph
    pos = nx.spring_layout(G)
    labels = {name: node["name"] for name, node in G.nodes().items()}
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_color="lightblue",
        node_size=2000,
        arrowsize=15,
        font_size=15,
        font_weight="bold"
    )
    nx.draw_networkx_labels(G, pos, labels=labels)
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels={
            ("weather", "temperature"): "mean = 25 - 10 * weather",
            ("weather", "park"): "logistic(5 - 3 * weather + (temperature - 20) / 5)",
            ("temperature", "park"): "logistic(5 - 3 * weather + (temperature - 20) / 5)"
        },
        font_size=8,
        font_color='red'
    )

    # Save the graph
    plt.savefig(filename)


if __name__ == '__main__':
    # Sampling from the model
    num_samples = 1000
    predictive = Predictive(my_pgm, num_samples=num_samples)
    samples = predictive()

    # Inspect the samples
    weather_samples = samples["weather"]
    temperature_samples = samples["temperature"]
    park_samples = samples["park"]

    # Print some values
    n_values = 10
    idx = np.random.choice(num_samples, n_values)
    # e.g. [1, 1, 1, 1, 1, 1, 0, 0, 1, 0]
    print(f"{n_values} Weather samples: {weather_samples[idx].int().tolist()}")
    # e.g. [13.32525634765625, 18.57118797302246, 20.02625846862793, 25.918180465698242, 15.915643692016602, 22.78240203857422, 18.47606658935547, 20.850561141967773, 0.4364767074584961, 30.41807746887207]
    print(f"{n_values} Temperature samples: {temperature_samples[idx].tolist()}")
    # e.g. ['Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes']
    print(f"{n_values} Go To Park samples: {['Yes' if psample > 0 else 'No' for psample in park_samples[idx]]}")

    # Plot and save the PGM
    plot_pgm(filename="my_pgm.png")