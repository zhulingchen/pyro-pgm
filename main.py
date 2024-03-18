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

    # Mean and standard deviation of temperature are stochastic variables that depend on weather
    # Mean temperature: Normally distributed around 25 if sunny, 15 if rainy
    mean_temp_loc = torch.tensor(25.0) - weather * torch.tensor(10.0)
    mean_temp = pyro.sample("mean_temp", dist.Normal(mean_temp_loc, torch.tensor(2.0)))
    # Standard deviation of temperature: Deterministic, 5 if sunny, 2.5 if rainy
    std_temp_v = torch.tensor(5.0) - weather * torch.tensor(2.5)
    std_temp = pyro.sample("std_temp", dist.Delta(std_temp_v))

    # Temperature: Normally distributed around mean_temp with std_temp
    temperature = pyro.sample("temperature", dist.Normal(mean_temp, std_temp))

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
    # e.g. [1, 1, 0, 1, 0, 0, 1, 1, 1, 0]
    print(f"{n_values} Weather samples: {weather_samples[idx].int().tolist()}")
    # e.g. [18.10348892211914, 13.129755020141602, 21.422134399414062, 11.453704833984375, 29.757862091064453, 14.479151725769043, 16.184894561767578, 11.41153335571289, 14.973945617675781, 32.28795623779297]
    print(f"{n_values} Temperature samples: {temperature_samples[idx].tolist()}")
    # e.g. ['Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes']
    print(f"{n_values} Go To Park samples: {['Yes' if psample > 0 else 'No' for psample in park_samples[idx]]}")

    # Plot and save the PGM
    # Automated method provided by Pyro using graphviz
    pyro.render_model(
        my_pgm,
        filename="my_pgm.png",
        render_distributions=True,
        render_params=True,
        render_deterministic=True
    )

    # Manual method
    # plot_pgm(filename="my_pgm.png")