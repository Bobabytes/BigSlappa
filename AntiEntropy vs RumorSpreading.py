# EPIDEMIC ALGORITHMS, GO!

import random
import time
import matplotlib.pyplot as plt
from collections import defaultdict


class Node:
    def __init__(self, node_id):
        self.id = node_id
        self.data = set()
        self.connections = []
        self.infected = False  # For rumor spreading
        self.last_update = 0  # For anti-entropy

    def connect(self, node):
        self.connections.append(node)

    def add_data(self, item):
        self.data.add(item)

    def has_data(self, item):
        return item in self.data


class Network:
    def __init__(self, num_nodes, connection_prob=0.3):
        self.nodes = [Node(i) for i in range(num_nodes)]
        self._connect_nodes(connection_prob)
        self.time = 0

    def _connect_nodes(self, prob):
        for i, node in enumerate(self.nodes):
            for other in self.nodes[i + 1:]:
                if random.random() < prob:
                    node.connect(other)
                    other.connect(node)

    def spread_rumor(self, origin_node_id, data_item):
        origin = self.nodes[origin_node_id]
        origin.add_data(data_item)
        origin.infected = True
        active_nodes = [origin]
        steps = 0
        infected_counts = []

        while active_nodes:
            new_active = []
            for node in active_nodes:
                for neighbor in node.connections:
                    if not neighbor.has_data(data_item):
                        neighbor.add_data(data_item)
                        neighbor.infected = True
                        new_active.append(neighbor)
            active_nodes = new_active
            steps += 1
            infected_counts.append(sum(1 for n in self.nodes if n.has_data(data_item)))

        success = all(n.has_data(data_item) for n in self.nodes)
        return steps, infected_counts, success

    def anti_entropy(self, origin_node_id, data_item):
        origin = self.nodes[origin_node_id]
        origin.add_data(data_item)
        steps = 0
        infected_counts = []
        converged = False

        while not converged:
            converged = True
            steps += 1
            for node in self.nodes:
                if node.has_data(data_item):
                    if node.connections:
                        neighbor = random.choice(node.connections)
                        if not neighbor.has_data(data_item):
                            neighbor.add_data(data_item)
                            converged = False
            infected_counts.append(sum(1 for n in self.nodes if n.has_data(data_item)))
            if all(n.has_data(data_item) for n in self.nodes):
                converged = True

        success = all(n.has_data(data_item) for n in self.nodes)
        return steps, infected_counts, success

    def reset(self):
        for node in self.nodes:
            node.data = set()
            node.infected = False
        self.time = 0


def compare_algorithms(network_sizes=[10, 20, 50], connection_probs=[0.2, 0.5, 0.8], trials=1000):
    results = defaultdict(dict)

    for size in network_sizes:
        for prob in connection_probs:
            rumor_steps = []
            anti_steps = []
            rumor_success = 0
            anti_success = 0

            for _ in range(trials):
                net = Network(size, prob)

                # Rumor spreading
                steps, _, success = net.spread_rumor(0, "test_data")
                rumor_steps.append(steps)
                if success:
                    rumor_success += 1
                net.reset()

                # Anti-entropy
                steps, _, success = net.anti_entropy(0, "test_data")
                anti_steps.append(steps)
                if success:
                    anti_success += 1
                net.reset()

            results[(size, prob, 'rumor')] = {
                'avg_steps': sum(rumor_steps) / len(rumor_steps),
                'success_rate': rumor_success / trials
            }
            results[(size, prob, 'anti-entropy')] = {
                'avg_steps': sum(anti_steps) / len(anti_steps),
                'success_rate': anti_success / trials
            }

    return results


def plot_results(results):
    import matplotlib.pyplot as plt
    import numpy as np

    sizes = sorted({size for size, _, _ in results.keys()})
    probs = sorted({prob for _, prob, _ in results.keys()})

    for prob in probs:
        plt.figure(figsize=(10, 6))

        rumor_data = [results[(size, prob, 'rumor')]['avg_steps'] for size in sizes]
        anti_data = [results[(size, prob, 'anti-entropy')]['avg_steps'] for size in sizes]

        x = np.arange(len(sizes))
        width = 0.35
        plt.bar(x - width / 2, rumor_data, width, label='Rumor Spreading')
        plt.bar(x + width / 2, anti_data, width, label='Anti-Entropy')

        plt.xlabel('Network Size')
        plt.ylabel('Average Steps to Full Dissemination')
        plt.title(f'Algorithm Comparison (Connection Probability = {prob})')
        plt.xticks(x, sizes)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'comparison_p{prob}.png')
        plt.show()


if __name__ == "__main__":
    print("Starting simulation...")
    results = compare_algorithms()

    print("\nResults Summary:")
    for (size, prob, algo), stats in results.items():
        print(f"Network size: {size}, Conn. prob: {prob}, Algorithm: {algo:<12} "
              f"=> Avg. steps: {stats['avg_steps']:.2f}, Success rate: {stats['success_rate'] * 100:.1f}%")

    plot_results(results)
    print("Simulation complete. Check generated plots.")
