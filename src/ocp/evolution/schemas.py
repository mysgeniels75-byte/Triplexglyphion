"""
Schema Evolution System

Implements:
- Schema recombination with novelty injection
- Topology-level mutations (not just parameter tuning)
- Fitness-based selection
- Cambrian explosion-style innovation
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import uuid
import numpy as np
from numpy.typing import NDArray


class MutationOperator(Enum):
    """Types of mutations that can be applied to schemas"""
    PARAMETER_TWEAK = "parameter_tweak"
    STRUCTURE_CHANGE = "structure_change"
    RECOMBINATION = "recombination"
    NOVELTY_INJECTION = "novelty_injection"


@dataclass
class Schema:
    """A decision-making schema that can evolve"""
    schema_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_type: str = "base"
    parameters: Dict[str, Any] = field(default_factory=dict)
    structure: Dict[str, Any] = field(default_factory=dict)
    fitness_history: List[float] = field(default_factory=list)
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)

    def compute_average_fitness(self) -> float:
        """Compute average historical fitness"""
        if not self.fitness_history:
            return 0.0
        return np.mean(self.fitness_history[-100:])  # Recent performance

    def add_fitness_score(self, score: float) -> None:
        """Record a fitness score"""
        self.fitness_history.append(score)


class SchemaEvolution:
    """
    Evolutionary system for decision schemas

    Uses genetic algorithm principles but operates on cognitive architectures
    rather than just parameters
    """

    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        novelty_rate: float = 0.05,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.novelty_rate = novelty_rate

        # Current population
        self.population: List[Schema] = []

        # Schema library (all schemas ever created)
        self.schema_library: Dict[str, Schema] = {}

        # Generation counter
        self.generation = 0

        # Statistics
        self.stats = {
            "total_schemas_created": 0,
            "total_mutations": 0,
            "total_crossovers": 0,
            "best_fitness": 0.0,
        }

    def initialize_population(self, seed_schemas: List[Schema]) -> None:
        """Initialize population with seed schemas"""

        self.population = seed_schemas.copy()

        # Fill remaining slots with mutations
        while len(self.population) < self.population_size:
            parent = np.random.choice(seed_schemas)
            child = self.mutate(parent)
            self.population.append(child)

        # Add to library
        for schema in self.population:
            self.schema_library[schema.schema_id] = schema

        self.stats["total_schemas_created"] = len(self.population)

    def evolve_generation(self) -> None:
        """Evolve population by one generation"""

        # Selection
        selected = self.selection()

        # Generate offspring
        offspring = []

        while len(offspring) < self.population_size:
            # Crossover
            if np.random.random() < self.crossover_rate and len(selected) >= 2:
                parent1, parent2 = np.random.choice(selected, size=2, replace=False)
                child = self.crossover(parent1, parent2)
                self.stats["total_crossovers"] += 1
            else:
                # Mutation only
                parent = np.random.choice(selected)
                child = self.mutate(parent)

            # Novelty injection
            if np.random.random() < self.novelty_rate:
                child = self.inject_novelty(child)

            offspring.append(child)
            self.schema_library[child.schema_id] = child
            self.stats["total_schemas_created"] += 1

        # Replace population
        self.population = offspring
        self.generation += 1

        # Update best fitness
        best_fitness = max(s.compute_average_fitness() for s in self.population)
        self.stats["best_fitness"] = max(self.stats["best_fitness"], best_fitness)

    def selection(self) -> List[Schema]:
        """
        Select schemas for breeding based on fitness

        Uses tournament selection
        """

        selected = []
        tournament_size = 5

        num_selected = self.population_size // 2

        for _ in range(num_selected):
            # Random tournament
            tournament = np.random.choice(self.population, size=tournament_size)

            # Select winner (highest fitness)
            winner = max(tournament, key=lambda s: s.compute_average_fitness())

            selected.append(winner)

        return selected

    def mutate(self, parent: Schema) -> Schema:
        """Mutate a schema"""

        child = Schema(
            schema_type=parent.schema_type,
            parameters=parent.parameters.copy(),
            structure=parent.structure.copy(),
            generation=self.generation + 1,
            parent_ids=[parent.schema_id],
        )

        # Choose mutation type
        mutation_type = np.random.choice(list(MutationOperator))

        if mutation_type == MutationOperator.PARAMETER_TWEAK:
            # Mutate parameters
            for key, value in child.parameters.items():
                if isinstance(value, (int, float)):
                    if np.random.random() < self.mutation_rate:
                        # Add Gaussian noise
                        noise = np.random.randn() * abs(value) * 0.1
                        child.parameters[key] = value + noise

        elif mutation_type == MutationOperator.STRUCTURE_CHANGE:
            # Modify structure
            if "layers" in child.structure:
                # Add or remove a layer
                if np.random.random() < 0.5 and len(child.structure["layers"]) > 1:
                    # Remove random layer
                    idx = np.random.randint(len(child.structure["layers"]))
                    child.structure["layers"].pop(idx)
                else:
                    # Add random layer
                    new_layer = {"type": "processing", "size": np.random.randint(10, 100)}
                    child.structure["layers"].append(new_layer)

        self.stats["total_mutations"] += 1

        return child

    def crossover(self, parent1: Schema, parent2: Schema) -> Schema:
        """Combine two schemas"""

        child = Schema(
            schema_type=parent1.schema_type,
            parameters={},
            structure={},
            generation=self.generation + 1,
            parent_ids=[parent1.schema_id, parent2.schema_id],
        )

        # Crossover parameters
        for key in set(parent1.parameters.keys()) | set(parent2.parameters.keys()):
            if key in parent1.parameters and key in parent2.parameters:
                # Average or randomly choose
                if np.random.random() < 0.5:
                    child.parameters[key] = parent1.parameters[key]
                else:
                    child.parameters[key] = parent2.parameters[key]
            elif key in parent1.parameters:
                child.parameters[key] = parent1.parameters[key]
            else:
                child.parameters[key] = parent2.parameters[key]

        # Crossover structure (simplified)
        child.structure = parent1.structure.copy()

        return child

    def inject_novelty(self, schema: Schema) -> Schema:
        """Inject completely novel elements"""

        # Add random new parameter
        novel_param = f"novel_{np.random.randint(10000)}"
        schema.parameters[novel_param] = np.random.randn()

        return schema

    def get_best_schemas(self, n: int = 10) -> List[Schema]:
        """Get top n schemas by fitness"""

        return sorted(
            self.population,
            key=lambda s: s.compute_average_fitness(),
            reverse=True,
        )[:n]

    def get_statistics(self) -> Dict[str, Any]:
        """Get evolution statistics"""
        avg_fitness = np.mean([s.compute_average_fitness() for s in self.population])

        return {
            **self.stats,
            "generation": self.generation,
            "population_size": len(self.population),
            "library_size": len(self.schema_library),
            "avg_fitness": avg_fitness,
        }
