"""
Main Monte Carlo simulation engine.

This module provides the core simulation orchestration that combines
market models, bond models, shockwave triggers, and panic dynamics
into a comprehensive financial simulation.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from monte_carlo_finance.core.config import SimulationConfig
from monte_carlo_finance.models.bond import BondModel
from monte_carlo_finance.models.market import MarketModel
from monte_carlo_finance.triggers.panic import FearGreedIndex, PanicModel
from monte_carlo_finance.triggers.shockwave import ShockwaveEvent, ShockwaveTrigger
from monte_carlo_finance.utils.random import RandomGenerator
from monte_carlo_finance.utils.statistics import calculate_statistics


@dataclass
class SimulationResults:
    """Container for simulation results.

    Attributes:
        market_paths: Array of market value paths (num_sims, num_steps+1)
        bond_prices: Array of bond price paths (num_sims, num_steps+1)
        bond_yields: Array of bond yield paths (num_sims, num_steps+1)
        panic_levels: Array of panic level paths (num_sims, num_steps+1)
        fear_greed_index: Array of fear/greed index paths (num_sims, num_steps+1)
        shockwave_events: List of all shockwave events per simulation
        selloff_events: List of all selloff events per simulation
        defaults: Array indicating which simulations had defaults
        statistics: Dictionary of calculated statistics
    """
    market_paths: np.ndarray
    bond_prices: np.ndarray
    bond_yields: np.ndarray
    panic_levels: np.ndarray
    fear_greed_index: np.ndarray
    shockwave_events: List[List[ShockwaveEvent]]
    selloff_events: List[List[dict]]
    defaults: np.ndarray
    statistics: Dict[str, dict] = field(default_factory=dict)

    def calculate_statistics(self) -> None:
        """Calculate comprehensive statistics for all result arrays."""
        # Final values statistics
        self.statistics["market_final"] = calculate_statistics(self.market_paths[:, -1])
        self.statistics["bond_price_final"] = calculate_statistics(self.bond_prices[:, -1])
        self.statistics["bond_yield_final"] = calculate_statistics(self.bond_yields[:, -1])
        self.statistics["panic_max"] = calculate_statistics(
            np.max(self.panic_levels, axis=1)
        )

        # Returns statistics
        market_returns = np.diff(np.log(self.market_paths), axis=1)
        self.statistics["market_returns"] = calculate_statistics(market_returns)

        # Default statistics
        self.statistics["default_rate"] = {
            "rate": float(np.mean(self.defaults)),
            "count": int(np.sum(self.defaults)),
            "total": len(self.defaults),
        }

        # Shockwave statistics
        total_events = sum(len(events) for events in self.shockwave_events)
        self.statistics["shockwave"] = {
            "total_events": total_events,
            "avg_events_per_sim": total_events / len(self.shockwave_events),
        }

    def get_percentile_paths(
        self,
        percentiles: List[float] = None
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """Get percentile paths for visualization.

        Args:
            percentiles: List of percentiles to calculate

        Returns:
            Dictionary with percentile paths for each metric
        """
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]
        result = {}

        for name, data in [
            ("market", self.market_paths),
            ("bond_price", self.bond_prices),
            ("bond_yield", self.bond_yields),
            ("panic", self.panic_levels),
        ]:
            result[name] = {}
            for p in percentiles:
                result[name][p] = np.percentile(data, p, axis=0)

        return result


class MonteCarloSimulation:
    """Main Monte Carlo simulation engine.

    This class orchestrates the entire simulation process, combining:
    - Market value dynamics (GBM with jumps)
    - Bond pricing and yield movements (Vasicek model)
    - Shockwave events and cascade triggers
    - Market panic dynamics

    All components are highly configurable through the SimulationConfig.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """Initialize the simulation engine.

        Args:
            config: Simulation configuration (uses defaults if not provided)
        """
        self.config = config or SimulationConfig()

        # Random number generator
        self.rng = RandomGenerator(self.config.random_seed)

        # Component models (will be initialized per simulation)
        self._market_model: Optional[MarketModel] = None
        self._bond_model: Optional[BondModel] = None
        self._shockwave_trigger: Optional[ShockwaveTrigger] = None
        self._panic_model: Optional[PanicModel] = None
        self._fear_greed: Optional[FearGreedIndex] = None

        # Custom hooks
        self._pre_step_hooks: List[Callable] = []
        self._post_step_hooks: List[Callable] = []
        self._event_hooks: List[Callable] = []

    def _initialize_components(self) -> None:
        """Initialize all simulation components."""
        self._market_model = MarketModel(self.config.market, self.rng)
        self._bond_model = BondModel(self.config.bond, self.rng)
        self._shockwave_trigger = ShockwaveTrigger(self.config.shockwave, self.rng)
        self._panic_model = PanicModel(self.config.panic, self.rng)
        self._fear_greed = FearGreedIndex()

    def _reset_components(self) -> None:
        """Reset all components for a new simulation path."""
        if self._market_model:
            self._market_model.reset()
        if self._bond_model:
            self._bond_model.reset()
        if self._shockwave_trigger:
            self._shockwave_trigger.reset()
        if self._panic_model:
            self._panic_model.reset()
        if self._fear_greed:
            self._fear_greed.reset()

    def register_pre_step_hook(self, hook: Callable) -> None:
        """Register a function to be called before each simulation step.

        The hook receives: (step, market_model, bond_model, panic_model)

        Args:
            hook: Callback function
        """
        self._pre_step_hooks.append(hook)

    def register_post_step_hook(self, hook: Callable) -> None:
        """Register a function to be called after each simulation step.

        The hook receives: (step, market_model, bond_model, panic_model, events)

        Args:
            hook: Callback function
        """
        self._post_step_hooks.append(hook)

    def register_event_hook(self, hook: Callable) -> None:
        """Register a function to be called when shockwave events occur.

        The hook receives: (event, step)

        Args:
            hook: Callback function
        """
        self._event_hooks.append(hook)

    def _run_single_simulation(self) -> dict:
        """Run a single simulation path.

        Returns:
            Dictionary with simulation results for this path
        """
        self._reset_components()

        dt = self.config.dt
        num_steps = self.config.num_steps

        # Storage for this path
        market_values = [self._market_model.current_value]
        bond_prices = [self._bond_model.current_price]
        bond_yields = [self._bond_model.current_yield]
        panic_levels = [self._panic_model.panic_level]
        fear_greed_values = [self._fear_greed.value]

        prev_market_value = self._market_model.current_value
        prev_yield = self._bond_model.current_yield

        events = []
        selloffs = []
        defaulted = False

        for step in range(num_steps):
            # Pre-step hooks
            for hook in self._pre_step_hooks:
                hook(step, self._market_model, self._bond_model, self._panic_model)

            # Calculate returns from previous step
            market_return = 0.0
            if prev_market_value > 0:
                market_return = (self._market_model.current_value - prev_market_value) / prev_market_value
            yield_change = self._bond_model.current_yield - prev_yield

            # Get shockwave impacts
            yield_impact, market_impact = self._shockwave_trigger.step()

            # Check for new shockwave triggers
            new_events = self._shockwave_trigger.check_triggers(
                market_value=self._market_model.current_value,
                market_return=market_return,
                bond_yield=self._bond_model.current_yield,
                yield_change=yield_change,
                panic_level=self._panic_model.panic_level
            )

            # Record events
            for event in new_events:
                events.append(event)
                for hook in self._event_hooks:
                    hook(event, step)

            # Record selloffs
            for selloff in self._shockwave_trigger.selloff_orders:
                if selloff.triggered_at == step:
                    selloffs.append({
                        "step": step,
                        "intensity": selloff.intensity,
                        "reason": selloff.reason,
                    })

            # Update panic model
            panic_state = self._panic_model.update(
                market_return=market_return,
                yield_change=yield_change,
                shockwave_intensity=market_impact
            )

            # Apply panic modifiers to market and bond models
            self._market_model.set_panic_modifiers(
                volatility_modifier=panic_state.volatility_multiplier,
                drift_modifier=panic_state.drift_modifier
            )
            self._bond_model.set_panic_modifiers(
                yield_volatility_modifier=panic_state.volatility_multiplier,
                spread_modifier=panic_state.correlation_boost * 0.01  # Convert to spread impact
            )

            # Store previous values for next iteration
            prev_market_value = self._market_model.current_value
            prev_yield = self._bond_model.current_yield

            # Step market model
            self._market_model.step(dt, shock_impact=market_impact)

            # Step bond model
            self._bond_model.step(dt, yield_shock=yield_impact)

            # Update fear/greed index
            vol_ratio = panic_state.volatility_multiplier
            self._fear_greed.update(market_return, vol_ratio)

            # Check for default
            if self._bond_model.is_defaulted:
                defaulted = True

            # Record values
            market_values.append(self._market_model.current_value)
            bond_prices.append(self._bond_model.current_price)
            bond_yields.append(self._bond_model.current_yield)
            panic_levels.append(self._panic_model.panic_level)
            fear_greed_values.append(self._fear_greed.value)

            # Post-step hooks
            for hook in self._post_step_hooks:
                hook(step, self._market_model, self._bond_model, self._panic_model, new_events)

        return {
            "market_values": np.array(market_values),
            "bond_prices": np.array(bond_prices),
            "bond_yields": np.array(bond_yields),
            "panic_levels": np.array(panic_levels),
            "fear_greed": np.array(fear_greed_values),
            "events": events,
            "selloffs": selloffs,
            "defaulted": defaulted,
        }

    def run(self, progress_callback: Optional[Callable[[int, int], None]] = None) -> SimulationResults:
        """Run the full Monte Carlo simulation.

        Args:
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            SimulationResults containing all simulation outputs
        """
        self._initialize_components()

        num_sims = self.config.num_simulations
        num_steps = self.config.num_steps

        # Pre-allocate result arrays
        market_paths = np.zeros((num_sims, num_steps + 1))
        bond_prices = np.zeros((num_sims, num_steps + 1))
        bond_yields = np.zeros((num_sims, num_steps + 1))
        panic_levels = np.zeros((num_sims, num_steps + 1))
        fear_greed_values = np.zeros((num_sims, num_steps + 1))
        defaults = np.zeros(num_sims, dtype=bool)

        all_events: List[List[ShockwaveEvent]] = []
        all_selloffs: List[List[dict]] = []

        # Run simulations
        for i in range(num_sims):
            result = self._run_single_simulation()

            market_paths[i] = result["market_values"]
            bond_prices[i] = result["bond_prices"]
            bond_yields[i] = result["bond_yields"]
            panic_levels[i] = result["panic_levels"]
            fear_greed_values[i] = result["fear_greed"]
            defaults[i] = result["defaulted"]
            all_events.append(result["events"])
            all_selloffs.append(result["selloffs"])

            if progress_callback:
                progress_callback(i + 1, num_sims)

        # Create results object
        results = SimulationResults(
            market_paths=market_paths,
            bond_prices=bond_prices,
            bond_yields=bond_yields,
            panic_levels=panic_levels,
            fear_greed_index=fear_greed_values,
            shockwave_events=all_events,
            selloff_events=all_selloffs,
            defaults=defaults,
        )

        # Calculate statistics
        results.calculate_statistics()

        return results

    def run_scenario(
        self,
        scenario_events: List[tuple],
        num_simulations: Optional[int] = None
    ) -> SimulationResults:
        """Run simulation with pre-defined scenario events.

        Args:
            scenario_events: List of (step, ShockwaveEvent) tuples
            num_simulations: Override number of simulations

        Returns:
            SimulationResults
        """
        if num_simulations:
            original_sims = self.config.num_simulations
            self.config.num_simulations = num_simulations

        self._initialize_components()

        # Create event injection hook
        event_schedule = dict(scenario_events)

        def inject_events(step, market_model, bond_model, panic_model):
            if step in event_schedule:
                self._shockwave_trigger.inject_event(event_schedule[step])

        self.register_pre_step_hook(inject_events)

        results = self.run()

        # Clean up
        self._pre_step_hooks.remove(inject_events)
        if num_simulations:
            self.config.num_simulations = original_sims

        return results

    def stress_test(
        self,
        stress_scenarios: List[dict]
    ) -> List[SimulationResults]:
        """Run multiple stress test scenarios.

        Args:
            stress_scenarios: List of scenario configurations, each containing:
                - name: Scenario name
                - events: List of (step, ShockwaveEvent) tuples
                - config_overrides: Optional config parameter overrides

        Returns:
            List of SimulationResults, one per scenario
        """
        results = []

        for scenario in stress_scenarios:
            # Apply config overrides if provided
            if "config_overrides" in scenario:
                # Store original values
                original_values = {}
                for key, value in scenario["config_overrides"].items():
                    parts = key.split(".")
                    obj = self.config
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    original_values[key] = getattr(obj, parts[-1])
                    setattr(obj, parts[-1], value)

            # Run scenario
            scenario_results = self.run_scenario(
                scenario.get("events", []),
                scenario.get("num_simulations")
            )

            # Store scenario name
            scenario_results.statistics["scenario_name"] = scenario.get("name", "Unnamed")
            results.append(scenario_results)

            # Restore original values
            if "config_overrides" in scenario:
                for key, value in original_values.items():
                    parts = key.split(".")
                    obj = self.config
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, parts[-1], value)

        return results
