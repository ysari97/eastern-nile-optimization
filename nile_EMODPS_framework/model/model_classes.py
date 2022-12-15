import numpy as np
import os
from scipy.constants import g
from array import array
from bisect import bisect_right
from utils import modified_interp

dir_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
data_directory = os.path.join(dir_path, "../data/")


class Catchment:
    # Parts of the model topology where water is naturally accumulated and
    # joins to the main flow. Only attribute other than name is the streamflow
    # vector which stores a value for each time-step
    def __init__(self, name):
        self.name = name
        # self.streamflow = np.loadtxt(f"{data_directory}Inflow{name}.txt")


class HydropowerPlant:
    """
    A class used to represent hydropower plants that may or may not be
    installed in a reservoir.

    Attributes
    ----------
    reservoir : Reservoir
        The reservoir object on which the hydropower plant operates
    identifier : str
        In case there are multiple plants in a reservoir (or set of turbines
        are separated from the others), a name given to identify the plant
    release_share : float
        In case there are multiple plants in a reservoir, the share of flow
        that the hydropower plant utilizes to generate hydroenergy
    efficiency : float
        Efficiency coefficient (mu) used in hydropower formula
    max_turbine_flow : float
        Maximum possible flow that can be passed through the turbines for the
        purpose of hydroenergy production
    head_start_level : float
        Minimum elevation in the reservoir that is used to calculate hydraulic
        head for hydropower production
    max_capacity : float
        mW
        Total design capacity of the plant
    """

    def __init__(self, reservoir, identifier=None, release_share=None):

        self.reservoir = reservoir
        # In case of multiple plants (or turbines) on a single reservoir:
        self.identifier = identifier
        self.release_share = release_share
        # Read the other parameters from file
        self.efficiency = float()
        self.max_turbine_flow = float()
        self.head_start_level = float()
        self.max_capacity = float()

    def calculate_hydroenergy_production(
        self, actual_release, reservoir_level, total_hours
    ):

        if self.release_share is not None:
            actual_release *= self.release_share

        m3_to_kg_factor = 1000
        w_mw_conversion = 1e-6
        turbine_flow = min(actual_release, self.max_turbine_flow)
        head = max(0, reservoir_level - self.head_start_level)
        power_in_mw = min(
            self.max_capacity,
            turbine_flow
            * head
            * m3_to_kg_factor
            * g
            * self.efficiency
            * w_mw_conversion,
        )

        hydroenergy_production = power_in_mw * total_hours  # MWh

        return hydroenergy_production


class IrrigationDistrict:
    """
    A class used to represent districts that demand irrigation

    Attributes
    ----------
    name : str
        Lowercase non-spaced name of the district
    demand : np.array
        m3/s
        Vector of monthly water demand from the district throughout
        the simulation horizon (converted to m3/s)
    incoming_flow : np.array
        m3/s
        Flow that reaches the irrigation district from the previous
        nodes of the model network
    received_flow : np.array
        m3/s
        Amount of water that the district diverted. It is formulated
        as the minimum of demand and the received flow. So, upstream
        has the priority in the current formulation (keep in mind for
        extending the model!)
    deficit : np.array
        m3/s
        Unmet demand in each time step. Lower bound is zero (no minus
        deficit)
    """

    def __init__(self, name):
        # Explanation placeholder
        self.name = name
        fh = os.path.join(data_directory, f"irr_demand_{name}.txt")
        self.demand = np.loadtxt(fh)
        self.received_flow = array("f", [])
        self.incoming_flow = array("f", [])
        self.deficit = array("f", [])


class Reservoir:
    """
    A class used to represent reservoirs of the problem

    Attributes
    ----------
    name : str
        Lowercase non-spaced name of the reservoir
    storage_vector : np.array (1xH)
        m3
        A vector that holds the volume of the water in the reservoir
        throughout the simulation horizon
    level_vector : np.array (1xH)
        m
        A vector that holds the elevation of the water in the reservoir
        throughout the simulation horizon
    release_vector : np.array (1xH)
        m3/s
        A vector that holds the actual average release per month
        from the reservoir throughout the simulation horizon
    hydropower_plant : HydropowerPlant
        A hydropower plant object belonging to the reservoir
    hydroenergy_produced : np.array (1xH)
        MWh
        Amount of hydroenergy produced in each month
    evap_rates : np.array (1x12)
        cm
        Monthly evaporation rates of the reservoir
    level_to_storage_rel : np.array (2x...)
        m -> m3
        Vectors of water level versus corresponding water storage
    level_to_surface_rel : np.array (2x...)
        m -> m2
        Vectors of water level versus corresponding surface area

    Methods
    -------
    storage_to_level(h=float)
        Returns the level(height) based on volume
    level_to_storage(s=float)
        Returns the volume based on level(height)
    level_to_surface(h=float)
        Returns the surface area based on level
    integration()
        FILL IN LATER!!!!
    """

    def __init__(self, name):
        self.name = name

        fh = os.path.join(data_directory, f"evap_{name}.txt")
        self.evap_rates = np.loadtxt(fh)

        fh = os.path.join(data_directory, f"store_min_max_release_{name}.txt")
        self.storage_to_minmax_rel = np.loadtxt(fh)

        fh = os.path.join(data_directory, f"store_level_rel_{name}.txt")
        self.storage_to_level_rel = np.loadtxt(fh)

        fh = os.path.join(data_directory, f"store_sur_rel_{name}.txt")
        self.storage_to_surface_rel = np.loadtxt(fh)

        self.storage_vector = array("f", [])
        self.level_vector = array("f", [])
        self.inflow_vector = array("f", [])
        self.release_vector = array("f", [])
        self.hydropower_plant = None
        self.hydroenergy_produced = array("f", [])
        # self.total_evap = np.empty(0)

    def storage_to_level(self, s):
        return modified_interp(s, self.storage_to_level_rel[0], self.storage_to_level_rel[1])

    def storage_to_surface(self, s):
        return modified_interp(
            s, self.storage_to_surface_rel[0], self.storage_to_surface_rel[1]
        )

    def storage_to_minmax(self, s):
        # For minimum release constraint, we regard the data points as a step function
        # such that once a given storage/elevation is surpassed, we have to release a
        # certain given amount. For maximum, we use interpolation as detailed discharge
        # capacity calculations are made for certain points

        minimum_index = max(bisect_right(self.storage_to_minmax_rel[0], s), 1)
        minimum_cons = self.storage_to_minmax_rel[1][minimum_index - 1]
        maximum_cons = modified_interp(s, self.storage_to_minmax_rel[0], self.storage_to_minmax_rel[2])

        return minimum_cons, maximum_cons

    def integration(
        self,
        total_seconds,
        policy_release_decision,
        net_secondly_inflow,
        current_month,
        integ_step,
    ):
        """Converts the flows of the reservoir into storage. Time step
        fidelity can be adjusted within a for loop. The core idea is to
        arrive at m3 storage from m3/s flows.

        Parameters
        ----------

        Returns
        -------
        """

        self.inflow_vector = np.append(self.inflow_vector, net_secondly_inflow)
        current_storage = self.storage_vector[-1]
        in_month_releases = array("f", [])
        monthly_evap_total = 0
        integ_step_count = total_seconds / integ_step

        for _ in np.arange(0, total_seconds, integ_step):

            surface = self.storage_to_surface(current_storage)

            evaporation = surface * (
                self.evap_rates[current_month - 1] / (100 * integ_step_count)
            )
            monthly_evap_total += evaporation

            min_possible_release, max_possible_release = self.storage_to_minmax(
                current_storage
            )

            secondly_release = min(
                max_possible_release, max(min_possible_release, policy_release_decision)
            )
            # if secondly_release == min_possible_release:
            #     self.constraint_check.append(("Hit LB", secondly_release, level))
            # elif secondly_release == max_possible_release:
            #     self.constraint_check.append(("Hit UB", secondly_release, level))
            # else:
            #     self.constraint_check.append("Smooth release")
            in_month_releases.append(secondly_release)

            total_addition = net_secondly_inflow * integ_step

            current_storage += (
                total_addition - evaporation - secondly_release * integ_step
            )

        self.storage_vector.append(current_storage)

        avg_monthly_release = np.mean(in_month_releases)
        self.release_vector.append(avg_monthly_release)

        # self.total_evap = np.append(self.total_evap, monthly_evap_total)

        # Record level  based on storage for time t:
        self.level_vector.append(self.storage_to_level(current_storage))

        return avg_monthly_release
