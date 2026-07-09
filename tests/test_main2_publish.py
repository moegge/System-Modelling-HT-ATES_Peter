import numpy as np
import pandas as pd


class MockSupplier:
    def __init__(self, name, t_out):
        self.name = name
        self.T_out = t_out


def test_flow_to_storage_example():
    # Generic timestep data, for example 5 hours
    timesteps = 5

    # DataFrame that stores flow-related values
    df_flow = pd.DataFrame(index=range(timesteps))

    # DataFrame that stores result-related values
    result = pd.DataFrame(index=range(timesteps))

    # Two generic suppliers connected to storage
    supplier_1 = MockSupplier(name="Source A", t_out=80)
    supplier_2 = MockSupplier(name="Source B", t_out=70)

    storage_suppliers = [supplier_1, supplier_2]

    # Generic supplier volume output per timestep
    df_flow["Source A Volume out"] = np.array([10, 10, 10, 10, 10])
    df_flow["Source B Volume out"] = np.array([5, 5, 5, 5, 5])

    # Fraction of storage-connected production that is used directly by demand
    #
    # Example:
    # 0.0 means none goes to demand, so all can go to storage
    # 0.5 means half goes to demand, half goes to storage
    # 1.0 means all goes to demand, none goes to storage
    percentage_used = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    # Storage injection switch per timestep
    #
    # 1 means injection allowed
    # 0 means injection blocked
    storage_injection = np.array([1, 1, 1, 0, 1])

    # Heat-pump correction factor.
    # Use 1.0 if there is no correction.
    Factor_due_HP = 1.0

    # Initialize total flow to storage
    df_flow["Total flow to storage"] = 0.0

    # Initialize weighted temperature accumulator
    T_total = 0.0

    for i in storage_suppliers:
        # Same logic as in your original code
        result[i.name + " percentage to storage"] = (1 - percentage_used) * storage_injection

        df_flow[i.name + " flow to storage"] = (
            (1 - percentage_used)
            * df_flow[i.name + " Volume out"]
            * storage_injection
            / Factor_due_HP
        )

        T_total = T_total + sum(df_flow[i.name + " flow to storage"]) * i.T_out

        df_flow["Total flow to storage"] = (
            df_flow["Total flow to storage"]
            + df_flow[i.name + " flow to storage"]
        )

    # Calculate average temperature of all injected water
    total_injected_volume = df_flow["Total flow to storage"].sum()

    if total_injected_volume > 0:
        T_average_to_storage = T_total / total_injected_volume
    else:
        T_average_to_storage = 0

    print("Result:")
    print(result)

    print("\nFlow table:")
    print(df_flow)

    print("\nTotal injected volume:")
    print(total_injected_volume)

    print("\nAverage injection temperature:")
    print(T_average_to_storage)

    # Simple checks
    assert df_flow["Source A flow to storage"].tolist() == [10, 7.5, 5, 0, 0]
    assert df_flow["Source B flow to storage"].tolist() == [5, 3.75, 2.5, 0, 0]
    assert df_flow["Total flow to storage"].tolist() == [15, 11.25, 7.5, 0, 0]

    assert total_injected_volume == 33.75

    # Weighted average:
    # Source A injects 22.5 volume at 80°C
    # Source B injects 11.25 volume at 70°C
    # Average = (22.5 * 80 + 11.25 * 70) / 33.75 = 76.666...
    assert round(T_average_to_storage, 3) == 76.667


if __name__ == "__main__":
    test_flow_to_storage_example()