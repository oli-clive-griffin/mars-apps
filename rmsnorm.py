import numpy as np
from utils import Plot, SubPlot, show_plot


def create_rmsnorm_stages_data(
    initial_data_ND: np.ndarray, gamma_D: np.ndarray
) -> Plot:
    """Prepare RMSNorm transformation stages"""
    rms = np.sqrt(np.mean(initial_data_ND**2, axis=1, keepdims=True))
    # THIS LINE:
    # post_projection_ND = initial_data_ND * (np.sqrt(initial_data_ND.shape[1]) / rms)
    post_projection_ND = initial_data_ND / rms
    post_scaling_ND = post_projection_ND * gamma_D

    return Plot(
        title="RMSNorm",
        plotgrid=[
            [
                SubPlot(
                    data_ND=initial_data_ND,
                    title="Original Data",
                    show_reference_sphere=False,
                ),
                SubPlot(
                    data_ND=post_projection_ND,
                    title="After Projection",
                    show_reference_sphere=True,
                ),
                SubPlot(
                    data_ND=post_scaling_ND,
                    title="After Scaling",
                    show_reference_sphere=True,
                ),
            ]
        ],
    )


def main():
    # Define parameters
    n_points = 200
    initial_data_ND = np.random.randn(n_points, 3)
    gamma_D = np.array([0.8, 1.4, 0.7])

    # Prepare and visualize data
    plot = create_rmsnorm_stages_data(initial_data_ND, gamma_D)

    show_plot(plot)


if __name__ == "__main__":
    np.random.seed(42)
    main()
