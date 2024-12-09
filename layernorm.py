import numpy as np
from utils import SubPlot, show_plot, Plot


def create_layernorm_stages_data(
    initial_data_ND: np.ndarray, gamma_D: np.ndarray, beta_D: np.ndarray
) -> Plot:
    """Prepare LayerNorm transformation stages"""
    mean = np.mean(initial_data_ND, axis=1, keepdims=True)
    centered_ND = initial_data_ND - mean
    std = np.sqrt(np.mean(centered_ND**2, axis=1, keepdims=True))
    post_projection_ND = centered_ND / std
    post_scaling_ND = post_projection_ND * gamma_D
    post_shift_ND = post_scaling_ND + beta_D

    return Plot(
        title="LayerNorm",
        plotgrid=[
            [
                SubPlot(
                    data_ND=initial_data_ND,
                    title="Original Data",
                    show_reference_sphere=False,
                ),
                SubPlot(
                    data_ND=centered_ND,
                    title="After Centering",
                    show_reference_sphere=False,
                ),
                SubPlot(
                    data_ND=post_projection_ND,
                    title="After Projection",
                    show_reference_sphere=True,
                ),
            ],
            [
                SubPlot(
                    data_ND=post_scaling_ND,
                    title="After Scaling",
                    show_reference_sphere=True,
                ),
                SubPlot(
                    data_ND=post_shift_ND,
                    title="After Shift",
                    show_reference_sphere=True,
                ),
            ],
        ],
    )


def main():
    # Define parameters
    n_points = 50
    initial_data_ND = np.random.randn(n_points, 3)
    gamma_D = np.array([0.8, 1.4, 0.7])
    beta_D = np.array([0.8, 1.2, 0.3])

    # Prepare and visualize data
    plot = create_layernorm_stages_data(initial_data_ND, gamma_D, beta_D)

    show_plot(plot)


if __name__ == "__main__":
    np.random.seed(42)
    main()
