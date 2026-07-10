# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Color palette for asset optimization visualizations."""

CHP = "rgba(100, 149, 237, 1)"
PV = "rgba(255, 243, 138, 1)"
WIND = "rgba(0, 176, 185, 1)"
CONSUMPTION = "rgba(0, 0, 0, 1)"
GRID = "rgba(128, 128, 128, 1)"
CHARGE = "rgba(236, 0, 140, 0.45)"
DISCHARGE = "rgba(146, 219, 68, 0.45)"
BUY = "rgba(139, 0, 0, 1)"
SELL = "rgba(0, 100, 0, 1)"
SOC = "rgba(0, 204, 150, 1)"
AVAILABLE = "rgba(0, 0, 0, 1)"
ZERO_LINE = "rgba(128, 128, 128, 1)"
TRANSPARENT = "rgba(0,0,0,0)"

# Palette for the cumulative PSC stack plot. Alpha is baked in: the bands overlap
# and cross zero, so they must read through each other. Charge/discharge use the
# green/red convention of the original plot, which inverts the hues above.
STACK_CHP = "rgba(100, 149, 237, 0.35)"
STACK_PV = "rgba(255, 215, 0, 0.40)"
STACK_WIND = "rgba(0, 176, 185, 0.40)"
STACK_CHARGE = "rgba(0, 128, 0, 0.30)"
STACK_DISCHARGE = "rgba(220, 20, 60, 0.40)"
STACK_GRID = "rgba(0, 0, 0, 1)"
STACK_CONSUMPTION = "rgba(128, 128, 128, 1)"
STACK_LINE_WIDTH = 2
