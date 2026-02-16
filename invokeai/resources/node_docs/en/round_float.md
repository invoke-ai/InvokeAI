# Round Float

The Round Float node reduces a floating-point number to a specified number of decimal places. Use it when you want to control numeric precision for display, comparison, or downstream calculations.

## Inputs

- `Value`: The float value to round.
- `Decimals`: Number of decimal places to retain (integer). Use 0 for whole-number results.

## Outputs

- Result: Float — the rounded value.

## Example Usage

![Example](./images/IMAGE_PLACEHOLDER.png)  
Round a noisy parameter to two decimal places for stable downstream behavior.

## Notes:

- Rounding uses Python's round behavior (ties round to the nearest even value).
- Negative values are rounded according to the same rule (e.g., rounding -1.5 to 0 decimals yields -2.0 under nearest-even tie resolution).
- If you need integer results, use the Float to Integer node which supports rounding to multiples and different rounding methods.
