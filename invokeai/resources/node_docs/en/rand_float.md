# Random Float

The Random Float node outputs a single non-deterministic floating-point number sampled uniformly from a range. Use it to introduce small continuous variation or random seeds for float parameters.

## Inputs

- `Low`: Inclusive lower bound (float).
- `High`: Exclusive upper bound (float) — generated value will be >= `Low` and < `High`.
- `Decimals`: Number of decimal places to round the result to (integer).

## Outputs

- Result: Float — a randomly chosen float rounded to the specified number of decimals.

## Example Usage

![Example](./images/IMAGE_PLACEHOLDER.png)  
Generate a randomized float parameter (e.g., color hue offset) with controlled precision.

## Notes:

- This node is non-deterministic (use_cache=False) and will produce a new value each run.
- The node rounds the sampled float to the requested number of decimals before output.
- Ensure low < high to avoid errors.
