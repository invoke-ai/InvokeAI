# Random Integer

The Random Integer node outputs a single non-deterministic integer drawn from a range. Use it when you need jittered counts, random indices, or other unpredictable integer values.

## Inputs

- `Low`: Inclusive lower bound (integer).
- `High`: Exclusive upper bound (integer) — the generated value will be >= `Low` and < `High`.

## Outputs

- Result: Integer — a randomly chosen integer in [`Low`, `High`).

## Example Usage

![Example](./images/IMAGE_PLACEHOLDER.png)  
Generate a random index or offset to vary results across runs.

## Notes:

- This node is non-deterministic (use_cache=False) and will produce a new value each run.
- high is exclusive; set high = low + 1 to get either low only.
- Ensure low < high to avoid errors.
