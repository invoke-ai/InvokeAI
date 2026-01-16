# Float to Integer
The Float to Integer node rounds floating-point numbers to integers. At "Multiple of" 1, it performs standard rounding to the nearest integer. By adjusting the "Multiple of" parameter, users can round to the nearest specified multiple. Multiples of 64 are helpfulfor image dimensions that are more compatible with denoising models, 2 will return the nearest even number, etc. The "Method" parameter allows you to choose the rounding direction.

## Inputs
- Value: The floating-point number to be converted to an integer.
- Multiple of: The multiple to which the value should be rounded. Leave at 1 for rounding to the nearest integer.
- Method: The direction to apply rounding:
    - Nearest: Rounds to the nearest multiple.
    - Floor: Rounds down to the nearest multiple.
    - Ceil: Rounds up to the nearest multiple.
    - Truncate: Rounds towards zero to the nearest multiple.

## Notes:
- This node uses numpy floor/ceiling operations, so direction is consistent for both positive and negative values. For example; flooring 3.7 results in 3, but flooring -3.7 results in -4, and not -3. To round towards the next lesser magnitude (i.e., -3), use the Truncate method.
