# Float Math

The Float Math node performs common floating-point operations on two inputs. Use it when you need precise decimal arithmetic, roots, exponentiation, or min/max comparisons with float inputs.

## Inputs

- `Operation`: The operation to perform. Choices:
  - Add A+B: Adds `A` and `B`.
  - Subtract A-B: Subtracts `B` from `A`.
  - Multiply A\*B: Multiplies `A` by `B`.
  - Divide A/B: Floating-point division.
  - Exponentiate A^B: Raises `A` to the power `B` (watch out for zero-to-negative exponents).
  - Absolute Value of A: Absolute value of `A` (ignores `B`).
  - Square Root of A: Square root of `A` (ignores `B`; result invalid for negative `A`).
  - Minimum(A,B): The smaller of `A` and `B`.
  - Maximum(A,B): The larger of `A` and `B`.
- `A`: First float input.
- `B`: Second float input.

## Outputs

- Result: Float — the result of the selected operation.

## Example Usage

![Example](./images/IMAGE_PLACEHOLDER.png)  
Use Float Math to compute a square root or fractional power for precise scaling.

## Notes:

- Division by zero is invalid and will produce a validation error; ensure `B` is not zero for DIV.
- Exponentiation will error if raising zero to a negative exponent. Root operations that produce complex numbers are rejected.
- Square Root operates on `A` only; negative `A` will be rejected because it would produce a complex result.
- For integer-only operations, use the Integer Math node.
