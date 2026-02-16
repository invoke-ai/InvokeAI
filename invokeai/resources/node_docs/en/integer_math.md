# Integer Math

The Integer Math node performs a range of integer operations (add, subtract, multiply, divide, modulus, power, absolute, min, max) on two integer inputs. Use it when you need a single node to perform common integer arithmetic with built-in validation for operations that would produce invalid integer results.

## Inputs

- operation: The operation to perform. Choices:
  - Add A+B: Adds `A` and `B`.
  - Subtract A-B: Subtracts `B` from `A`.
  - Multiply A\*B: Multiplies `A` by `B`.
  - Divide A/B: Integer division; fractional part discarded.
  - Exponentiate A^B: Raises `A` to the power `B` (b must be >= 0).
  - Modulus A%B: Remainder of `A` divided by `B`.
  - Absolute Value of A: Absolute value of `A` (ignores `B`).
  - Minimum(A,B): The smaller of `A` and `B`.
  - Maximum(A,B): The larger of `A` and `B`.
- `A`: First integer (primary operand).
- `B`: Second integer (secondary operand; some operations may ignore it).

## Outputs

- Result: Integer — the result of the selected operation.

## Example Usage

![Example](./images/IMAGE_PLACEHOLDER.png)  
Use the Integer Math node to compute integer exponents or combine two counters with a chosen operation.

## Notes:

- Division and modulus by zero are invalid and will produce a validation error; ensure `B` is not zero for DIV or MOD.
- Exponentiation (EXP) requires a non-negative exponent (`B` >= 0); negative exponents are rejected because they don't produce integers.
- Division uses integer division (equivalent to int(`A` / `B`)), so fractional portions are discarded rather than rounded.
- Absolute Value ignores the `B` input.
