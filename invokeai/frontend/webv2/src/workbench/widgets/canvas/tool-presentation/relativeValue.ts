export type RelativeOperator = '%' | '*' | '/';

const OPERATOR_ALIASES: Record<string, RelativeOperator> = { '%': '%', '*': '*', '/': '/', '×': '*', '÷': '/' };

export const parseRelativeInput = (text: string): { operator: RelativeOperator; operand: string } | null => {
  const match = /^\s*([*/×÷])?\s*(-?\d*\.?\d*)\s*(%)?\s*$/.exec(text);
  if (!match || (match[1] !== undefined) === (match[3] !== undefined)) {
    return null;
  }
  return { operand: match[2]!, operator: OPERATOR_ALIASES[match[1] ?? match[3]!]! };
};

export const resolveRelativeValue = (current: number, operator: RelativeOperator, operand: number): number | null => {
  if (!Number.isFinite(current) || !Number.isFinite(operand)) {
    return null;
  }
  switch (operator) {
    case '%':
      return (current * operand) / 100;
    case '*':
      return current * operand;
    case '/':
      return operand === 0 ? null : current / operand;
  }
};
