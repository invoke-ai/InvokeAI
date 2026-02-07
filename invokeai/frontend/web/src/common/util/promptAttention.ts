import { logger } from 'app/logging/logger';
import { serializeError } from 'serialize-error';

import { type ASTNode, type Attention, parseTokens, serialize, tokenize } from './promptAST';

const log = logger('events');

type AttentionDirection = 'increment' | 'decrement';
type AdjustmentResult = { prompt: string; selectionStart: number; selectionEnd: number };

const ATTENTION_STEP = 1.1;
const NUMERIC_ATTENTION_STEP = 0.1;

/**
 * Check if a weight is approximately ATTENTION_STEP^n for some integer n.
 * Returns n if so, or null if the weight is not a power of ATTENTION_STEP.
 */
function getAttentionStepCount(weight: number): number | null {
  if (weight <= 0) {
    return null;
  }
  if (Math.abs(weight - 1.0) < 0.001) {
    return 0;
  }
  const n = Math.round(Math.log(weight) / Math.log(ATTENTION_STEP));
  if (n === 0) {
    return null;
  }
  const expected = Math.pow(ATTENTION_STEP, n);
  if (Math.abs(expected - weight) < 0.005) {
    return n;
  }
  return null;
}

/**
 * Adjusts the attention of the prompt at the current cursor/selection position.
 */
export function adjustPromptAttention(
  prompt: string,
  selectionStart: number,
  selectionEnd: number,
  direction: AttentionDirection
): AdjustmentResult {
  try {
    const tokens = tokenize(prompt);
    const ast = parseTokens(tokens);
    const terminals = flattenAST(ast);

    let selectedTerminals = terminals.filter((t) => {
      const isSelected =
        (t.range.start < selectionEnd && t.range.end > selectionStart) ||
        (selectionStart === selectionEnd && t.range.start <= selectionStart && t.range.end >= selectionStart);

      if (!isSelected) {
        return false;
      }

      if (t.parentRange) {
        const parentContainsSelection = t.parentRange.start <= selectionStart && t.parentRange.end >= selectionEnd;
        const selectionCoversParent = selectionStart <= t.parentRange.start && selectionEnd >= t.parentRange.end;

        if (!parentContainsSelection && !selectionCoversParent) {
          // Partial overlap.
          if (t.hasExplicitAttention) {
            return false; // Don't modify explicit weight in partial group
          }
        }
      }
      return true;
    });

    for (const t of selectedTerminals) {
      t.isSelected = true;
    }

    if (selectedTerminals.length === 0) {
      const selectedGroup = findSelectedGroup(ast, selectionStart, selectionEnd);
      if (selectedGroup) {
        selectedTerminals = terminals.filter(
          (t) => t.range.start >= selectedGroup.range.start && t.range.end <= selectedGroup.range.end
        );
        for (const t of selectedTerminals) {
          t.isSelected = true;
        }
      }
    }

    if (selectedTerminals.length === 0) {
      return { prompt, selectionStart, selectionEnd };
    }

    for (const terminal of selectedTerminals) {
      if (terminal.hasNumericAttention) {
        // Additive step for explicit numeric weights (e.g. 1.15 → 1.25 / 1.05)
        if (direction === 'increment') {
          terminal.weight = Number((terminal.weight + NUMERIC_ATTENTION_STEP).toFixed(4));
        } else {
          terminal.weight = Number((terminal.weight - NUMERIC_ATTENTION_STEP).toFixed(4));
        }
      } else {
        // Multiplicative step for +/- syntax weights
        if (direction === 'increment') {
          terminal.weight *= ATTENTION_STEP;
        } else {
          terminal.weight /= ATTENTION_STEP;
        }
      }
    }

    const newAST = groupTerminals(terminals);
    const newPrompt = serialize(newAST);
    const newSelection = calculateSelectionRange(newAST);

    return {
      prompt: newPrompt,
      selectionStart: newSelection.start,
      selectionEnd: newSelection.end,
    };
  } catch (e) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    log.error({ error: serializeError(e) as any }, 'Failed to adjust prompt attention');
    return { prompt, selectionStart, selectionEnd };
  }
}

type Terminal = {
  text: string;
  type: ASTNode['type'];
  weight: number;
  range: { start: number; end: number };
  hasExplicitAttention: boolean;
  hasNumericAttention: boolean;
  parentRange?: { start: number; end: number };
  isSelected: boolean;
};

function flattenAST(
  ast: ASTNode[],
  currentWeight = 1.0,
  parentRange?: { start: number; end: number },
  numericAttention = false
): Terminal[] {
  let terminals: Terminal[] = [];

  for (const node of ast) {
    let nodeWeight = currentWeight;
    let nodeNumericAttention = numericAttention;
    if ('attention' in node && node.attention) {
      nodeWeight *= parseAttention(node.attention);
      nodeNumericAttention = typeof node.attention === 'number';
    }

    if (node.type === 'group') {
      terminals.push(...flattenAST(node.children, nodeWeight, node.range, nodeNumericAttention));
    } else {
      terminals.push({
        text: node.type === 'word' ? node.text : node.value,
        type: node.type,
        weight: nodeWeight,
        range: node.range,
        hasExplicitAttention: 'attention' in node && !!node.attention,
        hasNumericAttention: nodeNumericAttention,
        parentRange: parentRange,
        isSelected: false,
      });
    }
  }
  return terminals;
}

function findSelectedGroup(nodes: ASTNode[], start: number, end: number): ASTNode | null {
  for (const node of nodes) {
    if (node.type === 'group') {
      const foundInChildren = findSelectedGroup(node.children, start, end);
      if (foundInChildren) {
        return foundInChildren;
      }

      if (rangesOverlap(node.range, { start, end })) {
        return node;
      }
    }
  }
  return null;
}

function rangesOverlap(a: { start: number; end: number }, b: { start: number; end: number }) {
  return a.start < b.end && a.end > b.start;
}

function parseAttention(attention: Attention): number {
  if (typeof attention === 'number') {
    return attention;
  }
  if (attention.startsWith('+')) {
    return Math.pow(ATTENTION_STEP, attention.length);
  }
  if (attention.startsWith('-')) {
    return Math.pow(ATTENTION_STEP, -attention.length);
  }
  const num = Number(attention);
  return isNaN(num) ? 1.0 : num;
}

function calculateSelectionRange(nodes: ASTNode[]): { start: number; end: number } {
  let selectionStart = Infinity;
  let selectionEnd = -1;
  let currentPos = 0;

  function traverse(nodes: ASTNode[]) {
    for (const node of nodes) {
      if (node.isSelection) {
        const len = serialize([node]).length;
        selectionStart = Math.min(selectionStart, currentPos);
        selectionEnd = Math.max(selectionEnd, currentPos + len);
        currentPos += len;
      } else {
        if (node.type === 'group') {
          // Group is not fully selected, but children might be.
          // Group structure: "(" + children + ")" + attention
          currentPos += 1; // '('
          traverse(node.children);
          currentPos += 1; // ')'
          if (node.attention) {
            currentPos += String(node.attention).length;
          }
        } else {
          // Leaf node not selected.
          const len = serialize([node]).length;
          currentPos += len;
        }
      }
    }
  }

  traverse(nodes);

  if (selectionStart === Infinity) {
    return { start: 0, end: serialize(nodes).length };
  }
  return { start: selectionStart, end: selectionEnd };
}

function groupTerminals(terminals: Terminal[]): ASTNode[] {
  if (terminals.length === 0) {
    return [];
  }

  const nodes: ASTNode[] = [];
  let i = 0;
  while (i < terminals.length) {
    const t = terminals[i]!;
    const weight = t.weight;

    const findRunEnd = (predicate: (w: number) => boolean) => {
      let j = i;
      while (j < terminals.length) {
        const next = terminals[j]!;
        if (predicate(next.weight)) {
          j++;
        } else if (next.type === 'whitespace') {
          let k = j + 1;
          while (k < terminals.length && terminals[k]!.type === 'whitespace') {
            k++;
          }
          if (k < terminals.length && predicate(terminals[k]!.weight)) {
            j = k;
          } else {
            break;
          }
        } else {
          break;
        }
      }
      return j;
    };

    const stepCount = getAttentionStepCount(weight);

    // Check for + (positive power of ATTENTION_STEP)
    if (stepCount !== null && stepCount > 0) {
      const j = findRunEnd((w) => {
        const sc = getAttentionStepCount(w);
        return sc !== null && sc > 0;
      });

      let runStart = i;
      let runEnd = j;
      while (runStart < runEnd && terminals[runStart]!.type === 'whitespace') {
        runStart++;
      }
      while (runEnd > runStart && terminals[runEnd - 1]!.type === 'whitespace') {
        runEnd--;
      }

      for (let k = i; k < runStart; k++) {
        nodes.push(createNodeFromTerminal(terminals[k]!));
      }

      if (runStart < runEnd) {
        const slice = terminals.slice(runStart, runEnd).map((t) => ({ ...t, weight: t.weight / ATTENTION_STEP }));
        const children = groupTerminals(slice);
        const isSelection = slice.every((t) => t.isSelected);

        if (children.length === 1) {
          const child = children[0]!;
          if (child.type === 'word' || child.type === 'group') {
            const newAttention = addAttention(child.attention, '+');
            nodes.push({ ...child, attention: newAttention });
          } else {
            nodes.push({ type: 'group', children, attention: '+', range: { start: 0, end: 0 }, isSelection });
          }
        } else {
          nodes.push({ type: 'group', children, attention: '+', range: { start: 0, end: 0 }, isSelection });
        }
      }

      for (let k = runEnd; k < j; k++) {
        nodes.push(createNodeFromTerminal(terminals[k]!));
      }

      i = j;
      continue;
    }

    // Check for - (negative power of ATTENTION_STEP)
    if (stepCount !== null && stepCount < 0) {
      const j = findRunEnd((w) => {
        const sc = getAttentionStepCount(w);
        return sc !== null && sc < 0;
      });

      let runStart = i;
      let runEnd = j;
      while (runStart < runEnd && terminals[runStart]!.type === 'whitespace') {
        runStart++;
      }
      while (runEnd > runStart && terminals[runEnd - 1]!.type === 'whitespace') {
        runEnd--;
      }

      for (let k = i; k < runStart; k++) {
        nodes.push(createNodeFromTerminal(terminals[k]!));
      }

      if (runStart < runEnd) {
        const slice = terminals.slice(runStart, runEnd).map((t) => ({ ...t, weight: t.weight * ATTENTION_STEP }));
        const children = groupTerminals(slice);
        const isSelection = slice.every((t) => t.isSelected);

        if (children.length === 1) {
          const child = children[0]!;
          if (child.type === 'word' || child.type === 'group') {
            const newAttention = addAttention(child.attention, '-');
            nodes.push({ ...child, attention: newAttention });
          } else {
            nodes.push({ type: 'group', children, attention: '-', range: { start: 0, end: 0 }, isSelection });
          }
        } else {
          nodes.push({ type: 'group', children, attention: '-', range: { start: 0, end: 0 }, isSelection });
        }
      }

      for (let k = runEnd; k < j; k++) {
        nodes.push(createNodeFromTerminal(terminals[k]!));
      }

      i = j;
      continue;
    }

    // Residual or 1.0
    if (Math.abs(weight - 1.0) < 0.001) {
      nodes.push(createNodeFromTerminal(t));
      i++;
    } else {
      let j = i;
      while (j < terminals.length && Math.abs(terminals[j]!.weight - weight) < 0.001) {
        j++;
      }

      const groupTerminalsSlice = terminals.slice(i, j).map((t) => ({ ...t, weight: 1.0 }));
      const children = groupTerminals(groupTerminalsSlice);
      const isSelection = groupTerminalsSlice.every((t) => t.isSelected);

      const weightStr = Number(weight.toFixed(4));

      // Always create a group for numeric weights to preserve parentheses in output
      nodes.push({ type: 'group', children, attention: weightStr, range: { start: 0, end: 0 }, isSelection });
      i = j;
    }
  }
  return nodes;
}

function createNodeFromTerminal(t: Terminal): ASTNode {
  if (t.type === 'word') {
    return { type: 'word', text: t.text, range: t.range, isSelection: t.isSelected };
  }
  if (t.type === 'whitespace') {
    return { type: 'whitespace', value: t.text, range: t.range, isSelection: t.isSelected };
  }
  if (t.type === 'punct') {
    return { type: 'punct', value: t.text, range: t.range, isSelection: t.isSelected };
  }
  if (t.type === 'embedding') {
    return { type: 'embedding', value: t.text, range: t.range, isSelection: t.isSelected };
  }
  if (t.type === 'escaped_paren') {
    return { type: 'escaped_paren', value: t.text as '(' | ')', range: t.range, isSelection: t.isSelected };
  }
  return { type: 'word', text: t.text, range: t.range, isSelection: t.isSelected };
}

function addAttention(current: Attention | undefined, added: string): Attention | undefined {
  if (!current) {
    return added;
  }
  if (typeof current === 'number') {
    if (added === '+') {
      return Number((current * ATTENTION_STEP).toFixed(4));
    }
    if (added === '-') {
      return Number((current / ATTENTION_STEP).toFixed(4));
    }
    return current;
  }
  if (added === '+') {
    if (current.startsWith('-')) {
      const res = current.substring(1);
      return res === '' ? undefined : res;
    }
    return `${current}+`;
  }
  if (added === '-') {
    if (current.startsWith('+')) {
      const res = current.substring(1);
      return res === '' ? undefined : res;
    }
    return `${current}-`;
  }
  return current;
}
