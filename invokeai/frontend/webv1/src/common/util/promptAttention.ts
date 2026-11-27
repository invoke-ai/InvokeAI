import { logger } from 'app/logging/logger';
import { serializeError } from 'serialize-error';

import {
  type ASTNode,
  type Attention,
  parseTokens,
  type PromptFunctionArg,
  serializeWithSelection,
  tokenize,
} from './promptAST';

const log = logger('generation');

type AttentionDirection = 'increment' | 'decrement';
type AdjustmentResult = { prompt: string; selectionStart: number; selectionEnd: number };

const ATTENTION_STEP = 1.1;
const NUMERIC_ATTENTION_STEP = 0.1;

/** Tolerance for floating-point weight comparisons. */
const WEIGHT_TOLERANCE = 0.001;

/** Tolerance for checking if a weight is a power of ATTENTION_STEP. */
const STEP_COUNT_TOLERANCE = 0.005;

// #region Weight Helpers

/**
 * Check if a weight is approximately ATTENTION_STEP^n for some integer n.
 * Returns n if so, or null if the weight is not a power of ATTENTION_STEP.
 */
function getAttentionStepCount(weight: number): number | null {
  if (weight <= 0) {
    return null;
  }
  if (Math.abs(weight - 1.0) < WEIGHT_TOLERANCE) {
    return 0;
  }
  const n = Math.round(Math.log(weight) / Math.log(ATTENTION_STEP));
  if (n === 0) {
    return null;
  }
  const expected = Math.pow(ATTENTION_STEP, n);
  if (Math.abs(expected - weight) < STEP_COUNT_TOLERANCE) {
    return n;
  }
  return null;
}

/**
 * Convert an Attention value ('+', '--', 1.2, etc.) into a numeric multiplier.
 */
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

/**
 * Combine an existing attention value with an additional '+' or '-' level.
 * Handles cancellation: e.g. '++' + '-' → '+', '+' + '-' → undefined (neutral).
 */
function addAttention(current: Attention | undefined, added: '+' | '-'): Attention | undefined {
  if (!current) {
    return added;
  }
  if (typeof current === 'number') {
    if (added === '+') {
      return Number((current * ATTENTION_STEP).toFixed(4));
    }
    return Number((current / ATTENTION_STEP).toFixed(4));
  }
  // Check if the added direction cancels the current one
  const isCancel = (current.startsWith('+') && added === '-') || (current.startsWith('-') && added === '+');
  if (isCancel) {
    const res = current.substring(1);
    return res === '' ? undefined : res;
  }
  return `${current}${added}`;
}

// #region Terminal Type

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

// #region Main Entry Point

/**
 * Adjusts the attention of the prompt at the current cursor/selection position.
 * Supports regular prompts and prompt functions (.and(), .or(), .blend()).
 *
 * When a selection spans across a prompt function's argument separator, each
 * affected argument is adjusted independently and simultaneously.
 */
export function adjustPromptAttention(
  prompt: string,
  selectionStart: number,
  selectionEnd: number,
  direction: AttentionDirection,
  prefersNumericWeights = false
): AdjustmentResult {
  try {
    const tokens = tokenize(prompt);
    const ast = parseTokens(tokens);

    const regions = extractRegions(ast);
    const processedNodes: ASTNode[] = [];
    let anyModified = false;

    for (const region of regions) {
      if (region.type === 'normal') {
        const clipped = clipSelection(selectionStart, selectionEnd, region.range);
        if (clipped) {
          const result = adjustRegionNodes(region.nodes, clipped.start, clipped.end, direction, prefersNumericWeights);
          if (result.modified) {
            anyModified = true;
          }
          processedNodes.push(...result.nodes);
        } else {
          processedNodes.push(...region.nodes);
        }
      } else {
        // prompt_function region
        const pfNode = region.node;
        const clipped = clipSelection(selectionStart, selectionEnd, pfNode.range);
        if (clipped) {
          const result = adjustPromptFunctionNode(pfNode, clipped.start, clipped.end, direction, prefersNumericWeights);
          if (result.modified) {
            anyModified = true;
          }
          processedNodes.push(result.node);
        } else {
          processedNodes.push(pfNode);
        }
      }
    }

    if (!anyModified) {
      return { prompt, selectionStart, selectionEnd };
    }

    return serializeWithSelection(processedNodes);
  } catch (e) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    log.error({ error: serializeError(e) as any }, 'Failed to adjust prompt attention');
    return { prompt, selectionStart, selectionEnd };
  }
}

// #region Region Extraction

type Region =
  | { type: 'normal'; nodes: ASTNode[]; range: { start: number; end: number } }
  | { type: 'prompt_function'; node: ASTNode & { type: 'prompt_function' } };

/**
 * Split the top-level AST into contiguous "normal" regions and prompt function regions.
 * This allows us to process prompt function arguments independently.
 */
function extractRegions(ast: ASTNode[]): Region[] {
  const regions: Region[] = [];
  let currentNormal: ASTNode[] = [];

  const flushNormal = () => {
    if (currentNormal.length > 0) {
      const first = currentNormal[0]!;
      const last = currentNormal[currentNormal.length - 1]!;
      regions.push({
        type: 'normal',
        nodes: currentNormal,
        range: { start: first.range.start, end: last.range.end },
      });
      currentNormal = [];
    }
  };

  for (const node of ast) {
    if (node.type === 'prompt_function') {
      flushNormal();
      regions.push({ type: 'prompt_function', node });
    } else {
      currentNormal.push(node);
    }
  }
  flushNormal();

  return regions;
}

/**
 * Clip a selection range to a target range. Returns null if there is no overlap.
 * For cursor positions (start === end), checks containment including boundaries.
 */
function clipSelection(
  selStart: number,
  selEnd: number,
  range: { start: number; end: number }
): { start: number; end: number } | null {
  if (selStart === selEnd) {
    // Cursor position: check if within range (inclusive of boundaries)
    if (selStart >= range.start && selStart <= range.end) {
      return { start: selStart, end: selEnd };
    }
    return null;
  }
  const clippedStart = Math.max(selStart, range.start);
  const clippedEnd = Math.min(selEnd, range.end);
  if (clippedStart >= clippedEnd) {
    return null;
  }
  return { start: clippedStart, end: clippedEnd };
}

// #region Prompt Function Handling

/**
 * Adjust attention within a prompt function node by processing each argument
 * whose content range overlaps the selection independently.
 * Returns the (possibly updated) node and whether any modification was made.
 */
function adjustPromptFunctionNode(
  pf: ASTNode & { type: 'prompt_function' },
  selStart: number,
  selEnd: number,
  direction: AttentionDirection,
  prefersNumericWeights = false
): { node: ASTNode & { type: 'prompt_function' }; modified: boolean } {
  let modified = false;
  const newArgs: PromptFunctionArg[] = pf.promptArgs.map((arg) => {
    const clipped = clipSelection(selStart, selEnd, arg.contentRange);
    if (clipped) {
      const result = adjustRegionNodes(arg.nodes, clipped.start, clipped.end, direction, prefersNumericWeights);
      if (result.modified) {
        modified = true;
        return { ...arg, nodes: result.nodes };
      }
    }
    return arg;
  });

  if (!modified) {
    return { node: pf, modified: false };
  }

  return { node: { ...pf, promptArgs: newArgs }, modified: true };
}

// #region Core Attention Adjustment

/**
 * Adjust attention for a set of AST nodes (a "region") given a selection range.
 * This is the core flatten → select → adjust → regroup pipeline.
 * Returns the adjusted nodes and whether any modification was made.
 */
function adjustRegionNodes(
  nodes: ASTNode[],
  selStart: number,
  selEnd: number,
  direction: AttentionDirection,
  prefersNumericWeights = false
): { nodes: ASTNode[]; modified: boolean } {
  const terminals = flattenAST(nodes);

  let selectedTerminals = selectTerminals(terminals, selStart, selEnd);

  // Fallback: if no terminals were selected, try to find an overlapping group
  if (selectedTerminals.length === 0) {
    const group = findSelectedGroup(nodes, selStart, selEnd);
    if (group) {
      selectedTerminals = terminals.filter((t) => t.range.start >= group.range.start && t.range.end <= group.range.end);
    }
  }

  if (selectedTerminals.length === 0) {
    return { nodes, modified: false };
  }

  for (const t of selectedTerminals) {
    t.isSelected = true;
    // When the user prefers numeric weights and the terminal doesn't already
    // have explicit attention, mark it as numeric so adjustWeights uses
    // additive steps and groupTerminals emits numeric syntax.
    if (prefersNumericWeights && !t.hasExplicitAttention) {
      t.hasNumericAttention = true;
    }
  }

  adjustWeights(selectedTerminals, direction);

  return { nodes: groupTerminals(terminals), modified: true };
}

// #region Flatten AST to Terminals

/**
 * Flatten an AST into a flat list of terminals, computing the effective weight
 * of each terminal by accumulating attention from ancestor groups.
 */
function flattenAST(
  ast: ASTNode[],
  currentWeight = 1.0,
  parentRange?: { start: number; end: number },
  numericAttention = false
): Terminal[] {
  const terminals: Terminal[] = [];

  for (const node of ast) {
    let nodeWeight = currentWeight;
    let nodeNumericAttention = numericAttention;
    if ((node.type === 'word' || node.type === 'group') && node.attention) {
      nodeWeight *= parseAttention(node.attention);
      nodeNumericAttention = typeof node.attention === 'number';
    }

    if (node.type === 'group') {
      terminals.push(...flattenAST(node.children, nodeWeight, node.range, nodeNumericAttention));
    } else if (node.type === 'prompt_function') {
      // Prompt functions should not appear inside regions being flattened;
      // they are handled at the region level. If one somehow appears, skip it.
      continue;
    } else {
      terminals.push({
        text: node.type === 'word' ? node.text : node.value,
        type: node.type,
        weight: nodeWeight,
        range: node.range,
        hasExplicitAttention: node.type === 'word' && !!node.attention,
        hasNumericAttention: nodeNumericAttention,
        parentRange,
        isSelected: false,
      });
    }
  }
  return terminals;
}

// #region Terminal Selection

/**
 * Find terminals that overlap the selection range and should be affected
 * by the attention adjustment. Handles partial group overlap carefully:
 * terminals with explicit attention inside partially-overlapping groups
 * are excluded to avoid corrupting explicit weights.
 *
 * When the cursor is at a boundary between two tokens (e.g. "word|,"),
 * both tokens technically overlap the cursor position. In this case we
 * prefer word/embedding terminals over punctuation/whitespace so that
 * adjusting attention at a word boundary doesn't accidentally include
 * adjacent punctuation.
 */
function selectTerminals(terminals: Terminal[], selStart: number, selEnd: number): Terminal[] {
  const result = terminals.filter((t) => {
    const isOverlapping =
      (t.range.start < selEnd && t.range.end > selStart) ||
      (selStart === selEnd && t.range.start <= selStart && t.range.end >= selStart);

    if (!isOverlapping) {
      return false;
    }

    if (t.parentRange) {
      const parentContainsSelection = t.parentRange.start <= selStart && t.parentRange.end >= selEnd;
      const selectionCoversParent = selStart <= t.parentRange.start && selEnd >= t.parentRange.end;

      if (!parentContainsSelection && !selectionCoversParent) {
        // Partial overlap between selection and parent group
        if (t.hasExplicitAttention) {
          return false; // Don't modify explicit weight in partially-overlapping group
        }
      }
    }
    return true;
  });

  // When the cursor is at a token boundary (no selection range), multiple tokens
  // can match. Prefer word/embedding terminals over punctuation/whitespace.
  if (selStart === selEnd && result.length > 1) {
    const contentTerminals = result.filter((t) => t.type === 'word' || t.type === 'embedding');
    if (contentTerminals.length > 0) {
      return contentTerminals;
    }
  }

  return result;
}

// #region Weight Adjustment

/**
 * Apply weight changes to the selected terminals based on direction.
 * Numeric weights use additive steps; +/- syntax uses multiplicative steps.
 * All results are rounded to 4 decimal places to prevent floating-point drift.
 */
function adjustWeights(terminals: Terminal[], direction: AttentionDirection): void {
  for (const terminal of terminals) {
    if (terminal.hasNumericAttention) {
      // Additive step for explicit numeric weights (e.g. 1.1 → 1.2)
      if (direction === 'increment') {
        terminal.weight = Number((terminal.weight + NUMERIC_ATTENTION_STEP).toFixed(4));
      } else {
        terminal.weight = Number((terminal.weight - NUMERIC_ATTENTION_STEP).toFixed(4));
      }
    } else {
      // Multiplicative step for +/- syntax weights, rounded to prevent drift
      if (direction === 'increment') {
        terminal.weight = Number((terminal.weight * ATTENTION_STEP).toFixed(4));
      } else {
        terminal.weight = Number((terminal.weight / ATTENTION_STEP).toFixed(4));
      }
    }
  }
}

// #region Find Selected Group (fallback)

/**
 * When no terminals directly overlap the selection (e.g. cursor is on a group
 * boundary character), find the innermost group that overlaps the selection.
 */
function findSelectedGroup(nodes: ASTNode[], start: number, end: number): ASTNode | null {
  for (const node of nodes) {
    if (node.type === 'group') {
      const foundInChildren = findSelectedGroup(node.children, start, end);
      if (foundInChildren) {
        return foundInChildren;
      }
      if (node.range.start < end && node.range.end > start) {
        return node;
      }
    }
  }
  return null;
}

// #region Regroup Terminals into AST

/**
 * Reconstruct an AST from a flat list of terminals with adjusted weights.
 * Groups consecutive terminals with compatible weights using +/- or numeric syntax.
 *
 * Note: Reconstructed group nodes use `range: { start: 0, end: 0 }` as a sentinel
 * value since the original source positions are no longer meaningful after regrouping.
 * These nodes are only used for serialization output, never for source-position lookups.
 */
function groupTerminals(terminals: Terminal[]): ASTNode[] {
  if (terminals.length === 0) {
    return [];
  }

  /** Sentinel range for reconstructed nodes whose original positions are not applicable. */
  const NO_RANGE = { start: 0, end: 0 };

  const nodes: ASTNode[] = [];
  let i = 0;

  while (i < terminals.length) {
    const t = terminals[i]!;
    const weight = t.weight;
    const stepCount = getAttentionStepCount(weight);

    // ── +/- attention (weight is a non-zero power of ATTENTION_STEP) ──
    // Skip this branch if the terminal prefers numeric format to avoid an
    // infinite loop (predicate would reject it, findRunEnd returns i, i never advances).
    if (stepCount !== null && stepCount !== 0 && !t.hasNumericAttention) {
      const isPositive = stepCount > 0;
      const sign: '+' | '-' = isPositive ? '+' : '-';
      const predicate = (t: Terminal): boolean => {
        if (t.hasNumericAttention) {
          return false; // Numeric-preference terminals should not join +/- runs
        }
        const sc = getAttentionStepCount(t.weight);
        return sc !== null && (isPositive ? sc > 0 : sc < 0);
      };
      const factor = isPositive ? ATTENTION_STEP : 1 / ATTENTION_STEP;

      const j = findRunEnd(terminals, i, predicate);

      // Trim whitespace from the content run boundaries
      let runStart = i;
      let runEnd = j;
      while (runStart < runEnd && terminals[runStart]!.type === 'whitespace') {
        runStart++;
      }
      while (runEnd > runStart && terminals[runEnd - 1]!.type === 'whitespace') {
        runEnd--;
      }

      // Emit leading whitespace as standalone nodes
      for (let k = i; k < runStart; k++) {
        nodes.push(createNodeFromTerminal(terminals[k]!));
      }

      if (runStart < runEnd) {
        // Factor out one level of attention and recurse
        const slice = terminals.slice(runStart, runEnd).map((t) => ({ ...t, weight: t.weight / factor }));
        const children = groupTerminals(slice);
        const isSelection = slice.every((t) => t.isSelected);

        if (children.length === 1) {
          const child = children[0]!;
          if (child.type === 'word' || child.type === 'group') {
            const newAttention = addAttention(child.attention, sign);
            nodes.push({ ...child, attention: newAttention, isSelection: isSelection || undefined });
          } else {
            nodes.push({ type: 'group', children, attention: sign, range: NO_RANGE, isSelection });
          }
        } else {
          nodes.push({ type: 'group', children, attention: sign, range: NO_RANGE, isSelection });
        }
      }

      // Emit trailing whitespace as standalone nodes
      for (let k = runEnd; k < j; k++) {
        nodes.push(createNodeFromTerminal(terminals[k]!));
      }

      i = j;
      continue;
    }

    // ── Neutral weight (≈ 1.0) ──
    if (Math.abs(weight - 1.0) < WEIGHT_TOLERANCE) {
      nodes.push(createNodeFromTerminal(t));
      i++;
      continue;
    }

    // ── Numeric weight (not a power of ATTENTION_STEP) ──
    {
      const j = findRunEnd(terminals, i, (t) => Math.abs(t.weight - weight) < WEIGHT_TOLERANCE);

      // Trim whitespace from the content run boundaries (same as +/- branch)
      let runStart = i;
      let runEnd = j;
      while (runStart < runEnd && terminals[runStart]!.type === 'whitespace') {
        runStart++;
      }
      while (runEnd > runStart && terminals[runEnd - 1]!.type === 'whitespace') {
        runEnd--;
      }

      // Emit leading whitespace as standalone nodes
      for (let k = i; k < runStart; k++) {
        nodes.push(createNodeFromTerminal(terminals[k]!));
      }

      if (runStart < runEnd) {
        const groupSlice = terminals.slice(runStart, runEnd).map((t) => ({ ...t, weight: 1.0 }));
        const children = groupTerminals(groupSlice);
        const isSelection = groupSlice.every((t) => t.isSelected);
        const weightNum = Number(weight.toFixed(4));

        nodes.push({ type: 'group', children, attention: weightNum, range: NO_RANGE, isSelection });
      }

      // Emit trailing whitespace as standalone nodes
      for (let k = runEnd; k < j; k++) {
        nodes.push(createNodeFromTerminal(terminals[k]!));
      }

      i = j;
    }
  }
  return nodes;
}

/**
 * Find the end of a "run" of terminals whose weights satisfy a predicate.
 * Whitespace terminals are included if the next non-whitespace terminal also satisfies the predicate.
 * Note: The returned index may point to a whitespace token that is NOT included in the run;
 * the caller is responsible for trimming trailing whitespace from the run boundaries.
 */
function findRunEnd(terminals: Terminal[], start: number, predicate: (t: Terminal) => boolean): number {
  let j = start;
  while (j < terminals.length) {
    const next = terminals[j]!;
    if (predicate(next)) {
      j++;
    } else if (next.type === 'whitespace') {
      // Look ahead past consecutive whitespace
      let k = j + 1;
      while (k < terminals.length && terminals[k]!.type === 'whitespace') {
        k++;
      }
      if (k < terminals.length && predicate(terminals[k]!)) {
        j = k;
      } else {
        break;
      }
    } else {
      break;
    }
  }
  return j;
}

/**
 * Convert a Terminal back into a leaf ASTNode.
 */
function createNodeFromTerminal(t: Terminal): ASTNode {
  switch (t.type) {
    case 'word':
      return { type: 'word', text: t.text, range: t.range, isSelection: t.isSelected || undefined };
    case 'whitespace':
      return { type: 'whitespace', value: t.text, range: t.range, isSelection: t.isSelected || undefined };
    case 'punct':
      return { type: 'punct', value: t.text, range: t.range, isSelection: t.isSelected || undefined };
    case 'embedding':
      return { type: 'embedding', value: t.text, range: t.range, isSelection: t.isSelected || undefined };
    case 'escaped_paren':
      return {
        type: 'escaped_paren',
        value: t.text as '(' | ')',
        range: t.range,
        isSelection: t.isSelected || undefined,
      };
    default:
      return { type: 'word', text: t.text, range: t.range, isSelection: t.isSelected || undefined };
  }
}
