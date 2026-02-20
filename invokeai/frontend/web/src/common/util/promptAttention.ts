import { logger } from 'app/logging/logger';
import { serializeError } from 'serialize-error';

import { type ASTNode, type Attention, parseTokens, type PromptFunctionArg, tokenize } from './promptAST';

const log = logger('events');

type AttentionDirection = 'increment' | 'decrement';
type AdjustmentResult = { prompt: string; selectionStart: number; selectionEnd: number };

const ATTENTION_STEP = 1.1;
const NUMERIC_ATTENTION_STEP = 0.1;

// #region Weight Helpers

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
  if (added === '+') {
    if (current.startsWith('-')) {
      const res = current.substring(1);
      return res === '' ? undefined : res;
    }
    return `${current}+`;
  }
  // added === '-'
  if (current.startsWith('+')) {
    const res = current.substring(1);
    return res === '' ? undefined : res;
  }
  return `${current}-`;
}

// #reigion Terminal Type

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
          const adjusted = adjustRegionNodes(
            region.nodes,
            clipped.start,
            clipped.end,
            direction,
            prefersNumericWeights
          );
          if (adjusted !== region.nodes) {
            anyModified = true;
          }
          processedNodes.push(...adjusted);
        } else {
          processedNodes.push(...region.nodes);
        }
      } else {
        // prompt_function region
        const pfNode = region.node;
        const clipped = clipSelection(selectionStart, selectionEnd, pfNode.range);
        if (clipped) {
          const adjusted = adjustPromptFunctionNode(
            pfNode,
            clipped.start,
            clipped.end,
            direction,
            prefersNumericWeights
          );
          if (adjusted !== pfNode) {
            anyModified = true;
          }
          processedNodes.push(adjusted);
        } else {
          processedNodes.push(pfNode);
        }
      }
    }

    if (!anyModified) {
      return { prompt, selectionStart, selectionEnd };
    }

    const result = serializeWithSelection(processedNodes);
    return {
      prompt: result.prompt,
      selectionStart: result.selectionStart,
      selectionEnd: result.selectionEnd,
    };
  } catch (e) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    log.error({ error: serializeError(e) as any }, 'Failed to adjust prompt attention');
    return { prompt, selectionStart, selectionEnd };
  }
}

// ─── Region Extraction ─────────────────────────────────────────────────────────

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
 */
function adjustPromptFunctionNode(
  pf: ASTNode & { type: 'prompt_function' },
  selStart: number,
  selEnd: number,
  direction: AttentionDirection,
  prefersNumericWeights = false
): ASTNode & { type: 'prompt_function' } {
  let modified = false;
  const newArgs: PromptFunctionArg[] = pf.promptArgs.map((arg) => {
    const clipped = clipSelection(selStart, selEnd, arg.contentRange);
    if (clipped) {
      const adjusted = adjustRegionNodes(arg.nodes, clipped.start, clipped.end, direction, prefersNumericWeights);
      if (adjusted !== arg.nodes) {
        modified = true;
        return { ...arg, nodes: adjusted };
      }
    }
    return arg;
  });

  if (!modified) {
    return pf;
  }

  return { ...pf, promptArgs: newArgs };
}

// #region Core Attention Adjustment ─────────────────────────────────────────────────

/**
 * Adjust attention for a set of AST nodes (a "region") given a selection range.
 * This is the core flatten → select → adjust → regroup pipeline.
 */
function adjustRegionNodes(
  nodes: ASTNode[],
  selStart: number,
  selEnd: number,
  direction: AttentionDirection,
  prefersNumericWeights = false
): ASTNode[] {
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
    return nodes;
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

  return groupTerminals(terminals);
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
    if ('attention' in node && node.attention) {
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
        hasExplicitAttention: 'attention' in node && !!node.attention,
        hasNumericAttention: nodeNumericAttention,
        parentRange,
        isSelected: false,
      });
    }
  }
  return terminals;
}

// #reigion Terminal Selection

/**
 * Find terminals that overlap the selection range and should be affected
 * by the attention adjustment. Handles partial group overlap carefully:
 * terminals with explicit attention inside partially-overlapping groups
 * are excluded to avoid corrupting explicit weights.
 */
function selectTerminals(terminals: Terminal[], selStart: number, selEnd: number): Terminal[] {
  return terminals.filter((t) => {
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
}

// #region Weight Adjustment

/**
 * Apply weight changes to the selected terminals based on direction.
 * Numeric weights use additive steps; +/- syntax uses multiplicative steps.
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
      // Multiplicative step for +/- syntax weights
      if (direction === 'increment') {
        terminal.weight *= ATTENTION_STEP;
      } else {
        terminal.weight /= ATTENTION_STEP;
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
 */
function groupTerminals(terminals: Terminal[]): ASTNode[] {
  if (terminals.length === 0) {
    return [];
  }

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
            nodes.push({ type: 'group', children, attention: sign, range: { start: 0, end: 0 }, isSelection });
          }
        } else {
          nodes.push({ type: 'group', children, attention: sign, range: { start: 0, end: 0 }, isSelection });
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
    if (Math.abs(weight - 1.0) < 0.001) {
      nodes.push(createNodeFromTerminal(t));
      i++;
      continue;
    }

    // ── Numeric weight (not a power of ATTENTION_STEP) ──
    {
      let j = i;
      while (j < terminals.length && Math.abs(terminals[j]!.weight - weight) < 0.001) {
        j++;
      }

      const groupSlice = terminals.slice(i, j).map((t) => ({ ...t, weight: 1.0 }));
      const children = groupTerminals(groupSlice);
      const isSelection = groupSlice.every((t) => t.isSelected);
      const weightNum = Number(weight.toFixed(4));

      nodes.push({ type: 'group', children, attention: weightNum, range: { start: 0, end: 0 }, isSelection });
      i = j;
    }
  }
  return nodes;
}

/**
 * Find the end of a "run" of terminals whose weights satisfy a predicate.
 * Whitespace terminals are included if the next non-whitespace terminal also satisfies the predicate.
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

// #region Serialize with Selection Tracking

/**
 * Serialize an AST to a prompt string while simultaneously computing the
 * selection range from `isSelection` flags on nodes.
 *
 * This is more reliable than separate serialize + selection computation because
 * the position tracking is guaranteed to match the serialized output.
 */
function serializeWithSelection(ast: ASTNode[]): { prompt: string; selectionStart: number; selectionEnd: number } {
  let prompt = '';
  let selStart = Infinity;
  let selEnd = -1;

  function markSelected(nodeStart: number, nodeEnd: number) {
    selStart = Math.min(selStart, nodeStart);
    selEnd = Math.max(selEnd, nodeEnd);
  }

  function visit(nodes: ASTNode[]) {
    for (const node of nodes) {
      const nodeStart = prompt.length;

      switch (node.type) {
        case 'punct':
        case 'whitespace': {
          prompt += node.value;
          break;
        }
        case 'escaped_paren': {
          prompt += `\\${node.value}`;
          break;
        }
        case 'word': {
          prompt += node.text;
          if (node.attention) {
            prompt += String(node.attention);
          }
          break;
        }
        case 'group': {
          prompt += '(';
          visit(node.children);
          prompt += ')';
          if (node.attention) {
            prompt += String(node.attention);
          }
          break;
        }
        case 'embedding': {
          prompt += `<${node.value}>`;
          break;
        }
        case 'prompt_function': {
          prompt += '(';
          for (let idx = 0; idx < node.promptArgs.length; idx++) {
            if (idx > 0) {
              prompt += ', ';
            }
            const arg = node.promptArgs[idx]!;
            prompt += arg.quote;
            visit(arg.nodes);
            prompt += arg.quote;
          }
          prompt += ').';
          prompt += node.name;
          prompt += '(';
          prompt += node.functionParams;
          prompt += ')';
          break;
        }
      }

      if (node.isSelection) {
        markSelected(nodeStart, prompt.length);
      }
    }
  }

  visit(ast);

  if (selStart === Infinity) {
    selStart = 0;
    selEnd = prompt.length;
  }

  return { prompt, selectionStart: selStart, selectionEnd: selEnd };
}
