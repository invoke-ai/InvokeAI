import type { PromptAstNode, PromptAttention, PromptRange } from './ast';

import { parsePrompt, serializePromptWithSelection } from './ast';

export type PromptAttentionDirection = 'increment' | 'decrement';

export interface PromptAttentionAdjustment {
  prompt: string;
  selectionStart: number;
  selectionEnd: number;
}

type LeafNode = Exclude<PromptAstNode, { type: 'group' }>;

interface Weight {
  value: number;
  /** Exact symbolic steps avoid logarithms, rounding drift and tolerance-based regrouping. */
  steps: number | null;
  explicit: boolean;
}

interface WeightedLeaf {
  node: LeafNode;
  weight: Weight;
  selected: boolean;
}

const NEUTRAL: Weight = { value: 1, steps: 0, explicit: false };
const GENERATED_RANGE: PromptRange = { start: 0, end: 0 };
const roundWeight = (weight: number): number => Number(weight.toPrecision(15));
const symbolicValue = (steps: number): number => (steps >= 0 ? 1.1 ** steps : 0.9 ** -steps);

const combineWeights = (parent: Weight, child: Weight): Weight => {
  if (!parent.explicit) {
    return child;
  }
  if (!child.explicit) {
    return parent;
  }
  // Opposite symbolic signs multiply in Compel (1.1 * 0.9 = 0.99); they do not cancel.
  const combinedSteps =
    parent.steps !== null &&
    child.steps !== null &&
    (parent.steps === 0 || child.steps === 0 || Math.sign(parent.steps) === Math.sign(child.steps))
      ? parent.steps + child.steps
      : null;
  return {
    value: combinedSteps === null ? roundWeight(parent.value * child.value) : symbolicValue(combinedSteps),
    steps: combinedSteps,
    explicit: parent.explicit || child.explicit,
  };
};

const inheritWeight = (parent: Weight, attention: PromptAttention | undefined): Weight => {
  if (attention === undefined) {
    return parent;
  }
  const steps = typeof attention === 'number' ? null : attention.startsWith('+') ? attention.length : -attention.length;
  return combineWeights(parent, {
    value: typeof attention === 'number' ? attention : symbolicValue(steps!),
    steps,
    explicit: true,
  });
};

const overlaps = (selection: PromptRange, range: PromptRange): boolean =>
  selection.start === selection.end
    ? range.start <= selection.start && range.end >= selection.start
    : range.start < selection.end && range.end > selection.start;

/** A selection of a group's delimiters/weight targets that group's contents. */
const resolveSelection = (nodes: PromptAstNode[], selection: PromptRange): PromptRange => {
  for (const node of nodes) {
    if (node.type !== 'group' || !overlaps(selection, node.range)) {
      continue;
    }
    const first = node.children[0];
    const last = node.children.at(-1);
    if (!first || !last) {
      continue;
    }
    if (
      selection.start >= node.range.start &&
      selection.end <= node.range.end &&
      (selection.start === selection.end
        ? selection.end < first.range.start || selection.start > last.range.end
        : selection.end <= first.range.start || selection.start >= last.range.end)
    ) {
      return { start: first.range.start, end: last.range.end };
    }
    const childSelection = resolveSelection(node.children, selection);
    if (childSelection !== selection) {
      return childSelection;
    }
  }
  return selection;
};

const flattenNodes = (nodes: PromptAstNode[], leaves: WeightedLeaf[], weight = NEUTRAL): void => {
  for (const node of nodes) {
    const inherited = node.type === 'group' || node.type === 'word' ? inheritWeight(weight, node.attention) : weight;
    if (node.type === 'group') {
      flattenNodes(node.children, leaves, inherited);
    } else {
      const previous = leaves.at(-1);
      // Preserve Compel's implicit word boundaries when removing parentheses.
      if (previous?.node.type === 'word' && node.type === 'word' && previous.node.range.end < node.range.start) {
        leaves.push({
          node: { type: 'whitespace', value: ' ', range: { start: previous.node.range.end, end: node.range.start } },
          weight: NEUTRAL,
          selected: false,
        });
      }
      leaves.push({ node, weight: inherited, selected: false });
    }
  }
};

const selectLeaves = (leaves: WeightedLeaf[], selection: PromptRange): void => {
  if (selection.start !== selection.end) {
    for (const leaf of leaves) {
      leaf.selected = overlaps(selection, leaf.node.range);
    }
    return;
  }
  // Prefer content over punctuation/whitespace, then the token starting at the caret; edit only one neighbor.
  let best: WeightedLeaf | undefined;
  let bestRank = -1;
  for (const leaf of leaves) {
    if (!overlaps(selection, leaf.node.range)) {
      continue;
    }
    const content = leaf.node.type === 'word' || leaf.node.type === 'embedding' || leaf.node.type === 'prompt_function';
    const rank = (content ? 2 : 0) + (leaf.node.range.start === selection.start ? 1 : 0);
    if (rank > bestRank) {
      best = leaf;
      bestRank = rank;
    }
  }
  if (best) {
    best.selected = true;
  }
};

const stepWeight = (weight: Weight, direction: PromptAttentionDirection, preferNumeric: boolean): Weight => {
  const delta = direction === 'increment' ? 1 : -1;
  if (weight.steps === null || (preferNumeric && !weight.explicit)) {
    return { value: roundWeight(weight.value + delta * 0.1), steps: null, explicit: true };
  }
  const steps = weight.steps + delta;
  return { value: symbolicValue(steps), steps, explicit: true };
};

const leafNode = (leaf: WeightedLeaf): PromptAstNode => {
  const isSelection = leaf.selected || undefined;
  return leaf.node.type === 'word'
    ? { ...leaf.node, attention: undefined, isSelection }
    : { ...leaf.node, isSelection };
};

const sameWeight = (a: Weight, b: Weight): boolean => {
  if (a.steps !== null && b.steps !== null) {
    return a.steps === b.steps;
  }
  return a.value === b.value || roundWeight(a.value) === roundWeight(b.value);
};

/** Equal numeric and symbolic neighbours must merge in either document order. */
const unifyEqualWeights = (leaves: WeightedLeaf[]): void => {
  let start = 0;
  while (start < leaves.length) {
    const first = leaves[start]!;
    if (first.node.type === 'whitespace') {
      start++;
      continue;
    }
    let end = start + 1;
    let numeric = first.weight.steps === null;
    while (end < leaves.length) {
      const leaf = leaves[end]!;
      if (leaf.node.type !== 'whitespace') {
        if (!sameWeight(first.weight, leaf.weight)) {
          break;
        }
        numeric ||= leaf.weight.steps === null;
      }
      end++;
    }
    if (numeric) {
      for (let index = start; index < end; index++) {
        const leaf = leaves[index]!;
        if (leaf.node.type !== 'whitespace') {
          leaf.weight = { ...leaf.weight, steps: null };
        }
      }
    }
    start = end;
  }
};

const wrapRun = (children: PromptAstNode[], attention: PromptAttention, selected: boolean): PromptAstNode => {
  const child = children.length === 1 ? children[0] : undefined;
  if (typeof attention === 'string' && (child?.type === 'word' || child?.type === 'group')) {
    return { ...child, attention, isSelection: selected || undefined };
  }
  return { type: 'group', attention, children, range: GENERATED_RANGE, isSelection: selected || undefined };
};

const groupWeightedLeaves = (leaves: WeightedLeaf[]): PromptAstNode[] => {
  const nodes: PromptAstNode[] = [];
  let index = 0;
  while (index < leaves.length) {
    const first = leaves[index]!;
    if (first.node.type === 'whitespace' || first.weight.value === 1) {
      nodes.push(leafNode(first));
      index++;
      continue;
    }
    const symbolic = first.weight.steps !== null;
    const sign = Math.sign(first.weight.steps ?? 0);
    let commonSteps = first.weight.steps ?? 0;
    let end = index + 1;
    let contentEnd = end;
    while (end < leaves.length) {
      const leaf = leaves[end]!;
      if (leaf.node.type !== 'whitespace') {
        if (
          symbolic
            ? leaf.weight.steps === null || Math.sign(leaf.weight.steps) !== sign
            : !sameWeight(first.weight, leaf.weight)
        ) {
          break;
        }
        if (symbolic) {
          commonSteps = sign * Math.min(Math.abs(commonSteps), Math.abs(leaf.weight.steps!));
        }
        contentEnd = end + 1;
      }
      end++;
    }
    const run = leaves.slice(index, contentEnd);
    const children = symbolic
      ? groupWeightedLeaves(
          run.map((leaf) => {
            const steps = leaf.node.type === 'whitespace' ? 0 : leaf.weight.steps! - commonSteps;
            return { ...leaf, weight: { ...leaf.weight, steps, value: symbolicValue(steps) } };
          })
        )
      : run.map(leafNode);
    const attention = symbolic ? (sign > 0 ? '+' : '-').repeat(Math.abs(commonSteps)) : first.weight.value;
    nodes.push(
      wrapRun(
        children,
        attention,
        run.every((leaf) => leaf.selected)
      )
    );
    index = contentEnd;
  }
  return nodes;
};

const adjustNodes = (
  nodes: PromptAstNode[],
  selection: PromptRange,
  direction: PromptAttentionDirection,
  preferNumeric: boolean,
  inherited = NEUTRAL
): { nodes: PromptAstNode[]; modified: boolean } => {
  const target = resolveSelection(nodes, selection);
  const leaves: WeightedLeaf[] = [];
  flattenNodes(nodes, leaves);
  selectLeaves(leaves, target);
  let modified = false;
  for (const leaf of leaves) {
    if (!Number.isFinite(leaf.weight.value)) {
      throw new TypeError('Non-finite prompt attention');
    }
    if (!leaf.selected) {
      continue;
    }
    if (leaf.node.type === 'prompt_function') {
      const context = combineWeights(inherited, leaf.weight);
      let functionModified = false;
      const promptArgs = leaf.node.promptArgs.map((arg) => {
        if (!overlaps(target, arg.contentRange)) {
          return arg;
        }
        const result = adjustNodes(arg.nodes, target, direction, preferNumeric, context);
        functionModified ||= result.modified;
        return result.modified ? { ...arg, nodes: result.nodes } : arg;
      });
      leaf.node = functionModified ? { ...leaf.node, promptArgs } : leaf.node;
      leaf.selected = false;
      modified ||= functionModified;
      continue;
    }
    if (leaf.node.type === 'whitespace') {
      continue;
    }
    if (inherited.value === 0) {
      leaf.selected = false;
      continue;
    }
    const stepped = stepWeight(combineWeights(inherited, leaf.weight), direction, preferNumeric);
    const relativeSteps = stepped.steps !== null && inherited.steps !== null ? stepped.steps - inherited.steps : null;
    const canUseSymbolic =
      relativeSteps !== null &&
      (relativeSteps === 0 || inherited.steps === 0 || Math.sign(relativeSteps) === Math.sign(inherited.steps!));
    leaf.weight = {
      value: canUseSymbolic ? symbolicValue(relativeSteps) : roundWeight(stepped.value / inherited.value),
      steps: canUseSymbolic ? relativeSteps : null,
      explicit: true,
    };
    if (!Number.isFinite(leaf.weight.value)) {
      throw new TypeError('Non-finite prompt attention');
    }
    modified = true;
  }
  if (!modified) {
    return { nodes, modified: false };
  }
  unifyEqualWeights(leaves);
  return { nodes: groupWeightedLeaves(leaves), modified: true };
};

export const adjustPromptAttention = (
  prompt: string,
  selectionStart: number,
  selectionEnd: number,
  direction: PromptAttentionDirection,
  preferNumericAttentionStyle = false
): PromptAttentionAdjustment => {
  const unchanged = { prompt, selectionStart, selectionEnd };
  try {
    const selection = {
      start: Math.max(0, Math.min(prompt.length, selectionStart, selectionEnd)),
      end: Math.min(prompt.length, Math.max(0, selectionStart, selectionEnd)),
    };
    const result = adjustNodes(parsePrompt(prompt), selection, direction, preferNumericAttentionStyle);
    return result.modified ? serializePromptWithSelection(result.nodes, prompt) : unchanged;
  } catch {
    // Prompts are edited while incomplete, and pathological nesting must not break typing.
    return unchanged;
  }
};
