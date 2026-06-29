import type { PromptAstNode, PromptAttention, PromptFunctionArg, PromptRange } from './ast';

import { parsePrompt, serializePromptWithSelection } from './ast';

export type PromptAttentionDirection = 'increment' | 'decrement';

export interface PromptAttentionAdjustment {
  prompt: string;
  selectionStart: number;
  selectionEnd: number;
}

const ATTENTION_FACTOR = 1.1;
const NUMERIC_STEP = 0.1;
const WEIGHT_EPSILON = 0.001;
const FACTOR_EPSILON = 0.005;
const GENERATED_RANGE: PromptRange = { start: 0, end: 0 };

type PromptFunctionNode = Extract<PromptAstNode, { type: 'prompt_function' }>;
type LeafNode = Exclude<PromptAstNode, { type: 'group' | 'prompt_function' }>;

interface WeightedLeaf {
  text: string;
  type: LeafNode['type'];
  weight: number;
  range: PromptRange;
  parentRange?: PromptRange;
  hasExplicitAttention: boolean;
  usesNumericAttention: boolean;
  isSelected: boolean;
}

type Region =
  | { type: 'normal'; nodes: PromptAstNode[]; range: PromptRange }
  | { type: 'prompt_function'; node: PromptFunctionNode };

const roundWeight = (weight: number): number => Number(weight.toFixed(4));

const isNeutralWeight = (weight: number): boolean => Math.abs(weight - 1) < WEIGHT_EPSILON;

const parseAttention = (attention: PromptAttention): number => {
  if (typeof attention === 'number') {
    return attention;
  }

  if (attention.startsWith('+')) {
    return ATTENTION_FACTOR ** attention.length;
  }

  if (attention.startsWith('-')) {
    return ATTENTION_FACTOR ** -attention.length;
  }

  const numeric = Number(attention);

  return Number.isNaN(numeric) ? 1 : numeric;
};

const getSymbolicStepCount = (weight: number): number | null => {
  if (weight <= 0) {
    return null;
  }

  if (isNeutralWeight(weight)) {
    return 0;
  }

  const steps = Math.round(Math.log(weight) / Math.log(ATTENTION_FACTOR));

  if (steps === 0) {
    return null;
  }

  return Math.abs(ATTENTION_FACTOR ** steps - weight) < FACTOR_EPSILON ? steps : null;
};

const addSymbolicAttention = (current: PromptAttention | undefined, next: '+' | '-'): PromptAttention | undefined => {
  if (current === undefined) {
    return next;
  }

  if (typeof current === 'number') {
    return next === '+' ? roundWeight(current * ATTENTION_FACTOR) : roundWeight(current / ATTENTION_FACTOR);
  }

  const cancels = (current.startsWith('+') && next === '-') || (current.startsWith('-') && next === '+');

  if (!cancels) {
    return `${current}${next}`;
  }

  const remaining = current.slice(1);

  return remaining ? remaining : undefined;
};

const clipRange = (selectionStart: number, selectionEnd: number, range: PromptRange): PromptRange | null => {
  if (selectionStart === selectionEnd) {
    return selectionStart >= range.start && selectionStart <= range.end
      ? { start: selectionStart, end: selectionEnd }
      : null;
  }

  const start = Math.max(selectionStart, range.start);
  const end = Math.min(selectionEnd, range.end);

  return start < end ? { start, end } : null;
};

const rangesOverlap = (selectionStart: number, selectionEnd: number, range: PromptRange): boolean => {
  if (selectionStart === selectionEnd) {
    return range.start <= selectionStart && range.end >= selectionStart;
  }

  return range.start < selectionEnd && range.end > selectionStart;
};

const extractRegions = (nodes: PromptAstNode[]): Region[] => {
  const regions: Region[] = [];
  let normalNodes: PromptAstNode[] = [];

  const flushNormalNodes = () => {
    const first = normalNodes[0];
    const last = normalNodes.at(-1);

    if (!first || !last) {
      return;
    }

    regions.push({ type: 'normal', nodes: normalNodes, range: { start: first.range.start, end: last.range.end } });
    normalNodes = [];
  };

  for (const node of nodes) {
    if (node.type === 'prompt_function') {
      flushNormalNodes();
      regions.push({ type: 'prompt_function', node });
    } else {
      normalNodes.push(node);
    }
  }

  flushNormalNodes();

  return regions;
};

const flattenNodes = (
  nodes: PromptAstNode[],
  inheritedWeight = 1,
  parentRange?: PromptRange,
  inheritedNumericAttention = false
): WeightedLeaf[] => {
  const leaves: WeightedLeaf[] = [];

  for (const node of nodes) {
    let weight = inheritedWeight;
    let usesNumericAttention = inheritedNumericAttention;

    if ((node.type === 'word' || node.type === 'group') && node.attention !== undefined) {
      weight *= parseAttention(node.attention);
      usesNumericAttention = typeof node.attention === 'number';
    }

    if (node.type === 'group') {
      leaves.push(...flattenNodes(node.children, weight, node.range, usesNumericAttention));
      continue;
    }

    if (node.type === 'prompt_function') {
      continue;
    }

    leaves.push({
      hasExplicitAttention: node.type === 'word' && node.attention !== undefined,
      isSelected: false,
      parentRange,
      range: node.range,
      text: node.type === 'word' ? node.text : node.value,
      type: node.type,
      usesNumericAttention,
      weight,
    });
  }

  return leaves;
};

const selectLeaves = (leaves: WeightedLeaf[], selectionStart: number, selectionEnd: number): WeightedLeaf[] => {
  const selected = leaves.filter((leaf) => {
    if (!rangesOverlap(selectionStart, selectionEnd, leaf.range)) {
      return false;
    }

    if (!leaf.parentRange) {
      return true;
    }

    const parentContainsSelection = leaf.parentRange.start <= selectionStart && leaf.parentRange.end >= selectionEnd;
    const selectionCoversParent = selectionStart <= leaf.parentRange.start && selectionEnd >= leaf.parentRange.end;

    return parentContainsSelection || selectionCoversParent || !leaf.hasExplicitAttention;
  });

  if (selectionStart !== selectionEnd || selected.length < 2) {
    return selected;
  }

  const contentLeaves = selected.filter((leaf) => leaf.type === 'word' || leaf.type === 'embedding');

  return contentLeaves.length > 0 ? contentLeaves : selected;
};

const findSelectedGroup = (
  nodes: PromptAstNode[],
  selectionStart: number,
  selectionEnd: number
): Extract<PromptAstNode, { type: 'group' }> | null => {
  for (const node of nodes) {
    if (node.type !== 'group') {
      continue;
    }

    const childGroup = findSelectedGroup(node.children, selectionStart, selectionEnd);

    if (childGroup) {
      return childGroup;
    }

    if (rangesOverlap(selectionStart, selectionEnd, node.range)) {
      return node;
    }
  }

  return null;
};

const applyWeightStep = (leaves: WeightedLeaf[], direction: PromptAttentionDirection): void => {
  for (const leaf of leaves) {
    if (leaf.usesNumericAttention) {
      leaf.weight = roundWeight(leaf.weight + (direction === 'increment' ? NUMERIC_STEP : -NUMERIC_STEP));
    } else {
      leaf.weight = roundWeight(leaf.weight * (direction === 'increment' ? ATTENTION_FACTOR : 1 / ATTENTION_FACTOR));
    }
  }
};

const createLeafNode = (leaf: WeightedLeaf): PromptAstNode => {
  const isSelection = leaf.isSelected || undefined;

  switch (leaf.type) {
    case 'word':
      return { type: 'word', text: leaf.text, range: leaf.range, isSelection };
    case 'whitespace':
      return { type: 'whitespace', value: leaf.text, range: leaf.range, isSelection };
    case 'punct':
      return { type: 'punct', value: leaf.text, range: leaf.range, isSelection };
    case 'embedding':
      return { type: 'embedding', value: leaf.text, range: leaf.range, isSelection };
    case 'escaped_paren':
      return { type: 'escaped_paren', value: leaf.text as '(' | ')', range: leaf.range, isSelection };
  }
};

const findRunEnd = (leaves: WeightedLeaf[], start: number, predicate: (leaf: WeightedLeaf) => boolean): number => {
  let end = start;

  while (end < leaves.length) {
    const leaf = leaves[end];

    if (!leaf) {
      break;
    }

    if (predicate(leaf)) {
      end++;
      continue;
    }

    if (leaf.type !== 'whitespace') {
      break;
    }

    let nextContent = end + 1;

    while (leaves[nextContent]?.type === 'whitespace') {
      nextContent++;
    }

    if (!leaves[nextContent] || !predicate(leaves[nextContent])) {
      break;
    }

    end = nextContent;
  }

  return end;
};

const trimWhitespaceRun = (leaves: WeightedLeaf[], start: number, end: number): PromptRange => {
  let trimmedStart = start;
  let trimmedEnd = end;

  while (trimmedStart < trimmedEnd && leaves[trimmedStart]?.type === 'whitespace') {
    trimmedStart++;
  }

  while (trimmedEnd > trimmedStart && leaves[trimmedEnd - 1]?.type === 'whitespace') {
    trimmedEnd--;
  }

  return { start: trimmedStart, end: trimmedEnd };
};

const pushLeaves = (nodes: PromptAstNode[], leaves: WeightedLeaf[], start: number, end: number): void => {
  for (let index = start; index < end; index++) {
    const leaf = leaves[index];

    if (leaf) {
      nodes.push(createLeafNode(leaf));
    }
  }
};

const selectedAll = (leaves: WeightedLeaf[]): boolean => leaves.length > 0 && leaves.every((leaf) => leaf.isSelected);

const createSymbolicRunNode = (leaves: WeightedLeaf[], sign: '+' | '-'): PromptAstNode | null => {
  const factor = sign === '+' ? ATTENTION_FACTOR : 1 / ATTENTION_FACTOR;
  const children = groupWeightedLeaves(leaves.map((leaf) => ({ ...leaf, weight: leaf.weight / factor })));

  if (children.length === 0) {
    return null;
  }

  const isSelection = selectedAll(leaves) || undefined;

  if (children.length === 1) {
    const child = children[0];

    if (child?.type === 'word' || child?.type === 'group') {
      return { ...child, attention: addSymbolicAttention(child.attention, sign), isSelection };
    }
  }

  return { type: 'group', attention: sign, children, range: GENERATED_RANGE, isSelection };
};

const createNumericRunNode = (leaves: WeightedLeaf[], weight: number): PromptAstNode | null => {
  const children = groupWeightedLeaves(leaves.map((leaf) => ({ ...leaf, weight: 1 })));

  if (children.length === 0) {
    return null;
  }

  return {
    type: 'group',
    attention: roundWeight(weight),
    children,
    range: GENERATED_RANGE,
    isSelection: selectedAll(leaves) || undefined,
  };
};

const groupWeightedLeaves = (leaves: WeightedLeaf[]): PromptAstNode[] => {
  const nodes: PromptAstNode[] = [];
  let index = 0;

  while (index < leaves.length) {
    const leaf = leaves[index];

    if (!leaf) {
      break;
    }

    const stepCount = getSymbolicStepCount(leaf.weight);

    if (stepCount !== null && stepCount !== 0 && !leaf.usesNumericAttention) {
      const isPositive = stepCount > 0;
      const sign = isPositive ? '+' : '-';
      const runEnd = findRunEnd(leaves, index, (candidate) => {
        if (candidate.usesNumericAttention) {
          return false;
        }

        const candidateSteps = getSymbolicStepCount(candidate.weight);

        return candidateSteps !== null && (isPositive ? candidateSteps > 0 : candidateSteps < 0);
      });
      const trimmed = trimWhitespaceRun(leaves, index, runEnd);

      pushLeaves(nodes, leaves, index, trimmed.start);

      const runNode = createSymbolicRunNode(leaves.slice(trimmed.start, trimmed.end), sign);

      if (runNode) {
        nodes.push(runNode);
      }

      pushLeaves(nodes, leaves, trimmed.end, runEnd);
      index = runEnd;
      continue;
    }

    if (isNeutralWeight(leaf.weight)) {
      nodes.push(createLeafNode(leaf));
      index++;
      continue;
    }

    const runWeight = leaf.weight;
    const runEnd = findRunEnd(leaves, index, (candidate) => Math.abs(candidate.weight - runWeight) < WEIGHT_EPSILON);
    const trimmed = trimWhitespaceRun(leaves, index, runEnd);

    pushLeaves(nodes, leaves, index, trimmed.start);

    const runNode = createNumericRunNode(leaves.slice(trimmed.start, trimmed.end), runWeight);

    if (runNode) {
      nodes.push(runNode);
    }

    pushLeaves(nodes, leaves, trimmed.end, runEnd);
    index = runEnd;
  }

  return nodes;
};

const adjustNodes = (
  nodes: PromptAstNode[],
  selectionStart: number,
  selectionEnd: number,
  direction: PromptAttentionDirection,
  preferNumericAttentionStyle: boolean
): { nodes: PromptAstNode[]; modified: boolean } => {
  const leaves = flattenNodes(nodes);
  let selectedLeaves = selectLeaves(leaves, selectionStart, selectionEnd);

  if (selectedLeaves.length === 0) {
    const group = findSelectedGroup(nodes, selectionStart, selectionEnd);

    if (group) {
      selectedLeaves = leaves.filter(
        (leaf) => leaf.range.start >= group.range.start && leaf.range.end <= group.range.end
      );
    }
  }

  if (selectedLeaves.length === 0) {
    return { nodes, modified: false };
  }

  for (const leaf of selectedLeaves) {
    leaf.isSelected = true;

    if (preferNumericAttentionStyle && !leaf.hasExplicitAttention) {
      leaf.usesNumericAttention = true;
    }
  }

  applyWeightStep(selectedLeaves, direction);

  return { nodes: groupWeightedLeaves(leaves), modified: true };
};

const adjustPromptFunction = (
  node: PromptFunctionNode,
  selectionStart: number,
  selectionEnd: number,
  direction: PromptAttentionDirection,
  preferNumericAttentionStyle: boolean
): { node: PromptFunctionNode; modified: boolean } => {
  let modified = false;
  const promptArgs: PromptFunctionArg[] = node.promptArgs.map((arg) => {
    const clipped = clipRange(selectionStart, selectionEnd, arg.contentRange);

    if (!clipped) {
      return arg;
    }

    const result = adjustNodes(arg.nodes, clipped.start, clipped.end, direction, preferNumericAttentionStyle);

    if (!result.modified) {
      return arg;
    }

    modified = true;
    return { ...arg, nodes: result.nodes };
  });

  return modified ? { node: { ...node, promptArgs }, modified: true } : { node, modified: false };
};

const adjustPromptAttentionUnchecked = (
  prompt: string,
  selectionStart: number,
  selectionEnd: number,
  direction: PromptAttentionDirection,
  preferNumericAttentionStyle = false
): PromptAttentionAdjustment => {
  const nodes = parsePrompt(prompt);
  const nextNodes: PromptAstNode[] = [];
  let modified = false;

  for (const region of extractRegions(nodes)) {
    if (region.type === 'normal') {
      const clipped = clipRange(selectionStart, selectionEnd, region.range);

      if (!clipped) {
        nextNodes.push(...region.nodes);
        continue;
      }

      const result = adjustNodes(region.nodes, clipped.start, clipped.end, direction, preferNumericAttentionStyle);
      modified = modified || result.modified;
      nextNodes.push(...result.nodes);
      continue;
    }

    const clipped = clipRange(selectionStart, selectionEnd, region.node.range);

    if (!clipped) {
      nextNodes.push(region.node);
      continue;
    }

    const result = adjustPromptFunction(
      region.node,
      clipped.start,
      clipped.end,
      direction,
      preferNumericAttentionStyle
    );
    modified = modified || result.modified;
    nextNodes.push(result.node);
  }

  return modified ? serializePromptWithSelection(nextNodes) : { prompt, selectionStart, selectionEnd };
};

export const adjustPromptAttention = (
  prompt: string,
  selectionStart: number,
  selectionEnd: number,
  direction: PromptAttentionDirection,
  preferNumericAttentionStyle = false
): PromptAttentionAdjustment => {
  try {
    return adjustPromptAttentionUnchecked(prompt, selectionStart, selectionEnd, direction, preferNumericAttentionStyle);
  } catch {
    return { prompt, selectionStart, selectionEnd };
  }
};
