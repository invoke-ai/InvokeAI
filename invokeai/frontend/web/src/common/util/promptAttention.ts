import { logger } from 'app/logging/logger';
import { serializeError } from 'serialize-error';

import { type ASTNode, type Attention, parseTokens, serialize, tokenize } from './promptAST';

const log = logger('events');

type AttentionDirection = 'increment' | 'decrement';
type SelectionBounds = { start: number; end: number };
type AdjustmentResult = { prompt: string; selectionStart: number; selectionEnd: number };

type PositionedNode = {
  node: ASTNode;
  start: number;
  end: number;
  contentStart: number;
  contentEnd: number;
  parent: PositionedNode | null;
};

// ============================================================================
// ATTENTION HELPERS
// ============================================================================

function isSymbolAttention(a: Attention | undefined): a is string {
  return typeof a === 'string' && /^[+-]+$/.test(a);
}

function isNumericAttention(a: Attention | undefined): a is number {
  return typeof a === 'number';
}

/** Convert attention to numeric level: + = 1, ++ = 2, - = -1, 1.1 = 1, etc. */
function attentionToLevel(attention: Attention | undefined): number {
  if (isNumericAttention(attention)) {
    return Math.round((attention - 1.0) * 10);
  }
  if (!attention) {
    return 0;
  }
  let level = 0;
  for (const c of String(attention)) {
    if (c === '+') {
      level++;
    } else if (c === '-') {
      level--;
    }
  }
  return level;
}

/** Convert level back to symbol attention */
function levelToSymbol(level: number): string | undefined {
  if (level === 0) {
    return undefined;
  }
  return level > 0 ? '+'.repeat(level) : '-'.repeat(-level);
}

/** Combine two attention values (for flattening group attention onto children) */
function combineAttention(a: Attention | undefined, b: Attention | undefined): Attention | undefined {
  return levelToSymbol(attentionToLevel(a) + attentionToLevel(b));
}

/** Adjust attention by one step in the given direction */
function computeAttention(direction: AttentionDirection, attention: Attention | undefined): Attention | undefined {
  if (isNumericAttention(attention)) {
    const result = parseFloat((attention + (direction === 'increment' ? 0.1 : -0.1)).toFixed(1));
    return result === 1.0 ? undefined : result;
  }
  if (isSymbolAttention(attention)) {
    const level = attentionToLevel(attention);
    return levelToSymbol(direction === 'increment' ? level + 1 : level - 1);
  }
  return direction === 'increment' ? '+' : '-';
}

/** Apply attention to a node, flattening groups if combined attention is neutral */
function applyAttentionToNode(node: ASTNode, attention: Attention | undefined): ASTNode[] {
  if (node.type === 'word') {
    const combined = combineAttention(attention, node.attention);
    return combined === node.attention ? [node] : [{ ...node, attention: combined }];
  }
  if (node.type === 'group') {
    const combined = combineAttention(attention, node.attention);
    if (combined === undefined) {
      return node.children.flatMap((c) => applyAttentionToNode(c, undefined));
    }
    return [{ ...node, attention: combined }];
  }
  return [node];
}

/** Adjust a content node's attention, unwrapping groups if result is neutral */
function adjustContentNode(node: ASTNode, direction: AttentionDirection): ASTNode[] {
  if (node.type === 'word') {
    return [{ ...node, attention: computeAttention(direction, node.attention) }];
  }
  if (node.type === 'group') {
    const newAttn = computeAttention(direction, node.attention);
    return newAttn === undefined ? node.children : [{ ...node, attention: newAttn }];
  }
  return [node];
}

// ============================================================================
// POSITION MAPPING
// ============================================================================

/**
 * Builds a flat map of all nodes with their positions in the prompt string.
 * Groups include both their full bounds and content bounds.
 */
function buildPositionMap(
  ast: ASTNode[],
  startPos = 0,
  parent: PositionedNode | null = null
): { positions: PositionedNode[]; endPos: number } {
  const positions: PositionedNode[] = [];
  let currentPos = startPos;

  for (const node of ast) {
    const nodeStart = currentPos;
    let contentStart = currentPos;
    let contentEnd = currentPos;
    let nodeEnd = currentPos;

    switch (node.type) {
      case 'word': {
        nodeEnd = currentPos + node.text.length;
        if (node.attention !== undefined) {
          nodeEnd += String(node.attention).length;
        }
        contentStart = currentPos;
        contentEnd = currentPos + node.text.length;
        currentPos = nodeEnd;
        break;
      }

      case 'whitespace':
      case 'punct': {
        nodeEnd = currentPos + node.value.length;
        contentStart = nodeStart;
        contentEnd = nodeEnd;
        currentPos = nodeEnd;
        break;
      }

      case 'escaped_paren': {
        nodeEnd = currentPos + 2; // \( or \)
        contentStart = nodeStart;
        contentEnd = nodeEnd;
        currentPos = nodeEnd;
        break;
      }

      case 'embedding': {
        nodeEnd = currentPos + node.value.length + 2; // <value>
        contentStart = currentPos + 1;
        contentEnd = currentPos + 1 + node.value.length;
        currentPos = nodeEnd;
        break;
      }

      case 'group': {
        // Opening paren
        currentPos += 1;
        contentStart = currentPos;

        // Create placeholder for parent reference
        const groupNode: PositionedNode = {
          node,
          start: nodeStart,
          end: nodeStart, // Will be updated
          contentStart,
          contentEnd: contentStart, // Will be updated
          parent,
        };

        // Process children with this group as parent
        const childResult = buildPositionMap(node.children, currentPos, groupNode);
        positions.push(...childResult.positions);
        currentPos = childResult.endPos;

        contentEnd = currentPos;

        // Closing paren
        currentPos += 1;

        // Attention
        if (node.attention !== undefined) {
          currentPos += String(node.attention).length;
        }

        nodeEnd = currentPos;

        // Update the group node with final positions
        groupNode.end = nodeEnd;
        groupNode.contentEnd = contentEnd;

        positions.push(groupNode);
        continue; // Skip the push at the end
      }
    }

    positions.push({
      node,
      start: nodeStart,
      end: nodeEnd,
      contentStart,
      contentEnd,
      parent,
    });
  }

  return { positions, endPos: currentPos };
}

// ============================================================================
// NODE FINDING
// ============================================================================

/**
 * Finds the deepest group that fully contains the selection.
 * Returns null if selection is not fully within any group.
 */
function findEnclosingGroup(positions: PositionedNode[], selection: SelectionBounds): PositionedNode | null {
  const groups = positions
    .filter((p) => p.node.type === 'group')
    .filter((p) => selection.start >= p.start && selection.end <= p.end)
    // Sort by size (smallest = deepest nesting)
    .sort((a, b) => a.end - a.start - (b.end - b.start));

  return groups[0] ?? null;
}

/**
 * Checks if the cursor/selection is at the boundary of a group
 * (touching parentheses or weight).
 */
function isTouchingGroupBoundary(group: PositionedNode, selection: SelectionBounds): boolean {
  const { start, end, contentStart, contentEnd } = group;

  // Touching or at opening paren
  if (selection.start <= contentStart && selection.end <= contentStart) {
    return true;
  }

  // Touching or at closing paren/weight
  if (selection.start >= contentEnd && selection.end >= contentEnd) {
    return true;
  }

  // Selection spans the entire group content
  if (selection.start <= contentStart && selection.end >= contentEnd) {
    return true;
  }

  // Cursor is exactly at group start or end
  if (selection.start === selection.end) {
    if (selection.start === start || selection.start === end) {
      return true;
    }
  }

  return false;
}

/**
 * Finds content nodes (words, groups, embeddings) that intersect with selection.
 */
function findContentNodes(positions: PositionedNode[], selection: SelectionBounds): PositionedNode[] {
  return positions.filter((p) => {
    // Only content nodes
    if (p.node.type !== 'word' && p.node.type !== 'group' && p.node.type !== 'embedding') {
      return false;
    }

    // Check intersection
    return !(selection.end <= p.start || selection.start >= p.end);
  });
}

/**
 * Elevates nodes to their parent groups when selection crosses group boundaries.
 * This ensures we don't try to extract partial groups which would break parsing.
 *
 * For example, if selection spans from inside a group to outside it,
 * the nodes inside the group are replaced with the group itself.
 */
function elevateToTopLevelNodes(nodes: PositionedNode[]): PositionedNode[] {
  if (nodes.length === 0) {
    return nodes;
  }

  // Check if any nodes have different parent contexts
  const hasRootLevel = nodes.some((n) => n.parent === null);
  const hasNestedNodes = nodes.some((n) => n.parent !== null);

  // If all nodes are at the same level (all root or all same parent), no elevation needed
  if (!hasRootLevel || !hasNestedNodes) {
    return nodes;
  }

  // We have nodes at different levels - elevate nested nodes to their parent groups
  const result: PositionedNode[] = [];
  const seenNodes = new Set<PositionedNode>();

  for (const node of nodes) {
    if (node.parent === null) {
      // Root level node - keep as is (if not already added)
      if (!seenNodes.has(node)) {
        seenNodes.add(node);
        result.push(node);
      }
    } else {
      // Nested node - elevate to parent group (if not already added)
      if (!seenNodes.has(node.parent)) {
        seenNodes.add(node.parent);
        result.push(node.parent);
      }
    }
  }

  // Sort by start position to maintain correct order
  return result.sort((a, b) => a.start - b.start);
}

/**
 * Finds the single word the cursor is within (not just touching).
 */
function findWordAtCursor(positions: PositionedNode[], selection: SelectionBounds): PositionedNode | null {
  const words = positions.filter((p) => p.node.type === 'word' && selection.start >= p.start && selection.end <= p.end);

  return words[0] ?? null;
}

// ============================================================================
// AST MANIPULATION
// ============================================================================

/**
 * Replaces a node in the AST with replacement node(s).
 * Uses reference equality to find the target.
 */
function replaceNodeInAST(ast: ASTNode[], target: ASTNode, replacement: ASTNode | ASTNode[]): ASTNode[] {
  const replacements = Array.isArray(replacement) ? replacement : [replacement];

  return ast.flatMap((node) => {
    if (node === target) {
      return replacements;
    }

    if (node.type === 'group') {
      const newChildren = replaceNodeInAST(node.children, target, replacement);
      // Only create new object if children changed
      if (newChildren !== node.children) {
        return [{ ...node, children: newChildren }];
      }
    }

    return [node];
  });
}

// ============================================================================
// MAIN ADJUSTMENT FUNCTION
// ============================================================================

/**
 * Determines the adjustment strategy based on selection and AST structure.
 */
type AdjustmentStrategy =
  | { type: 'adjust-word'; node: PositionedNode }
  | { type: 'adjust-group'; node: PositionedNode }
  | { type: 'create-group'; nodes: PositionedNode[] }
  | { type: 'split-group'; group: PositionedNode; selectedChildren: PositionedNode[]; allPositions: PositionedNode[] }
  | { type: 'no-op' };

/**
 * Gets direct children of a group from the position map.
 */
function getGroupChildren(positions: PositionedNode[], group: PositionedNode): PositionedNode[] {
  return positions.filter((p) => p.parent === group);
}

/**
 * Checks if a positioned node is a content node (word, group, or embedding).
 */
function isContentNode(p: PositionedNode): boolean {
  return p.node.type === 'word' || p.node.type === 'group' || p.node.type === 'embedding';
}

function determineStrategy(positions: PositionedNode[], selection: SelectionBounds): AdjustmentStrategy {
  const contentNodes = findContentNodes(positions, selection);

  if (contentNodes.length === 0) {
    return { type: 'no-op' };
  }

  // Check if we're in a group context first
  const enclosingGroup = findEnclosingGroup(positions, selection);

  if (enclosingGroup) {
    // If touching group boundary, adjust the group
    if (isTouchingGroupBoundary(enclosingGroup, selection)) {
      return { type: 'adjust-group', node: enclosingGroup };
    }

    // Check for single word within the group
    const wordAtCursor = findWordAtCursor(positions, selection);
    if (wordAtCursor) {
      return { type: 'adjust-word', node: wordAtCursor };
    }

    // Selection spans content within group - adjust the group
    return { type: 'adjust-group', node: enclosingGroup };
  }

  // No enclosing group - check for single word
  const wordAtCursor = findWordAtCursor(positions, selection);
  if (wordAtCursor) {
    return { type: 'adjust-word', node: wordAtCursor };
  }

  // Single content node (could be word, embedding, or group)
  if (contentNodes.length === 1) {
    const node = contentNodes[0]!;
    if (node.node.type === 'group') {
      return { type: 'adjust-group', node };
    }
    if (node.node.type === 'word') {
      return { type: 'adjust-word', node };
    }
    // Embeddings don't support attention adjustment - wrap in group
    return { type: 'create-group', nodes: contentNodes };
  }

  // Multiple content nodes - check if selection crosses group boundaries
  // This happens when some nodes are inside a group and some are outside
  const hasRootLevel = contentNodes.some((n) => n.parent === null);
  const nestedNodes = contentNodes.filter((n) => n.parent !== null);

  if (hasRootLevel && nestedNodes.length > 0) {
    // Selection crosses group boundaries - need to split groups
    // Find all unique parent groups that are partially selected
    const parentGroups = new Set<PositionedNode>();
    for (const node of nestedNodes) {
      if (node.parent) {
        parentGroups.add(node.parent);
      }
    }

    // For each parent group, check if it's partially selected
    for (const group of parentGroups) {
      const groupChildren = getGroupChildren(positions, group).filter(isContentNode);
      const selectedChildren = groupChildren.filter((child) =>
        contentNodes.some((cn) => cn === child || (cn.start >= child.start && cn.end <= child.end))
      );

      // If not all children are selected, we need to split
      if (selectedChildren.length > 0 && selectedChildren.length < groupChildren.length) {
        return { type: 'split-group', group, selectedChildren, allPositions: positions };
      }
    }
  }

  // No partial group selections - create a new group
  return { type: 'create-group', nodes: contentNodes };
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
    if (!prompt.trim()) {
      return { prompt, selectionStart, selectionEnd };
    }

    const selection: SelectionBounds = {
      start: Math.min(selectionStart, selectionEnd),
      end: Math.max(selectionStart, selectionEnd),
    };

    const ast = parseTokens(tokenize(prompt));
    const { positions } = buildPositionMap(ast);
    const strategy = determineStrategy(positions, selection);

    switch (strategy.type) {
      case 'no-op':
        return { prompt, selectionStart, selectionEnd };

      case 'adjust-word': {
        const { node: pos } = strategy;
        const word = pos.node as ASTNode & { type: 'word' };
        const updated: ASTNode = { ...word, attention: computeAttention(direction, word.attention) };
        const newPrompt = serialize(replaceNodeInAST(ast, word, updated));
        return {
          prompt: newPrompt,
          selectionStart: pos.start,
          selectionEnd: pos.start + serialize([updated]).length,
        };
      }

      case 'adjust-group': {
        const { node: pos } = strategy;
        const group = pos.node as ASTNode & { type: 'group' };
        const newAttn = computeAttention(direction, group.attention);

        if (newAttn === undefined) {
          const newPrompt = serialize(replaceNodeInAST(ast, group, group.children));
          return {
            prompt: newPrompt,
            selectionStart: pos.start,
            selectionEnd: pos.start + serialize(group.children).length,
          };
        }

        const updated: ASTNode = { ...group, attention: newAttn };
        const newPrompt = serialize(replaceNodeInAST(ast, group, updated));
        return {
          prompt: newPrompt,
          selectionStart: pos.start,
          selectionEnd: pos.start + serialize([updated]).length,
        };
      }

      case 'split-group': {
        const { group: groupPos, selectedChildren, allPositions } = strategy;
        const group = groupPos.node as ASTNode & { type: 'group' };
        const groupAttn = group.attention;
        const selectedSet = new Set(selectedChildren.map((c) => c.node));

        // Rebuild group children with proper attention handling
        const rebuiltNodes: ASTNode[] = [];
        for (const childPos of getGroupChildren(allPositions, groupPos)) {
          if (!isContentNode(childPos)) {
            rebuiltNodes.push(childPos.node); // Preserve whitespace/punct
            continue;
          }

          const isSelected = selectedSet.has(childPos.node);
          if (!isSelected) {
            // Not selected: flatten group attention onto child
            rebuiltNodes.push(...applyAttentionToNode(childPos.node, groupAttn));
          } else {
            // Selected: adjust based on whether child has own attention
            const hasOwnAttn =
              childPos.node.type === 'word' || childPos.node.type === 'group'
                ? childPos.node.attention !== undefined
                : false;
            const effectiveAttn = hasOwnAttn
              ? (childPos.node as { attention?: Attention }).attention
              : combineAttention(groupAttn, undefined);

            rebuiltNodes.push(
              ...adjustContentNode(
                childPos.node.type === 'word' || childPos.node.type === 'group'
                  ? { ...childPos.node, attention: effectiveAttn }
                  : childPos.node,
                direction
              )
            );
          }
        }

        // Replace group and adjust root-level selected nodes
        let newAST = replaceNodeInAST(ast, group, rebuiltNodes);
        for (const rootNode of findContentNodes(allPositions, selection).filter((n) => n.parent === null)) {
          newAST = replaceNodeInAST(newAST, rootNode.node, adjustContentNode(rootNode.node, direction));
        }

        const newPrompt = serialize(newAST);
        const allSelected = [
          ...selectedChildren,
          ...findContentNodes(allPositions, selection).filter((n) => n.parent === null),
        ];
        const sorted = allSelected.sort((a, b) => a.start - b.start);

        return {
          prompt: newPrompt,
          selectionStart: sorted[0]?.start ?? selectionStart,
          selectionEnd: newPrompt.length - (prompt.length - (sorted[sorted.length - 1]?.end ?? selectionEnd)),
        };
      }

      case 'create-group': {
        const elevated = elevateToTopLevelNodes(strategy.nodes);
        const sorted = elevated.sort((a, b) => a.start - b.start);
        const wrapStart = sorted[0]!.start;
        const wrapEnd = sorted[sorted.length - 1]!.end;

        const selectedAST = parseTokens(tokenize(prompt.substring(wrapStart, wrapEnd)));
        const newGroup: ASTNode = {
          type: 'group',
          children: selectedAST,
          attention: computeAttention(direction, undefined),
        };

        const newPrompt = prompt.substring(0, wrapStart) + serialize([newGroup]) + prompt.substring(wrapEnd);
        return {
          prompt: newPrompt,
          selectionStart: wrapStart,
          selectionEnd: wrapStart + serialize([newGroup]).length,
        };
      }
    }
  } catch (error) {
    log.error({ error: serializeError(error as Error) }, 'Error adjusting prompt attention');
    return { prompt, selectionStart, selectionEnd };
  }
}
