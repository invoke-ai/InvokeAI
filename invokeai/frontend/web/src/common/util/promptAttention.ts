import { logger } from 'app/logging/logger';
import { serializeError } from 'serialize-error';

import { type ASTNode, type Attention, parseTokens, serialize, tokenize } from './promptAST';

const log = logger('events');

/**
 * Behavior Rules:
 *
 * ATTENTION SYNTAX:
 * - Words support +/- attention: `word+`, `word-`, `word++`, `word--`, etc.
 * - Groups support both +/- and numeric: `(words)+`, `(words)1.2`, `(words)0.8`
 * - `word++` is roughly equivalent to `(word)1.2` in effect
 * - Mixed attention like `word+-` or numeric on words `word1.2` is invalid
 *
 * ADJUSTMENT RULES:
 * - `word++` down → `word+`
 * - `word+` down → `word` (attention removed)
 * - `word` down → `word-`
 * - `word-` down → `word--`
 * - `(content)1.2` down → `(content)1.1`
 * - `(content)1.1` down → `content` (group unwrapped)
 * - `content content` down → `(content content)0.9` (new group created)
 *
 * SELECTION BEHAVIOR:
 * - Cursor/selection within a word → expand to full word, adjust word attention
 * - Cursor touching group boundary (parens or weight) → adjust group attention
 * - Selection entirely within a group → adjust that group's attention
 * - Selection spans multiple content nodes → create new group with initial attention
 * - Whitespace and punctuation are ignored for content selection
 */

type AttentionDirection = 'increment' | 'decrement';

type SelectionBounds = {
  start: number;
  end: number;
};

type AdjustmentResult = {
  prompt: string;
  selectionStart: number;
  selectionEnd: number;
};

/**
 * A node with its position in the serialized prompt string.
 */
type PositionedNode = {
  node: ASTNode;
  start: number;
  end: number;
  /** Position of content start (after opening paren for groups) */
  contentStart: number;
  /** Position of content end (before closing paren for groups) */
  contentEnd: number;
  parent: PositionedNode | null;
};

// ============================================================================
// ATTENTION COMPUTATION
// ============================================================================

/**
 * Checks if attention is symbol-based (+, -, ++, etc.)
 */
function isSymbolAttention(attention: Attention | undefined): attention is string {
  return typeof attention === 'string' && /^[+-]+$/.test(attention);
}

/**
 * Checks if attention is numeric (1.2, 0.8, etc.)
 */
function isNumericAttention(attention: Attention | undefined): attention is number {
  return typeof attention === 'number';
}

/**
 * Computes adjusted attention for symbol-based attention (+/-).
 * Used for both words and groups with symbol attention.
 */
function adjustSymbolAttention(direction: AttentionDirection, attention: string | undefined): string | undefined {
  if (!attention) {
    return direction === 'increment' ? '+' : '-';
  }

  if (direction === 'increment') {
    // Going up: remove '-' if present, otherwise add '+'
    if (attention.endsWith('-')) {
      const result = attention.slice(0, -1);
      return result || undefined;
    }
    return `${attention}+`;
  } else {
    // Going down: remove '+' if present, otherwise add '-'
    if (attention.endsWith('+')) {
      const result = attention.slice(0, -1);
      return result || undefined;
    }
    return `${attention}-`;
  }
}

/**
 * Computes adjusted attention for numeric attention.
 * Only used for groups.
 */
function adjustNumericAttention(direction: AttentionDirection, attention: number): number | undefined {
  const step = direction === 'increment' ? 0.1 : -0.1;
  const result = parseFloat((attention + step).toFixed(1));

  // 1.0 is default - return undefined to signal unwrapping
  if (result === 1.0) {
    return undefined;
  }

  return result;
}

/**
 * Computes the new attention value based on direction and current attention.
 * Returns undefined if attention should be removed (normalize to default).
 */
function computeAttention(
  direction: AttentionDirection,
  attention: Attention | undefined,
  _isGroup: boolean
): Attention | undefined {
  // No current attention
  if (attention === undefined) {
    return direction === 'increment' ? '+' : '-';
  }

  // Symbol attention (+, -, ++, etc.)
  if (isSymbolAttention(attention)) {
    return adjustSymbolAttention(direction, attention);
  }

  // Numeric attention (only valid for groups)
  if (isNumericAttention(attention)) {
    return adjustNumericAttention(direction, attention);
  }

  // Parse string numbers
  const numValue = parseFloat(String(attention));
  if (!isNaN(numValue)) {
    return adjustNumericAttention(direction, numValue);
  }

  // Fallback: treat as no attention
  return direction === 'increment' ? '+' : '-';
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
  | { type: 'no-op' };

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

  // Multiple content nodes - create a new group
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
    // Handle empty prompt
    if (!prompt.trim()) {
      return { prompt, selectionStart, selectionEnd };
    }

    // Normalize selection
    const selection: SelectionBounds = {
      start: Math.min(selectionStart, selectionEnd),
      end: Math.max(selectionStart, selectionEnd),
    };

    // Parse and build position map
    const tokens = tokenize(prompt);
    const ast = parseTokens(tokens);
    const { positions } = buildPositionMap(ast);

    // Determine what to do
    const strategy = determineStrategy(positions, selection);

    switch (strategy.type) {
      case 'no-op':
        return { prompt, selectionStart, selectionEnd };

      case 'adjust-word': {
        const wordPos = strategy.node;
        const word = wordPos.node as ASTNode & { type: 'word' };
        const newAttention = computeAttention(direction, word.attention, false);

        const updatedWord: ASTNode = {
          type: 'word',
          text: word.text,
          attention: newAttention,
        };

        const newAST = replaceNodeInAST(ast, word, updatedWord);
        const newPrompt = serialize(newAST);
        const newWordText = serialize([updatedWord]);

        return {
          prompt: newPrompt,
          selectionStart: wordPos.start,
          selectionEnd: wordPos.start + newWordText.length,
        };
      }

      case 'adjust-group': {
        const groupPos = strategy.node;
        const group = groupPos.node as ASTNode & { type: 'group' };
        const newAttention = computeAttention(direction, group.attention, true);

        // If attention becomes undefined (1.0), unwrap the group
        if (newAttention === undefined) {
          const newAST = replaceNodeInAST(ast, group, group.children);
          const newPrompt = serialize(newAST);
          const childrenText = serialize(group.children);

          return {
            prompt: newPrompt,
            selectionStart: groupPos.start,
            selectionEnd: groupPos.start + childrenText.length,
          };
        }

        const updatedGroup: ASTNode = {
          type: 'group',
          children: group.children,
          attention: newAttention,
        };

        const newAST = replaceNodeInAST(ast, group, updatedGroup);
        const newPrompt = serialize(newAST);
        const newGroupText = serialize([updatedGroup]);

        return {
          prompt: newPrompt,
          selectionStart: groupPos.start,
          selectionEnd: groupPos.start + newGroupText.length,
        };
      }

      case 'create-group': {
        // Elevate any nested nodes to their parent groups when selection crosses boundaries
        const elevatedNodes = elevateToTopLevelNodes(strategy.nodes);
        const sortedNodes = elevatedNodes.sort((a, b) => a.start - b.start);
        const firstNode = sortedNodes[0]!;
        const lastNode = sortedNodes[sortedNodes.length - 1]!;

        // Get the text range to wrap
        const wrapStart = firstNode.start;
        const wrapEnd = lastNode.end;

        // Parse just the selected portion
        const selectedText = prompt.substring(wrapStart, wrapEnd);
        const selectedTokens = tokenize(selectedText);
        const selectedAST = parseTokens(selectedTokens);

        // Create new group with appropriate attention
        const newAttention = computeAttention(direction, undefined, true);
        const newGroup: ASTNode = {
          type: 'group',
          children: selectedAST,
          attention: newAttention,
        };

        // Reconstruct prompt
        const before = prompt.substring(0, wrapStart);
        const after = prompt.substring(wrapEnd);
        const newGroupText = serialize([newGroup]);
        const newPrompt = before + newGroupText + after;

        return {
          prompt: newPrompt,
          selectionStart: wrapStart,
          selectionEnd: wrapStart + newGroupText.length,
        };
      }
    }
  } catch (error) {
    log.error({ error: serializeError(error as Error) }, 'Error adjusting prompt attention');
    return { prompt, selectionStart, selectionEnd };
  }
}
