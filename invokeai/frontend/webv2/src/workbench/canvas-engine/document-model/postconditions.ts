import type {
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasLayerSourceContract,
  CanvasNodeContract,
} from '@workbench/canvas-engine/contracts';
import type { LayerStackKind } from '@workbench/canvas-engine/document/layerStacks';
import type { CanvasLayerBasePatch, CanvasLayerConfigPatch } from '@workbench/canvas-engine/mutationContracts';

import { getDocumentIndex, getDocumentNode } from '@workbench/canvas-engine/document/documentIndex';
import { isGroupNode } from '@workbench/canvas-engine/document/documentTree';
import { isNodeHidden } from '@workbench/canvas-engine/document/layerEligibility';
import { getSiblingOrder } from '@workbench/canvas-engine/document/layerStacks';
import { GROUP_PATCH_KEYS } from '@workbench/canvas-engine/mutationContracts';

/** What a document must show after a prepared edit landed; evaluated against the reducer document. */
export type EditPostcondition =
  | { readonly kind: 'present'; readonly ids: readonly string[] }
  | { readonly kind: 'absent'; readonly ids: readonly string[] }
  | {
      readonly kind: 'sibling-order';
      readonly stack: LayerStackKind;
      readonly parentId: string | null;
      readonly orderedIds: readonly string[];
    }
  | { readonly kind: 'selection'; readonly id: string | null }
  | { readonly kind: 'patched'; readonly id: string; readonly patch: CanvasLayerBasePatch }
  | { readonly kind: 'config'; readonly id: string; readonly config: CanvasLayerConfigPatch }
  | { readonly kind: 'source'; readonly id: string; readonly source: CanvasLayerSourceContract }
  | { readonly kind: 'hidden'; readonly id: string; readonly isHidden: boolean };

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null;

/** Structural equality over the plain data a layer contract holds. */
export const sameValue = (left: unknown, right: unknown): boolean => {
  if (Object.is(left, right)) {
    return true;
  }
  if (Array.isArray(left) || Array.isArray(right)) {
    return (
      Array.isArray(left) &&
      Array.isArray(right) &&
      left.length === right.length &&
      left.every((item, index) => sameValue(item, right[index]))
    );
  }
  if (!isRecord(left) || !isRecord(right)) {
    return false;
  }
  const keys = Object.keys(left);
  return keys.length === Object.keys(right).length && keys.every((key) => sameValue(left[key], right[key]));
};

/** Whether every field named by `patch` already holds its value on `node`. */
export const isPatchApplied = (node: CanvasNodeContract, patch: CanvasLayerBasePatch): boolean =>
  (Object.keys(patch) as (keyof CanvasLayerBasePatch)[]).every((key) => {
    if (isGroupNode(node)) {
      return (
        GROUP_PATCH_KEYS.includes(key) &&
        node[key as 'name' | 'isEnabled' | 'isLocked' | 'opacity' | 'blendMode' | 'colorLabel'] === patch[key]
      );
    }
    return key === 'transform'
      ? (Object.keys(patch.transform ?? {}) as (keyof CanvasLayerContract['transform'])[]).every(
          (axis) => node.transform[axis] === patch.transform?.[axis]
        )
      : node[key] === patch[key];
  });

/**
 * Whether every field named by `config` already holds its value; nested
 * partials compare field by field, and a `null` config value asserts the field
 * is cleared: absent (the reducer deletes mask modifiers) or stored as null (prompts).
 */
export const isConfigApplied = (node: CanvasNodeContract, config: CanvasLayerConfigPatch): boolean => {
  if (node.type !== config.layerType) {
    return false;
  }
  const target = node as unknown as Record<string, unknown>;
  return Object.entries(config).every(([key, value]) => {
    if (key === 'layerType') {
      return true;
    }
    const current = target[key];
    if (value === null) {
      return current === undefined || current === null;
    }
    return isRecord(value) && isRecord(current) && (key === 'adapter' || key === 'mask')
      ? Object.entries(value).every(([field, expected]) => sameValue(current[field], expected))
      : sameValue(current, value);
  });
};

const sameOrder = (left: readonly string[], right: readonly string[]): boolean =>
  left.length === right.length && left.every((id, index) => id === right[index]);

export const checkEditPostconditions = (
  document: CanvasDocumentContractV3,
  postconditions: readonly EditPostcondition[]
): boolean => {
  const index = getDocumentIndex(document);
  return postconditions.every((postcondition) => {
    switch (postcondition.kind) {
      case 'present':
        return postcondition.ids.every((id) => index.byId.has(id));
      case 'absent':
        return postcondition.ids.every((id) => !index.byId.has(id));
      case 'sibling-order':
        return sameOrder(
          getSiblingOrder(document.stacks, postcondition.stack, postcondition.parentId).orderedIds,
          postcondition.orderedIds
        );
      case 'selection':
        return document.selectedLayerId === postcondition.id;
      case 'patched': {
        const node = getDocumentNode(document, postcondition.id);
        return node !== null && isPatchApplied(node, postcondition.patch);
      }
      case 'config': {
        const node = getDocumentNode(document, postcondition.id);
        return node !== null && isConfigApplied(node, postcondition.config);
      }
      case 'source': {
        const node = getDocumentNode(document, postcondition.id);
        return node !== null && 'source' in node && node.source === postcondition.source;
      }
      case 'hidden': {
        const node = getDocumentNode(document, postcondition.id);
        return node !== null && isNodeHidden(node) === postcondition.isHidden;
      }
    }
  });
};
