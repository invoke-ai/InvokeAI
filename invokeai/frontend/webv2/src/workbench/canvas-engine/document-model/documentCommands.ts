import type {
  CanvasLayerContract,
  CanvasLayerSourceContract,
  CanvasNodeContract,
} from '@workbench/canvas-engine/contracts';
import type {
  LayerStackKind,
  LayerStackMoveKind,
  ReorderSiblingsCommand,
} from '@workbench/canvas-engine/document/layerStacks';
import type {
  CanvasLayerBasePatch,
  CanvasLayerConfigPatch,
  CanvasProjectMutation,
} from '@workbench/canvas-engine/mutationContracts';

import type { EditPostcondition } from './postconditions';

export type DocumentCommand =
  | {
      readonly type: 'insert';
      readonly nodes: readonly CanvasNodeContract[];
      /** Land above this node when it belongs to the inserted node's stack; otherwise at the stack top. */
      readonly aboveId: string | null;
      /** Land at the top of this group instead, when it belongs to the inserted node's stack. */
      readonly insideId?: string | null;
      /** The primary selection afterwards; defaults to the last inserted node. */
      readonly selectId?: string | null;
      /** The stack an empty group joins; leaves and non-empty groups carry their own. */
      readonly stack?: LayerStackKind;
    }
  | { readonly type: 'remove'; readonly ids: readonly string[] }
  | { readonly type: 'duplicate'; readonly ids: readonly string[]; readonly createId: () => string }
  | { readonly type: 'move'; readonly ids: readonly string[]; readonly kind: LayerStackMoveKind }
  | { readonly type: 'reorder'; readonly orders: readonly ReorderSiblingsCommand[] }
  | {
      /** Moves subtrees under `parentId` (`null` for the stack root) directly above `beforeId`, or to the bottom. */
      readonly type: 'reparent';
      readonly ids: readonly string[];
      readonly parentId: string | null;
      readonly beforeId: string | null;
    }
  | { readonly type: 'group'; readonly ids: readonly string[]; readonly groupId: string; readonly name: string }
  | { readonly type: 'ungroup'; readonly ids: readonly string[] }
  | {
      readonly type: 'patch';
      readonly id: string;
      readonly patch: CanvasLayerBasePatch;
      /** The values before a previewed gesture; the inverse restores these instead of the current ones. */
      readonly before?: CanvasLayerBasePatch;
    }
  | {
      readonly type: 'patch-config';
      readonly id: string;
      readonly config: CanvasLayerConfigPatch;
      readonly before?: CanvasLayerConfigPatch;
    }
  | {
      /** One atomic edit patching several layers' config, e.g. moving a modifier between two of them. */
      readonly type: 'patch-config-batch';
      readonly patches: readonly {
        readonly id: string;
        readonly config: CanvasLayerConfigPatch;
        readonly before?: CanvasLayerConfigPatch;
      }[];
    }
  | { readonly type: 'patch-source'; readonly id: string; readonly source: CanvasLayerSourceContract }
  | { readonly type: 'set-enabled'; readonly updates: readonly { id: string; isEnabled: boolean }[] }
  | { readonly type: 'set-hidden'; readonly updates: readonly { id: string; isHidden: boolean }[] }
  | { readonly type: 'set-locked'; readonly updates: readonly { id: string; isLocked: boolean }[] }
  | { readonly type: 'translate'; readonly ids: readonly string[]; readonly dx: number; readonly dy: number }
  | { readonly type: 'select'; readonly id: string | null };

export type DocumentRefusal =
  | { readonly status: 'missing'; readonly ids: readonly string[] }
  | { readonly status: 'locked'; readonly ids: readonly string[] }
  | {
      readonly status: 'wrong-type';
      readonly expected: readonly (CanvasLayerContract['type'] | 'group')[];
      readonly actual: string;
    }
  | { readonly status: 'invalid-target'; readonly targetId: string; readonly reason: InvalidTargetReason }
  | { readonly status: 'unsupported'; readonly operation: string };

export type InvalidTargetReason =
  | 'id-exists'
  | 'foreign-stack'
  | 'not-siblings'
  | 'not-a-group'
  | 'cycle'
  | 'depth-exceeded'
  | 'node-limit'
  | 'no-layer-below'
  | 'not-mergeable';

export type EditHistoryPolicy = 'record' | 'none';

/** A document edit ready for the transaction module: what to dispatch, what must hold afterwards, how to record it. */
export interface PreparedDocumentEdit {
  readonly forward: CanvasProjectMutation;
  readonly inverse: CanvasProjectMutation;
  readonly projectId: string;
  readonly expectedRevision: number;
  readonly selectionBefore: string | null;
  readonly selectionAfter: string | null;
  /** Nodes the edit changes, including neighbours whose position it displaces. */
  readonly touchedIds: readonly string[];
  /** Ids the edit brings into the document. */
  readonly createdIds: readonly string[];
  readonly touchedStacks: readonly LayerStackKind[];
  readonly postconditions: readonly EditPostcondition[];
  /** Document commands never need prepared pixels; pixel-bearing edits keep their own controllers. */
  readonly rasterWork: null;
  readonly history: EditHistoryPolicy;
}

export type PrepareEditResult =
  | { readonly status: 'prepared'; readonly edit: PreparedDocumentEdit }
  /** The command describes the document as it already is; there is nothing to dispatch. */
  | { readonly status: 'unchanged' }
  | DocumentRefusal;

export type MergeDownEligibility =
  | { readonly status: 'eligible'; readonly upperId: string; readonly lowerId: string }
  | DocumentRefusal;
