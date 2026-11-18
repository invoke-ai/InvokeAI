import type {
  CanvasCoreStoreCapability,
  CanvasDocumentCapability,
  CanvasDocumentContractV3,
  CanvasDocumentModel,
  CanvasEngineLayerCapability,
  DocumentCommand,
  StructuralCommitResult,
} from '@workbench/canvas-engine/api';
import type { TFunction } from 'i18next';

import { getDocumentIndex, isGroupNode, outermostNodes } from '@workbench/canvas-engine/api';
import { publishLayerPanelSelection, readLayerPanelState, setLayerGroupExpanded } from '@workbench/layerPanelState';
import { useNotify } from '@workbench/useNotify';
import { useCanvasDocumentRead } from '@workbench/widgets/canvas/engineStoreHooks';
import { reportLayerOperation, reportStructuralCommit } from '@workbench/widgets/canvas/useStructuralCommit';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { createLayerId, nextGroupName } from './layerOps';

/**
 * The structural commands every layer surface offers: the Canvas hotkeys, the panel footer, the row and group menus,
 * and the tree's Delete key. Target rule, eligibility, undo label and feedback live here; a surface maps its trigger
 * to one of these and keeps only its own concerns (placement, copy, focus).
 */
export type LayerCommand = 'delete' | 'duplicate' | 'group' | 'ungroup';

/**
 * `committed`: the edit landed. `skipped`: the model refused it or it would change nothing (the command was
 * unavailable, as a disabled menu item is), or another edit holds the canvas; nothing is said. `failed`: an eligible
 * edit did not land, and the reporter has been told why.
 */
export type LayerCommandOutcome = 'committed' | 'failed' | 'skipped';

export interface LayerCommandEngine {
  readonly document: Pick<CanvasDocumentCapability, 'model'>;
  readonly layers: Pick<CanvasEngineLayerCapability, 'commitPrepared' | 'duplicateLayers' | 'endStructuralPreview'>;
  readonly projectId: string;
}

export interface LayerCommandDeps {
  readonly engine: LayerCommandEngine | null;
  readonly reportError: (title: string, message: string) => void;
  readonly t: TFunction;
}

/** The nodes a command on `subjectId` acts on: the whole selection when the subject is part of it, else itself. */
export const layerCommandTargets = (selectedIds: readonly string[], subjectId: string): readonly string[] =>
  selectedIds.includes(subjectId) ? selectedIds : [subjectId];

/**
 * The reparent that lifts `ids` out of their shared parent group to sit directly above it, in document order; null
 * unless every id is a leaf of the same group.
 */
export const removeFromGroupCommand = (
  document: CanvasDocumentContractV3,
  ids: readonly string[]
): Extract<DocumentCommand, { type: 'reparent' }> | null => {
  const index = getDocumentIndex(document);
  const entries = ids.flatMap((id) => index.byId.get(id) ?? []);
  const groupId = entries[0]?.parentId ?? null;
  if (
    groupId === null ||
    entries.length !== ids.length ||
    entries.some((entry) => isGroupNode(entry.node) || entry.parentId !== groupId)
  ) {
    return null;
  }
  return {
    beforeId: groupId,
    ids: [...entries].sort((a, b) => a.order - b.order).map((entry) => entry.node.id),
    parentId: index.byId.get(groupId)!.parentId,
    type: 'reparent',
  };
};

/** A group id no document holds, for asking the model whether a grouping would be accepted. */
const PROBE_GROUP_ID = '\0group';

/** The model decides a duplicate before it mints a copy's id, so asking it needs none. */
const PROBE_CREATE_ID = (): string => '\0copy';

/**
 * Ungroup dissolves the groups among `ids`; with none among them, leaves of one group move out above it instead.
 * Null when neither applies.
 */
const ungroupCommand = (model: CanvasDocumentModel, ids: readonly string[]): DocumentCommand | null => {
  const groupIds = ids.filter((id) => {
    const node = model.getNode(id);
    return !!node && isGroupNode(node);
  });
  return groupIds.length > 0 ? { ids: groupIds, type: 'ungroup' } : removeFromGroupCommand(model.document, ids);
};

const accepts = (model: CanvasDocumentModel, command: DocumentCommand | null): boolean =>
  !!command && model.refusalFor(command) === null;

/** Whether the document model accepts `command` on `ids`: the rule every surface enables its control by. */
export const canRunLayerCommand = (
  model: CanvasDocumentModel | null,
  command: LayerCommand,
  ids: readonly string[]
): boolean => {
  if (!model || ids.length === 0) {
    return false;
  }
  switch (command) {
    case 'delete':
      return accepts(model, { ids, type: 'remove' });
    case 'duplicate':
      return accepts(model, { createId: PROBE_CREATE_ID, ids, type: 'duplicate' });
    case 'group':
      return accepts(model, { groupId: PROBE_GROUP_ID, ids, name: '', type: 'group' });
    case 'ungroup':
      return accepts(model, ungroupCommand(model, ids));
  }
};

/**
 * How many nodes a command acts on, for its label: a node selected together with a group around it is part of that
 * group's subtree, so a group and its child count as one.
 */
const targetCount = (model: CanvasDocumentModel, ids: readonly string[]): number =>
  outermostNodes(getDocumentIndex(model.document), ids).length;

const LABEL_KEYS = {
  delete: { count: 'widgets.layers.actions.deleteCount', group: 'widgets.layers.actions.deleteGroup' },
  duplicate: { count: 'widgets.layers.actions.duplicateCount', group: 'widgets.layers.actions.duplicateGroup' },
} as const;

/**
 * The label a delete or duplicate of `ids` carries in menus and in history, so the two cannot drift: one group alone
 * is named as a group, anything else by its outermost count.
 */
export const layerCommandLabel = (
  document: CanvasDocumentContractV3,
  command: 'delete' | 'duplicate',
  ids: readonly string[]
): { readonly count: number; readonly key: string } => {
  const outer = outermostNodes(getDocumentIndex(document), ids);
  const keys = LABEL_KEYS[command];
  return outer.length === 1 && isGroupNode(outer[0]!.node)
    ? { count: 1, key: keys.group }
    : { count: outer.length, key: keys.count };
};

const translatedLabel = (
  t: TFunction,
  model: CanvasDocumentModel,
  command: 'delete' | 'duplicate',
  ids: readonly string[]
): string => {
  const { count, key } = layerCommandLabel(model.document, command, ids);
  return t(key, { count });
};

/** Busy means another edit holds the canvas, which already shows; anything else that did not land is reported. */
const settle = (deps: LayerCommandDeps, result: StructuralCommitResult): LayerCommandOutcome => {
  if (result.status === 'committed') {
    return 'committed';
  }
  if (result.status === 'busy') {
    return 'skipped';
  }
  reportStructuralCommit(result, deps.reportError, deps.t);
  return 'failed';
};

/**
 * The engine and the committed document's model, once the model accepts `command` on `ids` as it stands. Only then
 * does an open preview gesture end, so the edit and its inverse capture the committed document rather than previewed
 * values; a refused command leaves the gesture alone. Null when the command is unavailable.
 */
const committedModel = (
  engine: LayerCommandEngine | null,
  command: LayerCommand,
  ids: readonly string[]
): { engine: LayerCommandEngine; model: CanvasDocumentModel } | null => {
  if (!engine || !canRunLayerCommand(engine.document.model(), command, ids)) {
    return null;
  }
  engine.layers.endStructuralPreview();
  const model = engine.document.model();
  return model ? { engine, model } : null;
};

/** Prepares the command `plan` names on the committed document and commits it as one undo step under its label. */
const commitCommand = (
  deps: LayerCommandDeps,
  command: LayerCommand,
  ids: readonly string[],
  plan: (model: CanvasDocumentModel) => { command: DocumentCommand; label: string } | null
): LayerCommandOutcome => {
  const live = committedModel(deps.engine, command, ids);
  const planned = live ? plan(live.model) : null;
  const result = live && planned ? live.model.prepare(planned.command) : null;
  return live && planned && result?.status === 'prepared'
    ? settle(deps, live.engine.layers.commitPrepared(planned.label, result.edit))
    : 'skipped';
};

/** Removes `ids` and everything inside them as one undo step. */
export const deleteLayers = (deps: LayerCommandDeps, ids: readonly string[]): LayerCommandOutcome =>
  commitCommand(deps, 'delete', ids, (model) => ({
    command: { ids, type: 'remove' },
    label: translatedLabel(deps.t, model, 'delete', ids),
  }));

/** Dissolves the groups among `ids`, or lifts leaves of one group out above it, as one undo step. */
export const ungroupLayers = (deps: LayerCommandDeps, ids: readonly string[]): LayerCommandOutcome =>
  commitCommand(deps, 'ungroup', ids, (model) => {
    const command = ungroupCommand(model, ids);
    if (!command) {
      return null;
    }
    const labelKey =
      command.type === 'reparent' ? 'widgets.layers.actions.removeFromGroup' : 'widgets.layers.actions.ungroup';
    return { command, label: deps.t(labelKey) };
  });

/**
 * Wraps `ids` in a new group, selects it, and shows its children. The selection is published only for an edit the
 * model accepted, before the commit so reconciliation keeps it, and rolled back if the commit does not land.
 */
export const groupLayers = (deps: LayerCommandDeps, ids: readonly string[]): LayerCommandOutcome => {
  const live = committedModel(deps.engine, 'group', ids);
  if (!live) {
    return 'skipped';
  }
  const { engine, model } = live;
  const { projectId } = engine;
  const groupId = createLayerId();
  const names = getDocumentIndex(model.document).nodes.map((entry) => entry.node.name);
  const result = model.prepare({ groupId, ids, name: nextGroupName(names), type: 'group' });
  if (result.status !== 'prepared') {
    return 'skipped';
  }
  const previous = readLayerPanelState(projectId, model.document.selectedLayerId);
  setLayerGroupExpanded(projectId, model.document.selectedLayerId, [groupId], true);
  publishLayerPanelSelection({ primaryId: groupId, projectId, selectedIds: [groupId] });
  const outcome = engine.layers.commitPrepared(
    deps.t('widgets.layers.actions.groupCount', { count: targetCount(model, ids) }),
    result.edit
  );
  if (outcome.status !== 'committed') {
    publishLayerPanelSelection({
      anchorId: previous.anchorId,
      primaryId: previous.primaryId,
      projectId,
      selectedIds: previous.selectedIds,
    });
    setLayerGroupExpanded(projectId, previous.primaryId, [groupId], false);
  }
  return settle(deps, outcome);
};

/**
 * Copies `ids` with their pixels as one undo step; the engine selects the copies in the panel as it publishes them.
 * The model is asked first so a refused duplicate stays a silent no-op, as a disabled control would; the engine asks
 * again when it commits.
 */
export const duplicateLayers = async (deps: LayerCommandDeps, ids: readonly string[]): Promise<LayerCommandOutcome> => {
  const { engine, reportError, t } = deps;
  const model = engine?.document.model() ?? null;
  if (!engine || !model || !canRunLayerCommand(model, 'duplicate', ids)) {
    return 'skipped';
  }
  try {
    const result = await engine.layers.duplicateLayers(translatedLabel(t, model, 'duplicate', ids), ids);
    switch (result.status) {
      case 'duplicated':
        return 'committed';
      case 'busy':
        return 'skipped';
      case 'nothing':
        break;
      default:
        reportLayerOperation(result.status === 'unreadable' ? result : result.status, reportError, t);
        return 'failed';
    }
  } catch {
    // The engine leaves the document unchanged on a rejected transaction.
  }
  reportError(t('widgets.layers.actions.actionFailed'), t('widgets.layers.actions.copyFailed'));
  return 'failed';
};

type LayerCommandReadEngine = CanvasCoreStoreCapability & {
  readonly document: Pick<CanvasDocumentCapability, 'model'>;
};

const LAYER_COMMANDS: readonly LayerCommand[] = ['delete', 'duplicate', 'group', 'ungroup'];

/**
 * Which commands the model accepts on `ids`, read live: a lock, a removal or an undo from elsewhere updates it even
 * when nothing the caller passes changed. One read, re-run only when the document changes (never on a pixel
 * publish), answers with the accepted commands' names joined, a string that keeps its snapshot while unchanged.
 */
export const useLayerCommandAvailability = (
  engine: LayerCommandReadEngine | null,
  ids: readonly string[]
): Readonly<Record<LayerCommand, boolean>> => {
  const accepted = useCanvasDocumentRead(engine, () => {
    const model = engine?.document.model() ?? null;
    return LAYER_COMMANDS.filter((command) => canRunLayerCommand(model, command, ids)).join(' ');
  });
  return useMemo(() => {
    const names = accepted.split(' ');
    return {
      delete: names.includes('delete'),
      duplicate: names.includes('duplicate'),
      group: names.includes('group'),
      ungroup: names.includes('ungroup'),
    };
  }, [accepted]);
};

/** The commands' dependencies for a panel surface: its engine, the app's error notices and the active language. */
export const useLayerCommandDeps = (engine: LayerCommandEngine | null): LayerCommandDeps => {
  const notify = useNotify();
  const { t } = useTranslation();
  return useMemo(() => ({ engine, reportError: notify.error, t }), [engine, notify.error, t]);
};
