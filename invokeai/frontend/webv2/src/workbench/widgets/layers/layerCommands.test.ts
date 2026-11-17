import type {
  CanvasDocumentContractV3,
  CanvasNodeContract,
  DuplicateLayersResult,
  StructuralCommitResult,
} from '@workbench/canvas-engine/api';
import type { Project } from '@workbench/projectContracts';

import { createDocumentModel, getDocumentIndex } from '@workbench/canvas-engine/api';
import {
  groupContract as group,
  layerContract as layer,
  stacksFrom,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createEmptyCanvasDocument } from '@workbench/canvasMigration';
import { applyCanvasProjectMutation } from '@workbench/canvasProjectMutations';
import { clearLayerPanelStates, publishLayerPanelSelection, readLayerPanelState } from '@workbench/layerPanelState';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createInstance } from 'i18next';
import { afterEach, describe, expect, it } from 'vitest';

import {
  canRunLayerCommand,
  deleteLayers,
  duplicateLayers,
  groupLayers,
  layerCommandTargets,
  ungroupLayers,
  type LayerCommand,
  type LayerCommandDeps,
  type LayerCommandEngine,
  type LayerCommandOutcome,
} from './layerCommands';

const englishCatalog = Object.values(
  import.meta.glob('../../../../public/locales/en.json', { eager: true, import: 'default' })
)[0] as Record<string, unknown>;
const i18n = createInstance();
await i18n.init({ initAsync: false, lng: 'en', resources: { en: { translation: englishCatalog } } });

afterEach(() => clearLayerPanelStates());

/** Stacks top first, each node in preorder, indented by depth. */
const outline = (document: CanvasDocumentContractV3): string[] =>
  getDocumentIndex(document).nodes.map((entry) => `${'  '.repeat(entry.path.length)}${entry.node.id}`);

/**
 * The commands over a real document model whose commits apply through the real reducer. The engine's pixel copy is
 * the one double: it records its label and answers `duplicateResult`.
 */
const setup = (nodes: CanvasNodeContract[]) => {
  let project: Project = applyCanvasProjectMutation(createInitialWorkbenchState().projects[0]!, {
    document: { ...createEmptyCanvasDocument(), selectedLayerId: 'r1', stacks: stacksFrom(nodes) },
    type: 'replaceCanvasDocument',
  });
  const labels: string[] = [];
  const errors: [string, string][] = [];
  const control: {
    commitResult: StructuralCommitResult | null;
    duplicateResult: () => Promise<DuplicateLayersResult>;
    previewEnds: number;
  } = {
    commitResult: null,
    previewEnds: 0,
    duplicateResult: () => Promise.resolve({ duplicateIds: ['copy'], selectedLayerId: 'copy', status: 'duplicated' }),
  };
  const engine: LayerCommandEngine = {
    document: { model: () => createDocumentModel(project.canvas.document, { editRevision: 0, projectId: project.id }) },
    layers: {
      commitPrepared: (label, edit) => {
        if (control.commitResult) {
          return control.commitResult;
        }
        labels.push(label);
        project = applyCanvasProjectMutation(project, edit.forward);
        return { status: 'committed' };
      },
      duplicateLayers: (label) => {
        labels.push(label);
        return control.duplicateResult();
      },
      endStructuralPreview: () => {
        control.previewEnds += 1;
      },
    },
    projectId: project.id,
  };
  const deps: LayerCommandDeps = {
    engine,
    reportError: (title, message) => errors.push([title, message]),
    t: i18n.t,
  };
  const run = (command: LayerCommand, ids: readonly string[]): Promise<LayerCommandOutcome> | LayerCommandOutcome => {
    switch (command) {
      case 'delete':
        return deleteLayers(deps, ids);
      case 'duplicate':
        return duplicateLayers(deps, ids);
      case 'group':
        return groupLayers(deps, ids);
      case 'ungroup':
        return ungroupLayers(deps, ids);
    }
  };
  return {
    can: (command: LayerCommand, ids: readonly string[]) => canRunLayerCommand(engine.document.model(), command, ids),
    control,
    deps,
    document: () => project.canvas.document,
    errors,
    labels,
    panel: () => readLayerPanelState(project.id, project.canvas.document.selectedLayerId),
    projectId: project.id,
    run,
  };
};

/** raster: r1, G[r2, r3], r4 */
const tree = (locks: { G?: boolean; r1?: boolean; r3?: boolean } = {}): CanvasNodeContract[] => [
  layer('r1', 'raster', { isLocked: locks.r1 ?? false }),
  group('G', [layer('r2'), layer('r3', 'raster', { isLocked: locks.r3 ?? false })], { isLocked: locks.G ?? false }),
  layer('r4'),
];

describe('layerCommandTargets', () => {
  it('acts on the whole selection when the subject is part of it, else on the subject alone', () => {
    expect(layerCommandTargets(['r1', 'r4'], 'r4')).toEqual(['r1', 'r4']);
    expect(layerCommandTargets(['r1', 'r4'], 'r2')).toEqual(['r2']);
    expect(layerCommandTargets([], 'r2')).toEqual(['r2']);
  });
});

describe('eligibility', () => {
  it.each<[LayerCommand, readonly string[], readonly string[]]>([
    ['delete', ['r2', 'r3'], ['r1', 'r2']],
    ['group', ['r2', 'r3'], ['r1', 'r4']],
    ['ungroup', ['r3'], ['r2']],
  ])(
    '%s refuses a selection holding a locked layer, quietly and without touching anything',
    async (command, locked, free) => {
      const h = setup(tree({ r3: true }));
      const before = h.document();

      expect(h.can(command, free)).toBe(true);
      expect(h.can(command, locked)).toBe(false);
      expect(await h.run(command, locked)).toBe('skipped');

      expect(h.document()).toBe(before);
      expect(h.labels).toEqual([]);
      expect(h.errors).toEqual([]);
    }
  );

  it.each<[LayerCommand, readonly string[]]>([
    ['delete', ['G']],
    ['delete', ['r2']],
    ['group', ['r2']],
    ['ungroup', ['G']],
    ['ungroup', ['r2']],
  ])('%s %j refuses on or inside a locked group', async (command, ids) => {
    const h = setup(tree({ G: true }));

    expect(h.can(command, ids)).toBe(false);
    expect(await h.run(command, ids)).toBe('skipped');
    expect(h.labels).toEqual([]);
    expect(h.errors).toEqual([]);
  });

  it('duplicates a locked layer or a locked group beside itself, but copies nothing into a locked group', async () => {
    const lockedLayer = setup(tree({ r1: true }));
    expect(lockedLayer.can('duplicate', ['r1'])).toBe(true);
    expect(await lockedLayer.run('duplicate', ['r1'])).toBe('committed');

    const lockedGroup = setup(tree({ G: true }));
    expect(lockedGroup.can('duplicate', ['G'])).toBe(true);
    expect(await lockedGroup.run('duplicate', ['G'])).toBe('committed');
    expect([...lockedLayer.labels, ...lockedGroup.labels]).toEqual(['Duplicate layer', 'Duplicate layer']);

    expect(lockedGroup.can('duplicate', ['r2'])).toBe(false);
    expect(lockedGroup.can('duplicate', ['r1', 'r2'])).toBe(false);
    expect(await lockedGroup.run('duplicate', ['r2'])).toBe('skipped');
    expect(lockedGroup.labels).toHaveLength(1);
    expect(lockedGroup.errors).toEqual([]);
  });

  it('offers nothing for an empty selection or without an engine model', () => {
    const h = setup(tree());
    for (const command of ['delete', 'duplicate', 'group', 'ungroup'] as const) {
      expect(h.can(command, [])).toBe(false);
      expect(canRunLayerCommand(null, command, ['r1'])).toBe(false);
    }
  });

  it('offers ungroup on a top-level leaf only when it has a group to leave', () => {
    const h = setup(tree());
    expect(h.can('ungroup', ['r1'])).toBe(false);
    expect(h.can('ungroup', ['r2', 'r3'])).toBe(true);
    expect(h.can('ungroup', ['r2', 'r4'])).toBe(false);
  });

  it('ends an open preview gesture only for a command that will run', async () => {
    const h = setup(tree({ r3: true }));

    h.run('delete', ['r3']);
    h.run('group', ['r2', 'r3']);
    h.run('ungroup', ['r3']);
    await h.run('duplicate', ['ghost']);
    expect(h.control.previewEnds).toBe(0);

    expect(h.run('delete', ['r1'])).toBe('committed');
    expect(h.control.previewEnds).toBe(1);
  });
});

describe('undo labels', () => {
  it('names delete, duplicate and group by how many nodes they act on', async () => {
    const h = setup(tree());

    expect(await h.run('duplicate', ['r1'])).toBe('committed');
    expect(await h.run('duplicate', ['r1', 'G'])).toBe('committed');
    expect(h.run('group', ['r1', 'r4'])).toBe('committed');
    expect(h.run('delete', ['G'])).toBe('committed');
    expect(h.run('delete', ['r1', 'r4'])).toBe('committed');

    expect(h.labels).toEqual([
      'Duplicate layer',
      'Duplicate 2 layers',
      'Group 2 layers',
      'Delete layer',
      'Delete 2 layers',
    ]);
    expect(h.errors).toEqual([]);
  });

  it('counts a group selected with its own child once', async () => {
    const h = setup(tree());

    expect(await h.run('duplicate', ['G', 'r2'])).toBe('committed');
    expect(h.run('group', ['r1', 'G', 'r2'])).toBe('committed');
    expect(h.run('delete', ['G', 'r2'])).toBe('committed');

    expect(h.labels).toEqual(['Duplicate layer', 'Group 2 layers', 'Delete layer']);
  });

  it('ungroups a group, and lifts leaves of one group above it instead, each named for what it did', () => {
    const h = setup(tree());

    expect(h.run('ungroup', ['r3', 'r2'])).toBe('committed');
    expect(outline(h.document())).toEqual(['r1', 'r2', 'r3', 'G', 'r4']);

    const regrouped = setup(tree());
    expect(regrouped.run('ungroup', ['G'])).toBe('committed');
    expect(outline(regrouped.document())).toEqual(['r1', 'r2', 'r3', 'r4']);

    expect([...h.labels, ...regrouped.labels]).toEqual(['Remove from group', 'Ungroup']);
  });
});

describe('group selection', () => {
  it('selects the new group, and puts the previous selection back when the commit fails', () => {
    const h = setup(tree());
    publishLayerPanelSelection({ primaryId: 'r1', projectId: h.projectId, selectedIds: ['r1', 'r4'] });

    expect(h.run('group', ['r1', 'r4'])).toBe('committed');
    const groupId = h.panel().primaryId!;
    expect(getDocumentIndex(h.document()).byId.get(groupId)?.node.type).toBe('group');
    expect(h.panel()).toMatchObject({
      expandedGroupIds: [groupId],
      selectedIds: [groupId],
    });

    const failing = setup(tree());
    publishLayerPanelSelection({ primaryId: 'r1', projectId: failing.projectId, selectedIds: ['r1', 'r4'] });
    failing.control.commitResult = { recovered: 'reverted', status: 'postcondition-failed' };
    expect(failing.run('group', ['r1', 'r4'])).toBe('failed');
    expect(failing.panel()).toMatchObject({
      expandedGroupIds: [],
      primaryId: 'r1',
      selectedIds: ['r1', 'r4'],
    });
  });
});

describe('failure feedback', () => {
  it.each<LayerCommand>(['delete', 'group', 'ungroup'])(
    '%s reports an eligible edit that does not land, once',
    async (command) => {
      const h = setup(tree());
      h.control.commitResult = { recovered: 'unreverted', status: 'postcondition-failed' };

      expect(await h.run(command, command === 'ungroup' ? ['G'] : ['r1', 'r4'])).toBe('failed');

      expect(h.errors).toEqual([
        ['Canvas edit failed', 'The edit could not be verified or reverted. Check the canvas before continuing.'],
      ]);
    }
  );

  it.each<[string, () => Promise<DuplicateLayersResult>, [string, string]]>([
    [
      'refuses as stale',
      () => Promise.resolve({ status: 'stale' }),
      ['Layer action failed', 'The layer changed before this action finished, so nothing was changed. Try again.'],
    ],
    [
      'cannot read a layer',
      () => Promise.resolve({ layerId: 'r1', layerName: 'Sky', status: 'unreadable' }),
      [
        'Layer action failed',
        '“Sky” could not be read (its image may have been deleted or damaged), so nothing was changed. Delete or replace that layer, then try again.',
      ],
    ],
    [
      'rejects the transaction',
      () => Promise.reject(new Error('rejected')),
      ['Layer action failed', 'The layer could not be copied.'],
    ],
  ])('duplicate reports once when the engine %s', async (_name, result, error) => {
    const h = setup(tree());
    h.control.duplicateResult = result;

    expect(await h.run('duplicate', ['r1'])).toBe('failed');
    expect(h.errors).toEqual([error]);
  });

  it('stays quiet while another edit holds the canvas', async () => {
    const h = setup(tree());
    h.control.duplicateResult = () => Promise.resolve({ status: 'busy' });
    expect(await h.run('duplicate', ['r1'])).toBe('skipped');
    h.control.commitResult = { status: 'busy' };
    expect(h.run('delete', ['r1'])).toBe('skipped');
    expect(h.errors).toEqual([]);
  });
});
