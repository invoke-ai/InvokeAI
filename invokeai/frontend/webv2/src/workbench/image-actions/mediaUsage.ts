/**
 * Where a project still uses gallery media the user is about to delete. Deleting such media removes the backing
 * file under authored work, so the deletion flow must warn even when ordinary confirmations are turned off.
 *
 * The scan fails safe. The surfaces the panel names are listed below; whatever else in the project names a selected
 * item counts too, as `otherProjectState`, unless an exclusion below says it only shows or records media. A field added
 * later that stores a media name therefore warns until it is given a surface or an exclusion; it is never missed.
 */

import type { GalleryItemRef } from '@features/gallery/contracts';
import type { GalleryMediaUsageSurface } from '@features/gallery/mediaUsage';
import type { Project } from '@workbench/projectContracts';

import { toGalleryItemKey } from '@features/gallery/contracts';
import { collectMediaNames, MAX_HELD_NAME_LENGTH, type MediaNameRefs } from '@workbench/mediaReferences';
import { getProjectWidgetValues } from '@workbench/widgetState';

export interface MediaUsage {
  surface: GalleryMediaUsageSurface;
  /** The workflow's or snapshot's name, for the `workflow` and `canvasSnapshot` surfaces. */
  label?: string;
  /** Selected items this surface uses, in selection order. */
  refs: GalleryItemRef[];
}

type StackKind = keyof Project['canvas']['document']['stacks'];

/** The canvas layer stacks, each its own surface; a new stack kind fails to compile until it has one. */
const STACK_SURFACES = {
  raster: 'canvasRaster',
  control: 'canvasControl',
  regional_guidance: 'canvasRegionalGuidance',
  inpaint_mask: 'canvasInpaintMask',
} as const satisfies Record<StackKind, GalleryMediaUsageSurface>;

const STACK_KINDS = Object.keys(STACK_SURFACES) as StackKind[];

/**
 * How the remainder treats each project field: `scanned`, or `excluded` because it only shows or records media, or is
 * session state that is never saved. Typed against `Project`, so a new field fails to compile until it is placed here.
 * A key a newer document carries that this build does not know is scanned.
 */
const PROJECT_FIELDS = {
  // The listed surfaces read its layer stacks, staged results and each snapshot's document and staged results; the
  // rest of it, a stack kind or snapshot field added later included, is scanned here.
  canvas: 'scanned',
  // Session-only run log; it names every generated image.
  events: 'excluded',
  floatingWidgets: 'scanned',
  id: 'scanned',
  invocation: 'scanned',
  lastFloatingGeometry: 'scanned',
  layout: 'scanned',
  name: 'scanned',
  presetWorkingLayouts: 'scanned',
  promptHistory: 'scanned',
  // Session-only queue runs and their results; they name every generated image.
  queue: 'excluded',
  settings: 'scanned',
  // Session-only undo history, never saved; restoring it is not work the deletion breaks.
  undoRedo: 'excluded',
  // The last submitted graph per source widget, a record of a past run: the next run compiles from the widget's values.
  widgetGraphs: 'excluded',
  // Scanned without Gallery instances (below).
  widgetInstances: 'scanned',
  widgetRegions: 'scanned',
  // Session-only workflow edit histories, never saved.
  workflowHistories: 'excluded',
  // Scanned without each workflow's last run (below).
  workflows: 'scanned',
} as const satisfies Record<keyof Project, 'excluded' | 'scanned'>;

const isExcludedField = (key: string): boolean =>
  Object.hasOwn(PROJECT_FIELDS, key) && PROJECT_FIELDS[key as keyof Project] === 'excluded';

const scan = (roots: readonly unknown[]): MediaNameRefs => collectMediaNames(roots, () => false, MAX_HELD_NAME_LENGTH);

const selectedIn = (refs: MediaNameRefs, selected: readonly GalleryItemRef[]): GalleryItemRef[] =>
  selected.filter((ref) => (ref.kind === 'image' ? refs.images : refs.videos).has(ref.name));

/** `record` without `keys`: what a listed surface leaves of a container it reads only part of. */
const without = <Value extends object>(record: Value, keys: readonly PropertyKey[]): Partial<Value> =>
  Object.fromEntries(Object.entries(record).filter(([key]) => !keys.includes(key))) as Partial<Value>;

/**
 * The project without its excluded fields and without exactly what the listed surfaces scan (most of its bulk), so
 * anything they do not read, a stack kind or snapshot field added later included, is still scanned. Excluded within
 * scanned fields: Gallery instances (selection, recents, compare image, searches: they show media) and workflows'
 * last runs (a record of a run's output).
 */
const remainderRoots = (project: Project): unknown[] => {
  const { canvas, widgetInstances, workflows } = project;
  const rest: Record<string, unknown> = {
    ...project,
    canvas: {
      ...canvas,
      document: { ...canvas.document, stacks: without(canvas.document.stacks, STACK_KINDS) },
      snapshots: canvas.snapshots.map((snapshot) => without(snapshot, ['document', 'stagedImages'])),
      stagingArea: without(canvas.stagingArea, ['pendingImages']),
    },
    widgetInstances: Object.values(widgetInstances).filter((instance) => instance.typeId !== 'gallery'),
    workflows: {
      ...workflows,
      entries: workflows.entries.map(({ document, lastRun: _lastRun, ...entry }) => ({
        ...entry,
        document: without(document, ['nodes']),
      })),
    },
  };

  return Object.entries(rest).flatMap(([key, value]) => (isExcludedField(key) ? [] : [value]));
};

/**
 * The project's surfaces that use any of `selected`, in panel order: live work, staged canvas results and canvas
 * snapshots, which the user can still accept or restore; then `otherProjectState` for selected items that only the rest
 * of the project names (see `PROJECT_FIELDS` for what that leaves out).
 */
export const collectMediaUsage = (project: Project, selected: readonly GalleryItemRef[]): MediaUsage[] => {
  if (selected.length === 0) {
    return [];
  }

  const usage: MediaUsage[] = [];
  const reported = new Set<string>();
  const add = (surface: GalleryMediaUsageSurface, roots: readonly unknown[], label?: string): void => {
    const refs = selectedIn(scan(roots), selected);

    if (refs.length > 0) {
      refs.forEach((ref) => reported.add(toGalleryItemKey(ref)));
      usage.push(label === undefined ? { refs, surface } : { label, refs, surface });
    }
  };

  add('generateReferenceImages', [getProjectWidgetValues(project, 'generate').referenceImages]);

  for (const kind of STACK_KINDS) {
    add(STACK_SURFACES[kind], [project.canvas.document.stacks[kind]]);
  }
  add('canvasStaging', [project.canvas.stagingArea.pendingImages]);

  for (const snapshot of project.canvas.snapshots) {
    add('canvasSnapshot', [snapshot.document, snapshot.stagedImages], snapshot.name);
  }

  add('upscale', [getProjectWidgetValues(project, 'upscale').inputImage]);
  add('video', [getProjectWidgetValues(project, 'video')]);

  for (const entry of project.workflows.entries) {
    add('workflow', [entry.document.nodes], entry.document.name);
  }

  const elsewhere = selectedIn(scan(remainderRoots(project)), selected).filter(
    (ref) => !reported.has(toGalleryItemKey(ref))
  );

  if (elsewhere.length > 0) {
    usage.push({ refs: elsewhere, surface: 'otherProjectState' });
  }

  return usage;
};
