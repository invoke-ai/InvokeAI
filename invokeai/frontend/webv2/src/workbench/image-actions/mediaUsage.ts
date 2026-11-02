/**
 * Where a project still uses gallery media the user is about to delete. Deleting such media removes the backing
 * file under authored work, so the deletion flow must warn even when ordinary confirmations are turned off.
 */

import type { GalleryItemRef } from '@features/gallery/contracts';
import type { GalleryMediaUsageSurface } from '@features/gallery/mediaUsage';
import type { Project } from '@workbench/projectContracts';

import { collectMediaNames, MAX_HELD_NAME_LENGTH, type MediaNameRefs } from '@workbench/mediaReferences';
import { getProjectWidgetValues } from '@workbench/widgetState';

export interface MediaUsage {
  surface: GalleryMediaUsageSurface;
  /** The workflow's or snapshot's name, for the `workflow` and `canvasSnapshot` surfaces. */
  label?: string;
  /** Selected names this surface uses, in selection order. */
  names: string[];
}

type UsageProject = Pick<Project, 'canvas' | 'widgetInstances' | 'workflows'>;

const scan = (roots: readonly unknown[]): MediaNameRefs => collectMediaNames(roots, () => false, MAX_HELD_NAME_LENGTH);

const selectedNamesIn = (refs: MediaNameRefs, selected: readonly GalleryItemRef[]): string[] =>
  selected.filter((ref) => (ref.kind === 'image' ? refs.images : refs.videos).has(ref.name)).map((ref) => ref.name);

/**
 * The project's saved surfaces that use any of `selected`, in panel order: live work, staged canvas results and canvas
 * snapshots, which the user can still accept or restore. Undo history (never saved) does not count, and neither do the
 * Gallery's selection, recents or compare image, nor a workflow's last run: they show media, they are not work on it.
 */
export const collectMediaUsage = (project: UsageProject, selected: readonly GalleryItemRef[]): MediaUsage[] => {
  if (selected.length === 0) {
    return [];
  }

  const usage: MediaUsage[] = [];
  const add = (surface: GalleryMediaUsageSurface, roots: readonly unknown[], label?: string): void => {
    const names = selectedNamesIn(scan(roots), selected);

    if (names.length > 0) {
      usage.push(label === undefined ? { names, surface } : { label, names, surface });
    }
  };

  add('generateReferenceImages', [getProjectWidgetValues(project, 'generate').referenceImages]);

  const { stacks } = project.canvas.document;
  add('canvasRaster', [stacks.raster]);
  add('canvasControl', [stacks.control]);
  add('canvasRegionalGuidance', [stacks.regional_guidance]);
  add('canvasInpaintMask', [stacks.inpaint_mask]);
  add('canvasStaging', [project.canvas.stagingArea.pendingImages]);

  for (const snapshot of project.canvas.snapshots) {
    add('canvasSnapshot', [snapshot.document, snapshot.stagedImages], snapshot.name);
  }

  add('upscale', [getProjectWidgetValues(project, 'upscale').inputImage]);
  add('video', [getProjectWidgetValues(project, 'video')]);

  for (const entry of project.workflows.entries) {
    add('workflow', [entry.document.nodes], entry.document.name);
  }

  return usage;
};
