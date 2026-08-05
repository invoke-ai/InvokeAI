/**
 * Which images a project document points at, and how to rewrite those pointers.
 *
 * The document stores pixels the way `canvas-engine/contracts.ts` describes:
 * by name, never by URL or inline data. That is what makes the document small
 * enough to autosave, and also what makes a bare document export useless on
 * another install — the names resolve to nothing there. Bundling the bytes into
 * an `.invk` archive means first knowing which names the document actually uses.
 *
 * ### Collection is by key, not by path
 *
 * Exactly two keys in the document hold an image name: `imageName` (the webv2
 * canvas and gallery contracts) and `image_name` (graph node values, which
 * mirror the backend's field naming). Collecting every string found at those two
 * keys is complete by construction — a new control-adapter kind, a new node
 * field, a widget nobody has written yet, all get picked up without touching
 * this file.
 *
 * The alternative, walking a hand-written list of paths, is what the legacy
 * frontend does, and it is why its collector enumerates `ip_adapter` and
 * `flux_redux` by name and silently drops the pixels of any adapter added since.
 * A false positive here costs nothing: a string that is not really an image name
 * fails to resolve on the server and is skipped.
 *
 * ### Collection skips history, remapping does not
 *
 * A project document carries far more than the live document. Every queue entry
 * embeds a whole canvas snapshot, `canvas.snapshots` holds full document copies,
 * and the gallery widget keeps up to sixty recent results. Bundling all of that
 * turns a working project into a multi-gigabyte archive, so
 * {@link collectLiveImageRefs} walks only what the project needs in order to
 * open correctly.
 *
 * {@link remapImageRefs} walks everything, with no skips. The asymmetry is the
 * point: if a bundled image comes back from the server under a new name, every
 * reference to it has to follow — including the ones in history we chose not to
 * bundle, which would otherwise point at the pre-import name forever.
 */

/** Keys whose string values are image names. */
const IMAGE_NAME_KEYS = new Set(['imageName', 'image_name']);

/** Top-level document keys that are history rather than live content. */
export const PROJECT_HISTORY_ROOT_KEYS: ReadonlySet<string> = new Set(['events', 'graphHistory', 'queue']);

/** Keys that introduce history at any depth (`canvas.snapshots`, the gallery's recents). */
export const PROJECT_HISTORY_KEYS: ReadonlySet<string> = new Set(['recentImages', 'snapshot', 'snapshots']);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const collectFrom = (node: unknown, names: Set<string>): void => {
  if (Array.isArray(node)) {
    for (const item of node) {
      collectFrom(item, names);
    }

    return;
  }

  if (!isRecord(node)) {
    return;
  }

  for (const [key, value] of Object.entries(node)) {
    if (PROJECT_HISTORY_KEYS.has(key)) {
      continue;
    }

    if (IMAGE_NAME_KEYS.has(key) && typeof value === 'string' && value !== '') {
      names.add(value);
      continue;
    }

    collectFrom(value, names);
  }
};

/**
 * Every image name the project needs in order to open: canvas layers and masks,
 * reference images, widget values, and graph node inputs. History is excluded —
 * see the module docblock.
 */
export const collectLiveImageRefs = (projectDocument: Record<string, unknown>): Set<string> => {
  const names = new Set<string>();

  for (const [key, value] of Object.entries(projectDocument)) {
    if (PROJECT_HISTORY_ROOT_KEYS.has(key)) {
      continue;
    }

    collectFrom(value, names);
  }

  return names;
};

const remapNode = (node: unknown, mapping: ReadonlyMap<string, string>): unknown => {
  if (Array.isArray(node)) {
    let hasChanged = false;
    const next = node.map((item) => {
      const mapped = remapNode(item, mapping);

      hasChanged ||= mapped !== item;

      return mapped;
    });

    return hasChanged ? next : node;
  }

  if (!isRecord(node)) {
    return node;
  }

  let hasChanged = false;
  const next: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(node)) {
    if (IMAGE_NAME_KEYS.has(key) && typeof value === 'string') {
      const replacement = mapping.get(value);

      next[key] = replacement ?? value;
      hasChanged ||= replacement !== undefined && replacement !== value;
      continue;
    }

    const mapped = remapNode(value, mapping);

    next[key] = mapped;
    hasChanged ||= mapped !== value;
  }

  return hasChanged ? next : node;
};

/**
 * Rewrite every image reference in the document through `mapping`. Names absent
 * from the mapping are left alone, and subtrees with nothing to rewrite keep
 * their identity, so an empty mapping returns the document unchanged.
 */
export const remapImageRefs = (
  projectDocument: Record<string, unknown>,
  mapping: ReadonlyMap<string, string>
): Record<string, unknown> =>
  mapping.size === 0 ? projectDocument : (remapNode(projectDocument, mapping) as Record<string, unknown>);

const readGalleryRecentImageName = (projectDocument: Record<string, unknown>): string | null => {
  const instances = projectDocument.widgetInstances;

  if (!isRecord(instances)) {
    return null;
  }

  for (const instance of Object.values(instances)) {
    if (!isRecord(instance) || instance.typeId !== 'gallery' || !isRecord(instance.state)) {
      continue;
    }

    const values = instance.state.values;

    if (!isRecord(values) || !Array.isArray(values.recentImages)) {
      continue;
    }

    // `recentImages` is persisted newest-first and bounded (see
    // `getBoundedRecentImages`), so index 0 is the project's latest result.
    for (const image of values.recentImages) {
      if (isRecord(image) && typeof image.imageName === 'string' && image.imageName !== '') {
        return image.imageName;
      }
    }
  }

  return null;
};

const readTopmostCanvasImageName = (projectDocument: Record<string, unknown>): string | null => {
  const canvas = projectDocument.canvas;

  if (!isRecord(canvas) || !isRecord(canvas.document) || !Array.isArray(canvas.document.layers)) {
    return null;
  }

  // Layer index 0 is the top-most layer, which is the one a person would call
  // "what this project looks like".
  for (const layer of canvas.document.layers) {
    if (!isRecord(layer) || !isRecord(layer.source)) {
      continue;
    }

    const source = layer.source;
    const ref = source.type === 'image' ? source.image : source.type === 'paint' ? source.bitmap : null;

    if (isRecord(ref) && typeof ref.imageName === 'string' && ref.imageName !== '') {
      return ref.imageName;
    }
  }

  return null;
};

/**
 * The image that best stands for this project: its newest generated result,
 * falling back to the top-most canvas layer with pixels. `null` for a project
 * that has neither, which is the permanent state for one that has produced
 * nothing — `ProjectCover` renders a glyph for exactly that case.
 */
export const selectCoverImageName = (projectDocument: Record<string, unknown>): string | null =>
  readGalleryRecentImageName(projectDocument) ?? readTopmostCanvasImageName(projectDocument);
