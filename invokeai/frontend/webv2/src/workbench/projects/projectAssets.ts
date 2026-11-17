import {
  collectMediaNames,
  IMAGE_NAME_KEYS,
  MAX_HELD_NAME_LENGTH,
  type MediaNameRefs,
  VIDEO_NAME_KEYS,
} from '@workbench/mediaReferences';

export const PROJECT_HISTORY_ROOT_KEYS: ReadonlySet<string> = new Set(['events', 'graphHistory', 'queue']);

export const PROJECT_HISTORY_KEYS: ReadonlySet<string> = new Set(['recentImages', 'snapshot', 'snapshots']);

/** Skip and strip installation-specific selection references so unbundled names cannot travel silently. */
export const GALLERY_SELECTION_KEYS: ReadonlySet<string> = new Set([
  'compareImage',
  'selectedImage',
  'selectedImageName',
  'selectedImageNames',
]);

/** Strip gallery board IDs, preserving authored workflow board inputs; auto-add falls back to the inbox. */
export const GALLERY_INSTALLATION_KEYS: ReadonlySet<string> = new Set([
  'autoAddBoardId',
  'projectBoardId',
  'selectedBoardId',
]);

export const GALLERY_POSITION_KEYS: ReadonlySet<string> = new Set(['galleryPage']);

const INSTALLATION_STATE_KEYS: ReadonlySet<string> = new Set([
  ...GALLERY_SELECTION_KEYS,
  ...GALLERY_INSTALLATION_KEYS,
  ...GALLERY_POSITION_KEYS,
]);

/** Transfer blanks these and every load rebuilds them from the same lists; here because the Launchpad loads it eagerly. */
export const GENERATED_IMAGE_URL_FIELDS = { imageUrl: 'full', thumbnailUrl: 'thumbnail' } as const;

export const GALLERY_ITEM_URL_FIELDS = { fullUrl: 'full', thumbnailUrl: 'thumbnail' } as const;

const isDerivedMediaUrlKey = (key: string): boolean =>
  Object.hasOwn(GENERATED_IMAGE_URL_FIELDS, key) || Object.hasOwn(GALLERY_ITEM_URL_FIELDS, key);

export type ProjectAssetRefs = MediaNameRefs;

export interface CanvasHeldAssetRefs {
  readonly images: readonly string[];
  readonly videos: readonly string[];
}

/** A generation-frame rect, in document pixels. */
export interface CanvasFrameRect {
  readonly height: number;
  readonly width: number;
  readonly x: number;
  readonly y: number;
}

export interface LiveCanvasEngine {
  heldAssets(): CanvasHeldAssetRefs;
  subscribe(listener: () => void): () => void;
  flushPendingPixels(): Promise<void>;
  commitFrame(from: CanvasFrameRect, to: CanvasFrameRect, label: string): boolean;
}

export interface LiveCanvasEngines {
  heldAssets(projectId: string): CanvasHeldAssetRefs | undefined;
  register(projectId: string, source: LiveCanvasEngine): () => void;
  subscribe(listener: () => void): () => void;
  flushPendingPixels(projectId: string): Promise<void>;
  commitFrame(projectId: string, from: CanvasFrameRect, to: CanvasFrameRect, label: string): boolean;
}

export const createLiveCanvasEngines = (): LiveCanvasEngines => {
  const sources = new Map<string, LiveCanvasEngine>();
  const listeners = new Set<() => void>();
  const notify = (): void => listeners.forEach((listener) => listener());
  return {
    commitFrame: (projectId, from, to, label) => sources.get(projectId)?.commitFrame(from, to, label) ?? false,
    flushPendingPixels: (projectId) => sources.get(projectId)?.flushPendingPixels() ?? Promise.resolve(),
    heldAssets: (projectId) => sources.get(projectId)?.heldAssets(),
    register: (projectId, source) => {
      sources.set(projectId, source);
      const unsubscribe = source.subscribe(notify);
      notify();
      return () => {
        unsubscribe();
        if (sources.get(projectId) === source) {
          sources.delete(projectId);
          notify();
        }
      };
    },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const withoutHistoryRoots = (projectDocument: object): unknown[] =>
  Object.entries(projectDocument).flatMap(([key, value]) => (PROJECT_HISTORY_ROOT_KEYS.has(key) ? [] : [value]));

export const collectLiveAssetRefs = (projectDocument: Record<string, unknown>): ProjectAssetRefs =>
  collectMediaNames(
    withoutHistoryRoots(projectDocument),
    (key) => PROJECT_HISTORY_KEYS.has(key) || GALLERY_SELECTION_KEYS.has(key)
  );

const isHeldSkipKey = (key: string): boolean => key === 'recentImages' || GALLERY_SELECTION_KEYS.has(key);

export const collectHeldAssetRefs = (projects: readonly object[]): ProjectAssetRefs =>
  collectMediaNames(projects.flatMap(withoutHistoryRoots), isHeldSkipKey, MAX_HELD_NAME_LENGTH);

/** Containers whose children edits replace independently, so editing one workflow rescans only that workflow. */
const SPLIT_HELD_KEYS: ReadonlySet<string> = new Set([
  'canvas',
  'snapshots',
  'undoRedo',
  'past',
  'future',
  'workflows',
  'entries',
]);

const SPLIT_CHILDREN_HELD_KEYS: ReadonlySet<string> = new Set(['workflowHistories']);

type HeldPart = ProjectAssetRefs | CanvasHeldAssetRefs;

const scanHeld = (node: object): ProjectAssetRefs => collectMediaNames([node], isHeldSkipKey, MAX_HELD_NAME_LENGTH);

const pushNamed = (refs: ProjectAssetRefs, parts: HeldPart[]): void => {
  if (refs.images.size || refs.videos.size) {
    parts.push(refs);
  }
};

/** Unchanged part identities return the previous result; equal name sets under new identities are left to the lease. */
export const createOpenProjectsHeldMediaReader = (
  getProjects: () => readonly (object & { id: string })[],
  getCanvasRefs: (projectId: string) => CanvasHeldAssetRefs | undefined
): (() => { images: string[]; videos: string[] }) => {
  const scans = new WeakMap<object, ProjectAssetRefs>();
  const residueScans = new WeakMap<object, ProjectAssetRefs>();
  let last: { parts: HeldPart[]; result: { images: string[]; videos: string[] } } | null = null;

  const collectParts = (node: object, isRoot: boolean, parts: HeldPart[], splitChildren = false): void => {
    const entries = Array.isArray(node)
      ? node.map((value, index) => [String(index), value] as const)
      : Object.entries(node);
    let residue: Record<string, unknown> | null = null;
    for (const [key, value] of entries) {
      if (isHeldSkipKey(key) || (isRoot && PROJECT_HISTORY_ROOT_KEYS.has(key))) {
        continue;
      }
      if (typeof value !== 'object' || value === null) {
        (residue ??= {})[key] = value;
      } else if (splitChildren || SPLIT_HELD_KEYS.has(key)) {
        collectParts(value, false, parts, SPLIT_CHILDREN_HELD_KEYS.has(key));
      } else {
        let refs = scans.get(value);
        if (!refs) {
          refs = scanHeld(value);
          scans.set(value, refs);
        }
        pushNamed(refs, parts);
      }
    }
    if (residue) {
      let refs = residueScans.get(node);
      if (!refs) {
        refs = scanHeld(residue);
        residueScans.set(node, refs);
      }
      pushNamed(refs, parts);
    }
  };

  return () => {
    const parts: HeldPart[] = [];
    for (const project of getProjects()) {
      collectParts(project, true, parts);
      const retained = getCanvasRefs(project.id);
      if (retained) {
        parts.push(retained);
      }
    }
    if (last && last.parts.length === parts.length && parts.every((part, index) => part === last!.parts[index])) {
      return last.result;
    }
    const images = new Set<string>();
    const videos = new Set<string>();
    for (const part of parts) {
      part.images.forEach((name) => images.add(name));
      part.videos.forEach((name) => videos.add(name));
    }
    last = { parts, result: { images: [...images], videos: [...videos] } };
    return last.result;
  };
};

/** `{ drop: true }` removes the key, `{ value }` replaces it, `null` recurses into it. */
type NodeVisit = { drop: true } | { value: unknown } | null;

/** Preserve structural sharing; unchanged documents return the same object. */
const mapDocument = (node: unknown, visit: (key: string, value: unknown) => NodeVisit): unknown => {
  if (Array.isArray(node)) {
    let hasChanged = false;
    const next = node.map((item) => {
      const mapped = mapDocument(item, visit);

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
    const visited = visit(key, value);

    if (visited === null) {
      const mapped = mapDocument(value, visit);

      next[key] = mapped;
      hasChanged ||= mapped !== value;
      continue;
    }

    if ('drop' in visited) {
      hasChanged = true;
      continue;
    }

    next[key] = visited.value;
    hasChanged ||= visited.value !== value;
  }

  return hasChanged ? next : node;
};

export const stripInstallationState = (projectDocument: Record<string, unknown>): Record<string, unknown> =>
  mapDocument(projectDocument, (key, value) => {
    if (INSTALLATION_STATE_KEYS.has(key)) {
      return { drop: true };
    }

    // Blank derived URLs rather than deleting keys; each load derives them from the remapped names.
    return isDerivedMediaUrlKey(key) && typeof value === 'string' ? { value: '' } : null;
  }) as Record<string, unknown>;

export interface ProjectAssetMappings {
  images: ReadonlyMap<string, string>;
  videos: ReadonlyMap<string, string>;
}

export const remapAssetRefs = (
  projectDocument: Record<string, unknown>,
  mappings: ProjectAssetMappings
): Record<string, unknown> =>
  mappings.images.size === 0 && mappings.videos.size === 0
    ? projectDocument
    : (mapDocument(projectDocument, (key, value) => {
        if (typeof value !== 'string') {
          return null;
        }

        const mapping = IMAGE_NAME_KEYS.has(key) ? mappings.images : VIDEO_NAME_KEYS.has(key) ? mappings.videos : null;

        return mapping === null ? null : { value: mapping.get(value) ?? value };
      }) as Record<string, unknown>);

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

    // recentImages must be newest-first for index zero to identify the latest result.
    for (const image of values.recentImages) {
      if (isRecord(image) && typeof image.imageName === 'string' && image.imageName !== '') {
        return image.imageName;
      }
    }
  }

  return null;
};

const rawLeaves = (nodes: unknown): Record<string, unknown>[] =>
  Array.isArray(nodes)
    ? nodes.flatMap((node) => (!isRecord(node) ? [] : node.type === 'group' ? rawLeaves(node.children) : [node]))
    : [];

const readTopmostCanvasImageName = (projectDocument: Record<string, unknown>): string | null => {
  const canvas = projectDocument.canvas;

  if (!isRecord(canvas) || !isRecord(canvas.document) || !isRecord(canvas.document.stacks)) {
    return null;
  }

  for (const layer of rawLeaves(canvas.document.stacks.raster)) {
    if (!isRecord(layer.source)) {
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

export const selectCoverImageName = (projectDocument: Record<string, unknown>): string | null =>
  readGalleryRecentImageName(projectDocument) ?? readTopmostCanvasImageName(projectDocument);
