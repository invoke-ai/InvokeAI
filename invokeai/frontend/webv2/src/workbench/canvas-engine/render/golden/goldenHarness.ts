import type { RasterSurface } from '@workbench/canvas-engine/render/raster';

import { commands } from 'vitest/browser';

/**
 * Reviewed Chromium PNG goldens include raster/browser metadata. Flatten onto opaque backgrounds before channel
 * comparison to avoid premultiplied-alpha rounding. Update only via the golden command; normal failures write
 * ignored expected/actual/diff artifacts.
 */
export interface GoldenTolerance {
  /** Largest per-channel difference a pixel may show before it counts as differing. */
  readonly maxChannelDelta: number;
  /** How many differing pixels the frame may contain. */
  readonly maxDifferingPixels: number;
}

export interface GoldenOptions {
  readonly tolerance?: GoldenTolerance;
  /** The colour the frame is flattened over before comparison. */
  readonly background?: string;
  /** Where failure artifacts go when they must not shadow the golden's own (negative tests). */
  readonly artifactName?: string;
}

export const EXACT: GoldenTolerance = { maxChannelDelta: 0, maxDifferingPixels: 0 };

/** For interpolation-sensitive frames (rotation, scaling): small channel noise on a few edge pixels. */
export const INTERPOLATED: GoldenTolerance = { maxChannelDelta: 16, maxDifferingPixels: 64 };

const GOLDEN_DIR = 'src/workbench/canvas-engine/render/golden/__golden__';

// A string define such as "false" is truthy and would silently turn every run
// into a baseline rewrite (Vitest 5 browser mode injects strings verbatim).
if (typeof __CANVAS_GOLDEN_UPDATE__ !== 'boolean') {
  throw new TypeError(`__CANVAS_GOLDEN_UPDATE__ must be a boolean define, got ${typeof __CANVAS_GOLDEN_UPDATE__}.`);
}
export const ARTIFACT_DIR = 'src/workbench/canvas-engine/render/golden/__screenshots__/golden';
const DEFAULT_BACKGROUND = '#101010';

export type GoldenArtifactKind = 'expected' | 'actual' | 'diff';

export const goldenArtifactPath = (name: string, kind: GoldenArtifactKind): string =>
  `${ARTIFACT_DIR}/${name}.${kind}.png`;

interface GoldenRecord {
  readonly background: string;
  readonly height: number;
  readonly rasterBackend: 'dom' | 'offscreen';
  readonly tolerance: GoldenTolerance;
  readonly userAgent: string;
  readonly width: number;
}

const rasterBackendOf = (surface: RasterSurface): GoldenRecord['rasterBackend'] =>
  typeof OffscreenCanvas !== 'undefined' && surface.canvas instanceof OffscreenCanvas ? 'offscreen' : 'dom';

/** The record fields the current run must reproduce before pixels are worth comparing. */
const recordMismatch = (baseline: GoldenRecord, current: GoldenRecord): string | null => {
  const differences: string[] = [];
  for (const key of ['background', 'width', 'height', 'rasterBackend'] as const) {
    if (baseline[key] !== current[key]) {
      differences.push(`${key} ${String(baseline[key])} -> ${String(current[key])}`);
    }
  }
  if (
    baseline.tolerance.maxChannelDelta !== current.tolerance.maxChannelDelta ||
    baseline.tolerance.maxDifferingPixels !== current.tolerance.maxDifferingPixels
  ) {
    differences.push(`tolerance ${JSON.stringify(baseline.tolerance)} -> ${JSON.stringify(current.tolerance)}`);
  }
  return differences.length > 0 ? differences.join(', ') : null;
};

const flatten = (surface: RasterSurface, background: string): ImageData => {
  const canvas = document.createElement('canvas');
  canvas.width = surface.width;
  canvas.height = surface.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(surface.canvas, 0, 0);
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
};

const toBase64 = (bytes: Uint8Array): string => {
  let binary = '';
  for (let index = 0; index < bytes.length; index += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(index, index + 0x8000));
  }
  return btoa(binary);
};

const fromBase64 = (base64: string): Uint8Array<ArrayBuffer> => {
  const binary = atob(base64);
  const bytes = new Uint8Array(new ArrayBuffer(binary.length));
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
};

const encodePng = async (image: ImageData): Promise<string> => {
  const canvas = document.createElement('canvas');
  canvas.width = image.width;
  canvas.height = image.height;
  canvas.getContext('2d')!.putImageData(image, 0, 0);
  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((result) => {
      if (result) {
        resolve(result);
      } else {
        reject(new Error('PNG encoding failed'));
      }
    }, 'image/png');
  });
  return toBase64(new Uint8Array(await blob.arrayBuffer()));
};

const decodePng = async (base64: string): Promise<ImageData> => {
  const bitmap = await createImageBitmap(new Blob([fromBase64(base64)], { type: 'image/png' }), {
    colorSpaceConversion: 'none',
    premultiplyAlpha: 'none',
  });
  const canvas = document.createElement('canvas');
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
};

interface Comparison {
  readonly differing: number;
  readonly worstDelta: number;
  readonly diff: ImageData;
}

const compare = (expected: ImageData, actual: ImageData, tolerance: GoldenTolerance): Comparison => {
  const diff = new ImageData(expected.width, expected.height);
  let differing = 0;
  let worstDelta = 0;
  for (let index = 0; index < expected.data.length; index += 4) {
    let delta = 0;
    for (let channel = 0; channel < 3; channel += 1) {
      delta = Math.max(delta, Math.abs(expected.data[index + channel]! - actual.data[index + channel]!));
    }
    worstDelta = Math.max(worstDelta, delta);
    const differs = delta > tolerance.maxChannelDelta;
    differing += differs ? 1 : 0;
    diff.data[index] = differs ? 255 : expected.data[index]! >> 2;
    diff.data[index + 1] = differs ? 0 : expected.data[index + 1]! >> 2;
    diff.data[index + 2] = differs ? 0 : expected.data[index + 2]! >> 2;
    diff.data[index + 3] = 255;
  }
  return { diff, differing, worstDelta };
};

const isMissingFile = (error: unknown): boolean => error instanceof Error && error.message.includes('ENOENT');

const readGoldenFile = async (path: string, encoding: 'base64' | 'utf-8'): Promise<string | null> => {
  try {
    return await commands.readFile(path, encoding);
  } catch (error) {
    if (isMissingFile(error)) {
      return null;
    }
    throw new Error(`Golden file "${path}" is unreadable; inspect it before updating baselines.`, { cause: error });
  }
};

const readBaseline = async (name: string): Promise<{ image: ImageData; record: GoldenRecord } | null> => {
  const [png, json] = await Promise.all([
    readGoldenFile(`${GOLDEN_DIR}/${name}.png`, 'base64'),
    readGoldenFile(`${GOLDEN_DIR}/${name}.json`, 'utf-8'),
  ]);
  if (png === null && json === null) {
    return null;
  }
  if (png === null || json === null) {
    throw new Error(`Golden "${name}" has only one of its PNG and JSON files; restore the pair before updating.`);
  }
  try {
    return { image: await decodePng(png), record: JSON.parse(json) as GoldenRecord };
  } catch (error) {
    throw new Error(`Golden "${name}" could not be decoded; inspect it before updating baselines.`, { cause: error });
  }
};

const writeBaseline = async (name: string, image: ImageData, record: GoldenRecord): Promise<void> => {
  await commands.writeFile(`${GOLDEN_DIR}/${name}.png`, await encodePng(image), 'base64');
  await commands.writeFile(`${GOLDEN_DIR}/${name}.json`, `${JSON.stringify(record, null, 2)}\n`, 'utf-8');
};

const writeArtifacts = async (name: string, images: Partial<Record<GoldenArtifactKind, ImageData>>): Promise<void> => {
  await Promise.all(
    (Object.entries(images) as [GoldenArtifactKind, ImageData][]).map(async ([kind, image]) =>
      commands.writeFile(goldenArtifactPath(name, kind), await encodePng(image), 'base64')
    )
  );
};

/** Compares `surface` with the reviewed golden `name`, or rewrites the golden in update mode. */
export const expectGolden = async (
  name: string,
  surface: RasterSurface,
  options: GoldenOptions = {}
): Promise<void> => {
  const tolerance = options.tolerance ?? EXACT;
  const background = options.background ?? DEFAULT_BACKGROUND;
  const artifactName = options.artifactName ?? name;
  const actual = flatten(surface, background);
  const record: GoldenRecord = {
    background,
    height: actual.height,
    rasterBackend: rasterBackendOf(surface),
    tolerance,
    userAgent: navigator.userAgent,
    width: actual.width,
  };
  if (__CANVAS_GOLDEN_UPDATE__) {
    await writeBaseline(name, actual, record);
    return;
  }
  const baseline = await readBaseline(name);
  if (!baseline) {
    throw new Error(`Missing golden "${name}". Review the frame, then run pnpm run test:browser:update-goldens.`);
  }
  const mismatch = recordMismatch(baseline.record, record);
  if (mismatch) {
    await writeArtifacts(artifactName, { actual, expected: baseline.image });
    throw new Error(
      `Golden "${name}" was recorded under a different configuration (${mismatch}); ` +
        `re-review it and run pnpm run test:browser:update-goldens.`
    );
  }
  const result = compare(baseline.image, actual, tolerance);
  if (result.differing > tolerance.maxDifferingPixels) {
    await writeArtifacts(artifactName, { actual, diff: result.diff, expected: baseline.image });
    throw new Error(
      `Golden "${name}" differs: ${result.differing} pixels beyond ${tolerance.maxChannelDelta} per channel ` +
        `(worst ${result.worstDelta}), allowed ${tolerance.maxDifferingPixels}. ` +
        `Baseline recorded on ${baseline.record.userAgent}; this run is ${record.userAgent}. ` +
        `Artifacts: ${goldenArtifactPath(artifactName, 'expected')} and siblings.`
    );
  }
};
