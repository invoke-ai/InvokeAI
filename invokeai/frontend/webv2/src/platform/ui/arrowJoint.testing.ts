import { server } from 'vitest/browser';

declare module 'vitest/browser' {
  interface BrowserCommands {
    /** Captures a rect of the test frame at device pixels, as base64 PNG; defined in `vitest.browser.config.mts`. */
    captureDevicePixels: (
      rect: { height: number; width: number; x: number; y: number },
      frameWidth: number
    ) => Promise<string>;
    /** Emulates a display's device pixel ratio, `null` restores it; defined in `vitest.browser.config.mts`. */
    emulateDevicePixelRatio: (ratio: number | null) => Promise<void>;
  }
}

/** Per-channel difference two pixels may have and still read as the same colour. */
const TOLERANCE = 4;

const sameColor = (a: readonly number[], b: readonly number[]) =>
  a.slice(0, 3).every((channel, i) => Math.abs(channel - b[i]!) <= TOLERANCE);

const hex = (pixel: readonly number[]) =>
  `#${pixel
    .slice(0, 3)
    .map((channel) => channel.toString(16).padStart(2, '0'))
    .join('')}`;

/**
 * Finds the pixels where an arrowed surface's edge shows through its arrow's base. At device resolution, it scans the
 * content's edge band (its border and the shading just inside it, 3px deep) across the middle of the tip's base. Each
 * pixel there must be the tip's own fill or what the content shows beyond the band at that point: its background, or
 * a child flush with the edge. Past the first device row inside the border, only the latter, so the tip never paints
 * over the content's children. Returns the offending pixels, described; an empty list is a seamless joint.
 */
export const findArrowJointSeams = async (content: HTMLElement, tip: HTMLElement): Promise<string[]> => {
  const side = content.dataset.placement?.split('-')[0] ?? 'bottom';
  const box = content.getBoundingClientRect();
  const tipBox = tip.getBoundingClientRect();
  // A region around the tip whose origin falls on a device pixel at common ratios (1, 1.25, 1.5, 2).
  const region = {
    height: 48,
    width: 48,
    x: Math.floor((tipBox.left + tipBox.right) / 2 / 4) * 4 - 24,
    y: Math.floor((tipBox.top + tipBox.bottom) / 2 / 4) * 4 - 24,
  };
  const png = await server.commands.captureDevicePixels(region, document.documentElement.clientWidth);
  const bitmap = await createImageBitmap(await (await fetch(`data:image/png;base64,${png}`)).blob());
  const scale = bitmap.width / region.width;
  const context = new OffscreenCanvas(bitmap.width, bitmap.height).getContext('2d')!;
  context.drawImage(bitmap, 0, 0);

  const vertical = side === 'top' || side === 'bottom';
  // The content's outer edge facing the trigger and the tip's centre along it, in the region's device pixels.
  const edge = {
    bottom: (box.top - region.y) * scale,
    left: (box.right - region.x) * scale,
    right: (box.left - region.x) * scale,
    top: (box.bottom - region.y) * scale,
  }[side]!;
  const inward = side === 'bottom' || side === 'right' ? 1 : -1;
  const along = vertical
    ? ((tipBox.left + tipBox.right) / 2 - region.x) * scale
    : ((tipBox.top + tipBox.bottom) / 2 - region.y) * scale;
  // Device row `depth` rows inside the edge: 0 is the content's outermost row, negative is out in the tip.
  const across = (depth: number) => Math.round(edge) + (inward === 1 ? depth : -1 - depth);
  const pixel = (depth: number, offset: number) => {
    const a = Math.floor(along + offset);
    const b = across(depth);
    return [...context.getImageData(vertical ? a : b, vertical ? b : a, 1, 1).data];
  };

  const band = Math.round(3 * scale);
  const halfSpan = Math.round(4 * scale);
  const tipFill = pixel(-band, 0);
  const seams: string[] = [];
  for (let depth = 0; depth < band; depth++) {
    for (let offset = -halfSpan; offset <= halfSpan; offset++) {
      const seen = pixel(depth, offset);
      const beyond = pixel(band + 1, offset);
      const allowed = depth <= 1 ? [beyond, tipFill] : [beyond];
      if (!allowed.some((color) => sameColor(seen, color))) {
        seams.push(`depth ${depth}, offset ${offset}: ${hex(seen)}, not ${allowed.map(hex).join(' or ')}`);
      }
    }
  }
  return seams;
};
