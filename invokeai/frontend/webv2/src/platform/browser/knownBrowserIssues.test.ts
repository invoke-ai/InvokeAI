import { describe, expect, it, vi } from 'vitest';

import {
  detectKnownBrowserIssues,
  KNOWN_BROWSER_ISSUES,
  type BrowserIssueDetectionEnvironment,
} from './knownBrowserIssues';

interface TestEnvironmentOptions {
  isContextAvailable?: boolean;
  mutateReadback?: (pixels: Uint8ClampedArray) => void;
  readbackError?: Error;
}

const createTestEnvironment = ({
  isContextAvailable = true,
  mutateReadback,
  readbackError,
}: TestEnvironmentOptions = {}): BrowserIssueDetectionEnvironment => {
  let writtenPixels = new Uint8ClampedArray();
  const context = {
    createImageData: (width: number, height: number) =>
      ({
        data: new Uint8ClampedArray(width * height * 4),
        height,
        width,
      }) as ImageData,
    getImageData: () => {
      if (readbackError) {
        throw readbackError;
      }

      const pixels = new Uint8ClampedArray(writtenPixels);
      mutateReadback?.(pixels);

      return { data: pixels } as ImageData;
    },
    putImageData: (imageData: ImageData) => {
      writtenPixels = new Uint8ClampedArray(imageData.data);
    },
  } as unknown as CanvasRenderingContext2D;
  const canvas = {
    getContext: vi.fn(() => (isContextAvailable ? context : null)),
    height: 0,
    width: 0,
  } as unknown as HTMLCanvasElement;

  return {
    createCanvas: vi.fn(() => canvas),
  };
};

const getDetectedIssueIds = async (environment: BrowserIssueDetectionEnvironment): Promise<readonly string[]> =>
  (await detectKnownBrowserIssues(environment)).map((issue) => issue.id);

describe('known browser issues', () => {
  it('accepts an exact canvas pixel readback', async () => {
    expect(await getDetectedIssueIds(createTestEnvironment())).toEqual([]);
  });

  it('detects a modified RGB byte', async () => {
    const environment = createTestEnvironment({
      mutateReadback: (pixels) => {
        pixels[0] = pixels[0] === 255 ? 254 : pixels[0] + 1;
      },
    });

    expect(await getDetectedIssueIds(environment)).toEqual(['canvas-readback-integrity']);
  });

  it('detects a modified alpha byte', async () => {
    const environment = createTestEnvironment({
      mutateReadback: (pixels) => {
        pixels[3] = 254;
      },
    });

    expect(await getDetectedIssueIds(environment)).toEqual(['canvas-readback-integrity']);
  });

  it('detects an unavailable 2D context', async () => {
    expect(await getDetectedIssueIds(createTestEnvironment({ isContextAvailable: false }))).toEqual([
      'canvas-readback-integrity',
    ]);
  });

  it('detects a blocked canvas readback', async () => {
    expect(
      await getDetectedIssueIds(createTestEnvironment({ readbackError: new Error('Canvas readback blocked') }))
    ).toEqual(['canvas-readback-integrity']);
  });

  it('returns only registry entries whose detectors match', async () => {
    const detectedIssues = await detectKnownBrowserIssues(
      createTestEnvironment({
        mutateReadback: (pixels) => {
          pixels[4] += 1;
        },
      })
    );

    expect(detectedIssues).toEqual([KNOWN_BROWSER_ISSUES[0]]);
  });
});
