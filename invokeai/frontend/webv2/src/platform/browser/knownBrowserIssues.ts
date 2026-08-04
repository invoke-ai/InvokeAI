export type KnownBrowserIssueId = 'canvas-readback-integrity';

export type KnownBrowserIssueSeverity = 'warning' | 'error';

export interface KnownBrowserIssueWorkaround {
  id: string;
  instructionKey: string;
  copyableValue?: string;
}

export interface BrowserIssueDetectionEnvironment {
  createCanvas: () => HTMLCanvasElement | null;
}

export interface KnownBrowserIssue {
  id: KnownBrowserIssueId;
  severity: KnownBrowserIssueSeverity;
  titleKey: string;
  descriptionKey: string;
  workarounds: readonly KnownBrowserIssueWorkaround[];
  detect: (environment: BrowserIssueDetectionEnvironment) => boolean | Promise<boolean>;
}

const CANVAS_PROBE_SIZE = 32;

const DEFAULT_DETECTION_ENVIRONMENT: BrowserIssueDetectionEnvironment = {
  createCanvas: () => {
    if (typeof document === 'undefined') {
      return null;
    }

    return document.createElement('canvas');
  },
};

const createExpectedCanvasPixels = (): Uint8ClampedArray => {
  const pixels = new Uint8ClampedArray(CANVAS_PROBE_SIZE * CANVAS_PROBE_SIZE * 4);

  for (let pixelIndex = 0; pixelIndex < CANVAS_PROBE_SIZE * CANVAS_PROBE_SIZE; pixelIndex++) {
    const byteIndex = pixelIndex * 4;

    pixels[byteIndex] = 32 + ((pixelIndex * 17) % 192);
    pixels[byteIndex + 1] = 32 + ((pixelIndex * 31) % 192);
    pixels[byteIndex + 2] = 32 + ((pixelIndex * 47) % 192);
    pixels[byteIndex + 3] = 255;
  }

  return pixels;
};

const hasUnsafeCanvasReadback = (environment: BrowserIssueDetectionEnvironment): boolean => {
  try {
    const canvas = environment.createCanvas();

    if (!canvas) {
      return true;
    }

    canvas.width = CANVAS_PROBE_SIZE;
    canvas.height = CANVAS_PROBE_SIZE;

    const context = canvas.getContext('2d');

    if (!context) {
      return true;
    }

    const expectedPixels = createExpectedCanvasPixels();
    const imageData = context.createImageData(CANVAS_PROBE_SIZE, CANVAS_PROBE_SIZE);
    imageData.data.set(expectedPixels);
    context.putImageData(imageData, 0, 0);

    const actualPixels = context.getImageData(0, 0, CANVAS_PROBE_SIZE, CANVAS_PROBE_SIZE).data;

    if (actualPixels.length !== expectedPixels.length) {
      return true;
    }

    return actualPixels.some((value, index) => value !== expectedPixels[index]);
  } catch {
    return true;
  }
};

export const KNOWN_BROWSER_ISSUES = [
  {
    descriptionKey: 'launchpad.browserIssues.canvasReadbackIntegrity.description',
    detect: hasUnsafeCanvasReadback,
    id: 'canvas-readback-integrity',
    severity: 'warning',
    titleKey: 'launchpad.browserIssues.canvasReadbackIntegrity.title',
    workarounds: [
      {
        copyableValue: 'helium://flags/#helium-noise-canvas',
        id: 'helium',
        instructionKey: 'launchpad.browserIssues.canvasReadbackIntegrity.heliumWorkaround',
      },
      {
        id: 'other-browser',
        instructionKey: 'launchpad.browserIssues.canvasReadbackIntegrity.genericWorkaround',
      },
    ],
  },
] as const satisfies readonly KnownBrowserIssue[];

export const detectKnownBrowserIssues = async (
  environment: BrowserIssueDetectionEnvironment = DEFAULT_DETECTION_ENVIRONMENT
): Promise<readonly KnownBrowserIssue[]> => {
  const results: (KnownBrowserIssue | null)[] = await Promise.all(
    KNOWN_BROWSER_ISSUES.map(async (issue): Promise<KnownBrowserIssue | null> =>
      (await issue.detect(environment)) ? issue : null
    )
  );

  return results.filter((issue): issue is KnownBrowserIssue => issue !== null);
};
