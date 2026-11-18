import { writePsd } from 'ag-psd';

import type { PsdSerializationRequest, PsdSerializationResponse } from './psdExport';

// Pixel-backed default writes do not call createCanvas/createImageData. No DOM/OffscreenCanvas adapter is needed.
self.onmessage = ({ data }: MessageEvent<PsdSerializationRequest>): void => {
  try {
    const bytes = writePsd(data.psd);
    self.postMessage({ id: data.id, bytes } satisfies PsdSerializationResponse, { transfer: [bytes] });
  } catch (error) {
    self.postMessage({ id: data.id, error: String(error) } satisfies PsdSerializationResponse);
  }
};

self.onmessageerror = (): void => {
  self.postMessage({ id: 1, error: 'PSD serialization request could not be read.' } satisfies PsdSerializationResponse);
};
