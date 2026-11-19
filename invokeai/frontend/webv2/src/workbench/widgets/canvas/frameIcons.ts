import { createLucideIcon } from 'lucide-react';

/** Viewfinder corners around a diamond; the corners are Lucide `Scan` 1.46.0's paths, inlined. */
export const FitFrameToMasksIcon = createLucideIcon({
  name: 'fit-frame-to-masks',
  node: [
    ['path', { d: 'M3 7V5a2 2 0 0 1 2-2h2', key: 'scan-top-left' }],
    ['path', { d: 'M17 3h2a2 2 0 0 1 2 2v2', key: 'scan-top-right' }],
    ['path', { d: 'M21 17v2a2 2 0 0 1-2 2h-2', key: 'scan-bottom-right' }],
    ['path', { d: 'M7 21H5a2 2 0 0 1-2-2v-2', key: 'scan-bottom-left' }],
    ['path', { d: 'M12 7.5 16.5 12 12 16.5 7.5 12Z', key: 'diamond' }],
  ],
});
