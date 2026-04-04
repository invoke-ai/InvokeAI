import { getFocusedRegion, setFocusedRegion } from 'common/hooks/focus';
import { describe, expect, it } from 'vitest';

import { handleImageMetadataViewerPointerDown } from './ImageMetadataViewer';

describe('ImageMetadataViewer', () => {
  it('claims the viewer focus region on pointer down', () => {
    setFocusedRegion('workflowEditor');

    expect(getFocusedRegion()).toBe('workflowEditor');

    handleImageMetadataViewerPointerDown();

    expect(getFocusedRegion()).toBe('viewer');

    setFocusedRegion(null);
  });
});
