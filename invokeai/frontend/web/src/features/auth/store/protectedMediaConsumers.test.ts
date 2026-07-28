import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

const readConsumer = (relativePath: string) =>
  readFileSync(fileURLToPath(new URL(relativePath, import.meta.url)), 'utf8');

describe('protected media consumers', () => {
  it.each([
    [
      'video field thumbnails',
      '../../nodes/components/flow/nodes/Invocation/fields/inputs/VideoFieldInputComponent.tsx',
      'src={videoDTO.thumbnail_url}',
    ],
    [
      'video frame scrubbers',
      '../../nodes/components/flow/nodes/Invocation/fields/IntegerField/VideoFrameIndexFieldInput.tsx',
      'videoUrl={videoDTO.video_url}',
    ],
    [
      'reference image previews',
      '../../controlLayers/components/RefImage/RefImagePreview.tsx',
      'src={imageDTO.image_url}',
    ],
  ])('%s refreshes its URL after the media cookie changes', (_name, relativePath, rawUsage) => {
    const source = readConsumer(relativePath);

    expect(source).toContain("from 'features/auth/store/mediaCookieRefresh'");
    expect(source).toContain('useMediaUrl(');
    expect(source).not.toContain(rawUsage);
  });

  it('waits for media-cookie self-heal before opening the video metadata link', () => {
    const source = readConsumer('../../gallery/components/ImageMetadataViewer/VideoMetadataViewer.tsx');

    expect(source).toContain('openMediaInNewTab');
    expect(source).not.toContain('href={video.video_url}');
    expect(source).not.toContain('href={videoUrl}');
  });

  it('opens the video metadata link only for primary or middle clicks', () => {
    const source = readConsumer('../../gallery/components/ImageMetadataViewer/VideoMetadataViewer.tsx');

    expect(source).toContain('event.button === 1');
  });

  it('does not give the protected video link a misleading hash target', () => {
    const source = readConsumer('../../gallery/components/ImageMetadataViewer/VideoMetadataViewer.tsx');

    expect(source).not.toContain('href="#"');
  });
});
