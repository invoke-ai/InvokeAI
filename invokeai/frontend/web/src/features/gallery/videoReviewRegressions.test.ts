import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

const readSource = (relativePath: string) =>
  readFileSync(fileURLToPath(new URL(relativePath, import.meta.url)), 'utf8');

describe('video review regressions', () => {
  it('reports partial star and unstar failures centrally', () => {
    const source = readSource('../../services/api/endpoints/videos.ts');

    expect(source).toContain('result.failed_videos');
    expect(source).toContain('toast.videosFailedToUpdate');
  });

  it('waits for fanned-out video board mutations before clearing selection', () => {
    const source = readSource('../imageActions/actions.ts');

    expect(source).toContain('Promise.allSettled');
    expect(source).toContain('.unwrap()');
    expect(source).toMatch(/failed.*length === 0[\s\S]*selectionChanged/);
    expect(source).toContain('selectionChanged(failedVideoNames)');
  });

  it('keeps the change-board modal state when a video move fails', () => {
    const source = readSource('../changeBoardModal/components/ChangeBoardModal.tsx');

    expect(source).toContain('Promise.allSettled');
    expect(source).toContain('.unwrap()');
    expect(source).toMatch(/failed.*length === 0[\s\S]*changeBoardReset/);
    expect(source).toContain('videosToChangeSelected(failedVideoNames)');
  });

  it('makes gallery video buttons keyboard accessible', () => {
    const source = readSource('./components/ImageGrid/GalleryVideoItem.tsx');

    expect(source).toContain('tabIndex={0}');
    expect(source).toContain('aria-label={videoDTO.video_name}');
    expect(source).toContain('onKeyDown={onKeyDown}');
    expect(source).toContain("event.key === 'Enter'");
    expect(source).toContain("event.key === ' '");
  });

  it('renders an explicit fallback when both thumbnail sources fail', () => {
    const source = readSource('./components/ImageGrid/GalleryVideoThumbnail.tsx');

    expect(source).toContain('onError={onVideoError}');
    expect(source).toContain('IAINoContentFallback');
  });

  it('uses media wording and waits for cookie self-heal before downloading', () => {
    const hook = readSource('../../common/hooks/useDownloadImage.ts');
    const locale = readSource('../../../public/locales/en.json');

    expect(hook).toContain('waitForMediaCookieSelfHeal');
    expect(hook).toContain("t('toast.problemDownloadingMedia')");
    expect(hook).not.toContain("t('toast.problemDownloadingImage')");
    expect(locale).toContain('"problemDownloadingMedia": "Unable to Download Media"');
  });

  it('waits for cookie self-heal before opening protected video URLs', () => {
    const preview = readSource('./components/ImageViewer/CurrentVideoPreview.tsx');
    const contextMenu = readSource('./components/ContextMenu/MenuItems/ContextMenuItemOpenInNewTabVideo.tsx');

    expect(preview).toContain('openMediaInNewTab');
    expect(contextMenu).toContain('openMediaInNewTab');
    expect(preview).not.toContain('window.open(videoDTO.video_url');
    expect(contextMenu).not.toContain('window.open(videoDTO.video_url');
  });
});
