import { describe, expect, it } from 'vitest';

import { formatGalleryMediaCounts } from './mediaCounts';

const NOUNS: Record<string, string> = {
  'widgets.gallery.assetCount': 'upload',
  'widgets.gallery.imageCount': 'image',
  'widgets.gallery.videoCount': 'video',
};

const t = (key: string, values: Record<string, number | string>) => {
  if (key === 'widgets.gallery.mediaCountsTwo') {
    return `${values.first} and ${values.second}`;
  }
  if (key === 'widgets.gallery.mediaCountsThree') {
    return `${values.first}, ${values.second} and ${values.third}`;
  }

  const count = Number(values.count);

  return `${count} ${NOUNS[key]}${count === 1 ? '' : 's'}`;
};

describe('formatGalleryMediaCounts', () => {
  it('leaves out kinds with nothing to count', () => {
    expect(formatGalleryMediaCounts(t, { images: 2, videos: 0 })).toBe('2 images');
    expect(formatGalleryMediaCounts(t, { videos: 1 })).toBe('1 video');
  });

  it('joins the kinds it counts with the translated list words', () => {
    expect(formatGalleryMediaCounts(t, { images: 2, videos: 1 })).toBe('2 images and 1 video');
    expect(formatGalleryMediaCounts(t, { uploads: 4, videos: 1 })).toBe('1 video and 4 uploads');
    expect(formatGalleryMediaCounts(t, { images: 1, uploads: 3, videos: 2 })).toBe('1 image, 2 videos and 3 uploads');
  });

  it('says nothing when every count is zero', () => {
    expect(formatGalleryMediaCounts(t, { images: 0, videos: 0 })).toBe('');
  });
});
