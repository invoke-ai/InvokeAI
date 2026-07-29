import type { ImageProps } from '@chakra-ui/react';
import type { PromptTemplateRecord } from '@features/generation/data/promptTemplates';
import type { ReactNode } from 'react';

import { Image } from '@chakra-ui/react';
import { promptTemplateImageQueryOptions } from '@features/generation/data/promptTemplates';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';

const blobKeys = new WeakMap<Blob, number>();
let nextBlobKey = 1;
const IMAGE_OUTLINE_PROPS = { outline: '1px solid {colors.border.image}', outlineOffset: '-1px' };

const getBlobKey = (blob: Blob): number => {
  const existing = blobKeys.get(blob);

  if (existing !== undefined) {
    return existing;
  }

  const key = nextBlobKey;
  nextBlobKey += 1;
  blobKeys.set(blob, key);
  return key;
};

interface PromptTemplateImageProps extends Omit<ImageProps, 'src'> {
  fallback: ReactNode;
  /**
   * Undefined means "show the stored image"; a URL previews a local
   * replacement, and null explicitly hides a removed image.
   */
  localPreviewUrl?: string | null;
  template: Pick<PromptTemplateRecord, 'hasImage' | 'id'>;
}

export const PromptTemplateImage = ({
  fallback,
  localPreviewUrl,
  template,
  ...imageProps
}: PromptTemplateImageProps) => {
  const hasLocalOverride = localPreviewUrl !== undefined;
  const query = useQuery({
    ...promptTemplateImageQueryOptions(template.id),
    enabled: template.hasImage && !hasLocalOverride,
    retry: false,
  });

  if (hasLocalOverride) {
    return localPreviewUrl ? <Image {...IMAGE_OUTLINE_PROPS} {...imageProps} src={localPreviewUrl} /> : fallback;
  }

  if (!query.data) {
    return fallback;
  }

  return <BlobImage key={getBlobKey(query.data)} blob={query.data} fallback={fallback} imageProps={imageProps} />;
};

const BlobImage = ({
  blob,
  fallback,
  imageProps,
}: {
  blob: Blob;
  fallback: ReactNode;
  imageProps: Omit<ImageProps, 'src'>;
}) => {
  const [src, setSrc] = useState<string | null>(null);

  useMountEffect(() => {
    const objectUrl = URL.createObjectURL(blob);
    setSrc(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  });

  return src ? <Image {...IMAGE_OUTLINE_PROPS} {...imageProps} src={src} /> : fallback;
};
