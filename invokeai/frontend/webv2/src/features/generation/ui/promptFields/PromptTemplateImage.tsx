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
    return localPreviewUrl ? <Image {...imageProps} src={localPreviewUrl} /> : fallback;
  }

  if (!query.data) {
    return fallback;
  }

  return <BlobImage key={getBlobKey(query.data)} blob={query.data} imageProps={imageProps} />;
};

const BlobImage = ({ blob, imageProps }: { blob: Blob; imageProps: Omit<ImageProps, 'src'> }) => {
  const [src] = useState(() => URL.createObjectURL(blob));

  useMountEffect(() => () => URL.revokeObjectURL(src));

  return <Image {...imageProps} src={src} />;
};
