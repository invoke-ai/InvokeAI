import type { TextProps } from '@invoke-ai/ui-library';
import { Text } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import { DROP_SHADOW } from './common';

type Props = TextProps & {
  type: 'first' | 'second';
};

export const ImageComparisonLabel = memo(({ type, ...rest }: Props) => {
  const { t } = useTranslation();
  return (
    <Text
      position="absolute"
      bottom={4}
      insetInlineEnd={type === 'first' ? undefined : 4}
      insetInlineStart={type === 'first' ? 4 : undefined}
      textOverflow="clip"
      whiteSpace="nowrap"
      filter={DROP_SHADOW}
      color="base.50"
      transitionDuration="0.2s"
      transitionProperty="common"
      {...rest}
    >
      {type === 'first' ? t('gallery.viewerImage') : t('gallery.compareImage')}
    </Text>
  );
});

ImageComparisonLabel.displayName = 'ImageComparisonLabel';
