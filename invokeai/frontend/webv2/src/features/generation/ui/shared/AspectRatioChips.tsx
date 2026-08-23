/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { AspectRatioId } from '@features/generation/core/types';

import { Flex, Text } from '@chakra-ui/react';
import { ASPECT_RATIO_MAP, ASPECT_RATIO_OPTIONS } from '@features/generation/core/settings';
import { Button } from '@platform/ui';
import { useTranslation } from 'react-i18next';

import { AspectRatioPreview } from './AspectRatioPreview';

/** `Free` has no fixed ratio, so its glyph mirrors whatever the caller is currently showing. */
const getRatioGlyph = (id: AspectRatioId, fallbackRatio: number): number =>
  id === 'Free' ? fallbackRatio : ASPECT_RATIO_MAP[id].ratio;

export interface AspectRatioChipsProps {
  /** Ratio used to draw the `Free` glyph — normally the live width/height. */
  fallbackRatio: number;
  value: AspectRatioId;
  onChange: (id: AspectRatioId) => void;
}

/**
 * Shape-first aspect ratio picker: a wrapped run of chips whose glyphs *are*
 * the ratios. Every preset is visible — nothing hides behind an overflow menu.
 */
export const AspectRatioChips = ({ fallbackRatio, value, onChange }: AspectRatioChipsProps) => {
  const { t } = useTranslation();

  return (
    <Flex aria-label={t('widgets.generate.aspectRatio')} flexWrap="wrap" gap="1" role="group">
      {ASPECT_RATIO_OPTIONS.map((id) => (
        <Button
          key={id}
          aria-label={id}
          aria-pressed={value === id}
          gap="1"
          px="1.5"
          size="2xs"
          variant={value === id ? 'solid' : 'outline'}
          onClick={() => onChange(id)}
        >
          <AspectRatioPreview boxSize="3.5" ratio={getRatioGlyph(id, fallbackRatio)} />
          <Text as="span" fontSize="2xs">
            {id}
          </Text>
        </Button>
      ))}
    </Flex>
  );
};
