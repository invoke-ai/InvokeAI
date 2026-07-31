/* eslint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { SelectTriggerProps, SelectValueChangeDetails } from '@chakra-ui/react';
import type { AspectRatioId } from '@features/generation/core/types';

import { createListCollection, HStack, Text } from '@chakra-ui/react';
import { ASPECT_RATIO_MAP, ASPECT_RATIO_OPTIONS, isAspectRatioId } from '@features/generation/core/settings';
import { IconButton, Select, Tooltip } from '@platform/ui';
import { LockIcon, LockOpenIcon } from 'lucide-react';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { AspectRatioPreview } from './AspectRatioPreview';

type ControlSize = 'xs' | 'sm';

type AspectRatioOption = { id: AspectRatioId; ratio: number };

const SELECT_CONTENT_PROPS = { maxH: '18rem' } as const;

/** `Free` has no fixed ratio, so its thumbnail mirrors whatever the caller is currently showing. */
const getAspectRatioOptionRatio = (id: AspectRatioId, fallbackRatio: number): number =>
  id === 'Free' ? fallbackRatio : ASPECT_RATIO_MAP[id].ratio;

export interface AspectRatioSelectProps {
  /** Ratio used to draw the `Free` thumbnail — normally the live width/height. */
  fallbackRatio: number;
  size?: ControlSize;
  triggerProps?: SelectTriggerProps;
  value: AspectRatioId;
  onChange: (id: AspectRatioId) => void;
}

/**
 * The aspect-ratio preset dropdown, with a proportion thumbnail beside each
 * option. Presentational: callers own what a preset change does to their
 * dimensions, which differs between a free-floating size and a placed frame.
 */
export const AspectRatioSelect = ({
  fallbackRatio,
  size = 'xs',
  triggerProps,
  value,
  onChange,
}: AspectRatioSelectProps) => {
  const { t } = useTranslation();
  const options = useMemo<AspectRatioOption[]>(
    () => ASPECT_RATIO_OPTIONS.map((id) => ({ id, ratio: getAspectRatioOptionRatio(id, fallbackRatio) })),
    [fallbackRatio]
  );
  const collection = useMemo(
    () =>
      createListCollection({
        itemToString: (item: AspectRatioOption) => item.id,
        itemToValue: (item: AspectRatioOption) => item.id,
        items: options,
      }),
    [options]
  );
  const activePreviewRatio = getAspectRatioOptionRatio(value, fallbackRatio);
  const selectValue = useMemo(() => [value], [value]);

  const handleValueChange = ({ value: next }: SelectValueChangeDetails<AspectRatioOption>) => {
    const id = next[0];

    if (isAspectRatioId(id)) {
      onChange(id);
    }
  };

  return (
    <Select
      aria-label={t('widgets.generate.aspectRatio')}
      collection={collection}
      contentProps={SELECT_CONTENT_PROPS}
      flex="1"
      renderItem={(option) => (
        <HStack as="span" gap="2">
          <AspectRatioPreview boxSize="6" ratio={option.ratio} />
          <Text as="span" fontSize="xs">
            {option.id}
          </Text>
        </HStack>
      )}
      size={size}
      triggerProps={triggerProps}
      value={selectValue}
      valueText={
        <HStack as="span" gap="2" minW="0">
          <AspectRatioPreview boxSize="5" ratio={activePreviewRatio} />
          <Text as="span" fontSize="xs" truncate>
            {value}
          </Text>
        </HStack>
      }
      onValueChange={handleValueChange}
    />
  );
};

export interface AspectRatioLockButtonProps {
  isLocked: boolean;
  size?: ControlSize;
  onToggle: () => void;
}

/** Companion lock for {@link AspectRatioSelect}: a labelled icon button, not a bare dot. */
export const AspectRatioLockButton = ({ isLocked, size = 'xs', onToggle }: AspectRatioLockButtonProps) => {
  const { t } = useTranslation();
  const label = isLocked ? t('widgets.generate.unlockAspectRatio') : t('widgets.generate.lockAspectRatio');

  return (
    <Tooltip content={label}>
      <IconButton aria-label={label} size={size} variant={isLocked ? 'solid' : 'outline'} onClick={onToggle}>
        {isLocked ? <LockIcon /> : <LockOpenIcon />}
      </IconButton>
    </Tooltip>
  );
};
