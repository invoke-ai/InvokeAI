import type { SelectValueChangeDetails } from '@chakra-ui/react';
import type { MarqueeToolOptions, SelectionOp } from '@workbench/canvas-engine/api';

import { createListCollection, HStack, Text } from '@chakra-ui/react';
import { Select } from '@platform/ui';
import { useMarqueeOptions } from '@workbench/widgets/canvas/engineStoreHooks';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { ToolOptionsComponentProps } from './ToolOptionsBar';

import { SelectionOptionsRow } from './SelectionOptionsRow';

type MarqueeKind = MarqueeToolOptions['kind'];

const SELECT_POSITIONING = { placement: 'top-start', sameWidth: false } as const;
const SELECT_TRIGGER_PROPS = { minW: '6rem' } as const;

/**
 * Marquee tool options: the shape the next drag traces (rectangle / ellipse),
 * plus the shared selection controls bound to the marquee's own op mode.
 */
export const MarqueeOptions = ({ engine }: ToolOptionsComponentProps) => {
  const { t } = useTranslation();
  const options = useMarqueeOptions(engine);

  const kindCollection = useMemo(
    () =>
      createListCollection<{ label: string; value: MarqueeKind }>({
        items: [
          { label: t('widgets.canvas.toolOptions.shapeRect'), value: 'rect' },
          { label: t('widgets.canvas.toolOptions.shapeEllipse'), value: 'ellipse' },
        ],
      }),
    [t]
  );
  const kindValue = useMemo(() => [options.kind], [options.kind]);

  const onKindChange = useCallback(
    ({ value }: SelectValueChangeDetails<{ label: string; value: MarqueeKind }>) => {
      const next = value[0] as MarqueeKind | undefined;
      if (next && next !== options.kind) {
        engine.interaction.set('marqueeOptions', { ...options, kind: next });
      }
    },
    [engine, options]
  );

  const onModeChange = useCallback(
    (mode: SelectionOp) => engine.interaction.set('marqueeOptions', { ...options, mode }),
    [engine, options]
  );

  return (
    <HStack align="center" gap="3">
      <HStack align="center" gap="1.5">
        <Text color="fg.muted" fontSize="2xs">
          {t('widgets.canvas.toolOptions.marqueeKind')}
        </Text>
        <Select
          collection={kindCollection}
          positioning={SELECT_POSITIONING}
          size="xs"
          triggerProps={SELECT_TRIGGER_PROPS}
          value={kindValue}
          onValueChange={onKindChange}
        />
      </HStack>
      <SelectionOptionsRow
        engine={engine}
        hintKey="widgets.canvas.toolOptions.marqueeHint"
        mode={options.mode}
        onModeChange={onModeChange}
      />
    </HStack>
  );
};
