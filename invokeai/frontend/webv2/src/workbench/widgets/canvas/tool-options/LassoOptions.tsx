import type { SelectValueChangeDetails } from '@chakra-ui/react';
import type { LassoToolOptions, SelectionOp } from '@workbench/canvas-engine/api';

import { createListCollection, HStack, Text } from '@chakra-ui/react';
import { Select } from '@platform/ui';
import { useLassoOptions } from '@workbench/widgets/canvas/engineStoreHooks';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { ToolOptionsComponentProps } from './ToolOptionsBar';

import { SelectionOptionsRow } from './SelectionOptionsRow';

type LassoShape = LassoToolOptions['shape'];

const SELECT_POSITIONING = { placement: 'top-start', sameWidth: false } as const;
const SELECT_TRIGGER_PROPS = { minW: '6rem' } as const;

/**
 * Lasso tool options: how the path is drawn (freehand drag / polygon clicks),
 * plus the shared selection controls bound to the lasso's own op mode.
 */
export const LassoOptions = ({ engine }: ToolOptionsComponentProps) => {
  const { t } = useTranslation();
  const options = useLassoOptions(engine);

  const shapeCollection = useMemo(
    () =>
      createListCollection<{ label: string; value: LassoShape }>({
        items: [
          { label: t('widgets.canvas.toolOptions.lassoFreehand'), value: 'freehand' },
          { label: t('widgets.canvas.toolOptions.lassoPolygon'), value: 'polygon' },
        ],
      }),
    [t]
  );
  const shapeValue = useMemo(() => [options.shape], [options.shape]);

  const onShapeChange = useCallback(
    ({ value }: SelectValueChangeDetails<{ label: string; value: LassoShape }>) => {
      const next = value[0] as LassoShape | undefined;
      if (next && next !== options.shape) {
        engine.interaction.set('lassoOptions', { ...options, shape: next });
      }
    },
    [engine, options]
  );

  const onModeChange = useCallback(
    (mode: SelectionOp) => engine.interaction.set('lassoOptions', { ...options, mode }),
    [engine, options]
  );

  return (
    <HStack align="center" gap="3">
      <HStack align="center" gap="1.5">
        <Text color="fg.muted" fontSize="2xs">
          {t('widgets.canvas.toolOptions.lassoShape')}
        </Text>
        <Select
          collection={shapeCollection}
          positioning={SELECT_POSITIONING}
          size="xs"
          triggerProps={SELECT_TRIGGER_PROPS}
          value={shapeValue}
          onValueChange={onShapeChange}
        />
      </HStack>
      <SelectionOptionsRow
        engine={engine}
        hintKey={
          options.shape === 'polygon'
            ? 'widgets.canvas.toolOptions.lassoPolygonHint'
            : 'widgets.canvas.toolOptions.lassoHint'
        }
        mode={options.mode}
        onModeChange={onModeChange}
      />
    </HStack>
  );
};
