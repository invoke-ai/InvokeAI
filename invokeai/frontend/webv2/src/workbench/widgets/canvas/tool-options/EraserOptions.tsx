import { HStack } from '@chakra-ui/react';
import { useEraserOptions } from '@workbench/widgets/canvas/engineStoreHooks';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { ToolOptionsComponentProps } from './ToolOptionsBar';

import { clampBrushSize, PaintSizeOpacityControls } from './BrushOptions';

/** Eraser tool options: size (slider + numeric) and opacity. */
export const EraserOptions = ({ engine }: ToolOptionsComponentProps) => {
  const { t } = useTranslation();
  const options = useEraserOptions(engine);

  const setSize = useCallback(
    (size: number) => engine.interaction.set('eraserOptions', { ...options, size: clampBrushSize(size) }),
    [engine, options]
  );

  const setOpacity = useCallback(
    (opacity: number) => engine.interaction.set('eraserOptions', { ...options, opacity }),
    [engine, options]
  );

  return (
    <HStack align="center" gap="3">
      <PaintSizeOpacityControls
        opacity={options.opacity}
        setOpacity={setOpacity}
        setSize={setSize}
        size={options.size}
        sizeLabel={t('widgets.canvas.toolOptions.eraserSize')}
      />
    </HStack>
  );
};
