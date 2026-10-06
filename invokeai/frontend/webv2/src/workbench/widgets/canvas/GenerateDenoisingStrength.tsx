import type { CanvasDenoisingStrengthProps } from '@features/generation/react';

import { Badge } from '@chakra-ui/react';
import { useDebouncedDraftValue, useRegisterGenerateDraftFlusher } from '@features/generation/react';
import { ScrubberField } from '@platform/ui/ScrubberField';
import {
  CANVAS_DENOISING_STRENGTH_KEY,
  clampCanvasDenoisingStrength,
  DEFAULT_CANVAS_DENOISING_STRENGTH,
  MAX_CANVAS_DENOISING_STRENGTH,
  MIN_CANVAS_DENOISING_STRENGTH,
  readCanvasDenoisingStrength,
} from '@workbench/widgets/canvas/invoke/canvasStrength';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { DenoisingStrengthWave } from './DenoisingStrengthWave';

const STRENGTH_DEBOUNCE_MS = 250;

const formatStrengthPercent = (value: number): string => `${Math.round(value * 100)}%`;

const selectCanvasStrength = (project: Parameters<typeof getProjectWidgetValues>[0]): number =>
  readCanvasDenoisingStrength(getProjectWidgetValues(project, 'canvas'));

/**
 * Lays out the Render section with canvas denoising strength: its badge and wave in the header, its control under
 * guidance. The strength persists in canvas widget values; its draft flushes with the Generate form before invocation.
 */
export const GenerateDenoisingStrength = ({ children }: CanvasDenoisingStrengthProps) => {
  const { t } = useTranslation();
  const { widgets } = useWorkbenchCommands();
  const projectId = useActiveProjectSelector((project) => project.id);
  const strength = useActiveProjectSelector(selectCanvasStrength);

  const commitStrength = useCallback(
    (value: number) => {
      widgets.patchValues('canvas', { [CANVAS_DENOISING_STRENGTH_KEY]: clampCanvasDenoisingStrength(value) });
    },
    [widgets]
  );

  const {
    draftValue: draftStrength,
    flushDraftValue,
    setDraftValue: setStrength,
  } = useDebouncedDraftValue({
    delayMs: STRENGTH_DEBOUNCE_MS,
    onCommit: commitStrength,
    resetKey: projectId,
    value: strength,
  });

  useRegisterGenerateDraftFlusher(flushDraftValue);

  return children({
    badges: (
      <Badge gap="1.5">
        {formatStrengthPercent(draftStrength)}
        <DenoisingStrengthWave value={draftStrength} />
      </Badge>
    ),
    field: (
      <ScrubberField
        defaultValue={DEFAULT_CANVAS_DENOISING_STRENGTH}
        formatValue={formatStrengthPercent}
        label={t('widgets.generate.denoisingStrength')}
        max={MAX_CANVAS_DENOISING_STRENGTH}
        min={MIN_CANVAS_DENOISING_STRENGTH}
        step={0.01}
        value={draftStrength}
        onChange={setStrength}
      />
    ),
  });
};
