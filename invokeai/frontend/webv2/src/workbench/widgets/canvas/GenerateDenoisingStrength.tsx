import { Badge, Box } from '@chakra-ui/react';
import { GenerationSettingsSection } from '@features/generation/components';
import { useDebouncedDraftValue, useRegisterGenerateDraftFlusher } from '@features/generation/react';
import { ScrubberField } from '@platform/ui/ScrubberField';
import {
  CANVAS_DENOISING_STRENGTH_KEY,
  clampCanvasDenoisingStrength,
  MAX_CANVAS_DENOISING_STRENGTH,
  MIN_CANVAS_DENOISING_STRENGTH,
  readCanvasDenoisingStrength,
} from '@workbench/widgets/canvas/invoke/canvasStrength';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { DenoisingStrengthWave } from './DenoisingStrengthWave';

const STRENGTH_DEBOUNCE_MS = 250;

const formatStrengthPercent = (value: number): string => `${Math.round(value * 100)}%`;

const selectCanvasStrength = (project: Parameters<typeof getProjectWidgetValues>[0]): number =>
  readCanvasDenoisingStrength(getProjectWidgetValues(project, 'canvas'));

/** Persist canvas denoising strength in widget values and flush its draft with the Generate form before invocation. */
export const GenerateDenoisingStrength = () => {
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

  const badges = useMemo(
    () => (
      <>
        <Badge size="xs">{formatStrengthPercent(draftStrength)}</Badge>
        <DenoisingStrengthWave value={draftStrength} />
      </>
    ),
    [draftStrength]
  );

  return (
    <GenerationSettingsSection
      badges={badges}
      defaultOpen
      label={t('widgets.generate.denoisingStrength')}
      sectionId="canvas-denoising"
    >
      <Box p="2">
        <ScrubberField
          formatValue={formatStrengthPercent}
          label={t('widgets.generate.strength')}
          max={MAX_CANVAS_DENOISING_STRENGTH}
          min={MIN_CANVAS_DENOISING_STRENGTH}
          step={0.01}
          value={draftStrength}
          onChange={setStrength}
        />
      </Box>
    </GenerationSettingsSection>
  );
};
