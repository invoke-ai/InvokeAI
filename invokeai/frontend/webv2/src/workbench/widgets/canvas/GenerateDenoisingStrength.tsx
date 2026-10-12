import type { CanvasDenoisingStrength, CanvasDenoisingStrengthProps } from '@features/generation/react';
import type { ReactNode } from 'react';

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
import { createContext, use, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { DenoisingStrengthWave } from './DenoisingStrengthWave';

const STRENGTH_DEBOUNCE_MS = 250;

const formatStrengthPercent = (value: number): string => `${Math.round(value * 100)}%`;

const selectCanvasStrength = (project: Parameters<typeof getProjectWidgetValues>[0]): number =>
  readCanvasDenoisingStrength(getProjectWidgetValues(project, 'canvas'));

interface StrengthDraft {
  value: number;
  setValue: (value: number) => void;
}

const StrengthDraftContext = createContext<StrengthDraft | null>(null);

const useStrengthDraft = (): StrengthDraft => {
  const draft = use(StrengthDraftContext);
  if (!draft) {
    throw new Error('Denoising strength controls render inside GenerateDenoisingStrength.');
  }
  return draft;
};

/**
 * Owns the strength draft: persists it in canvas widget values after a debounce and flushes it with the Generate
 * form before invocation. Its `children` keep their element identity across draft changes, so a scrub re-renders
 * only the badge and field that read the draft, not the section laid out around them.
 */
const StrengthDraftProvider = ({ children }: { children: ReactNode }) => {
  const { widgets } = useWorkbenchCommands();
  const projectId = useActiveProjectSelector((project) => project.id);
  const strength = useActiveProjectSelector(selectCanvasStrength);

  const commitStrength = useCallback(
    (value: number) => {
      widgets.patchValues('canvas', { [CANVAS_DENOISING_STRENGTH_KEY]: clampCanvasDenoisingStrength(value) });
    },
    [widgets]
  );

  const { draftValue, flushDraftValue, setDraftValue } = useDebouncedDraftValue({
    delayMs: STRENGTH_DEBOUNCE_MS,
    onCommit: commitStrength,
    resetKey: projectId,
    value: strength,
  });

  useRegisterGenerateDraftFlusher(flushDraftValue);

  const draft = useMemo<StrengthDraft>(
    () => ({ setValue: setDraftValue, value: draftValue }),
    [draftValue, setDraftValue]
  );

  return <StrengthDraftContext value={draft}>{children}</StrengthDraftContext>;
};

const StrengthBadge = () => {
  const { value } = useStrengthDraft();
  return (
    <Badge gap="1.5">
      {formatStrengthPercent(value)}
      <DenoisingStrengthWave value={value} />
    </Badge>
  );
};

const StrengthField = () => {
  const { t } = useTranslation();
  const { setValue, value } = useStrengthDraft();
  return (
    <ScrubberField
      defaultValue={DEFAULT_CANVAS_DENOISING_STRENGTH}
      formatValue={formatStrengthPercent}
      label={t('widgets.generate.denoisingStrength')}
      max={MAX_CANVAS_DENOISING_STRENGTH}
      min={MIN_CANVAS_DENOISING_STRENGTH}
      step={0.01}
      value={value}
      onChange={setValue}
    />
  );
};

const STRENGTH_SLOTS: CanvasDenoisingStrength = { badges: <StrengthBadge />, field: <StrengthField /> };

/** Lays out the Render section with canvas denoising strength: its badge in the header, its control under guidance. */
export const GenerateDenoisingStrength = ({ children }: CanvasDenoisingStrengthProps) => (
  <StrengthDraftProvider>{children(STRENGTH_SLOTS)}</StrengthDraftProvider>
);
