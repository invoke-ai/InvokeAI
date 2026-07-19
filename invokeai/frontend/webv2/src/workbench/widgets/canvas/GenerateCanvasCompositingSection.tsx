/**
 * Canvas compositing settings for the generate widget.
 *
 * Denoising strength USED to live here; it moved to the layers panel's
 * Photoshop-style header (`widgets/layers/LayersPanelHeader.tsx`). This section
 * holds the rest of the legacy canvas compositing controls — infill method,
 * coherence pass mode / edge size / min denoise, and mask blur — persisted
 * per-project in the canvas widget's own `state.values` (same store as
 * `denoisingStrength`), read back by `prepareCanvasInvocation` and threaded into
 * the pure graph compiler. Rendered only when the active invocation source is
 * the canvas (the generate widget is shared with the Generate tab).
 */

import type { NumberInput as ChakraNumberInput, SelectValueChangeDetails } from '@chakra-ui/react';
import type { CanvasCoherenceMode, CanvasInfillMethod } from '@workbench/widgets/canvas/invoke/canvasCompositing';

import { createListCollection, NumberInput, Stack } from '@chakra-ui/react';
import { GenerationSettingsSection } from '@features/generation/settings-ui';
import { Field, Select } from '@platform/ui';
import {
  CANVAS_COHERENCE_EDGE_SIZE_MAX,
  CANVAS_COMPOSITING_KEYS,
  CANVAS_MASK_BLUR_MAX,
  readCanvasCompositingSettings,
} from '@workbench/widgets/canvas/invoke/canvasCompositing';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const INFILL_METHODS: readonly CanvasInfillMethod[] = ['patchmatch', 'lama', 'cv2', 'color', 'tile'];
const COHERENCE_MODES: readonly CanvasCoherenceMode[] = ['Gaussian Blur', 'Box Blur', 'Staged'];

const SELECT_POSITIONING = { placement: 'bottom-end', sameWidth: false } as const;

const selectCanvasValues = (project: Parameters<typeof getProjectWidgetValues>[0]): Record<string, unknown> =>
  getProjectWidgetValues(project, 'canvas');

export const GenerateCanvasCompositingSection = () => {
  const { t } = useTranslation();
  const { widgets } = useWorkbenchCommands();
  const values = useActiveProjectSelector(selectCanvasValues);
  const settings = useMemo(() => readCanvasCompositingSettings(values), [values]);

  const patch = useCallback(
    (partial: Record<string, unknown>) => {
      widgets.patchValues('canvas', partial);
    },
    [widgets]
  );

  const infillCollection = useMemo(
    () =>
      createListCollection({
        items: INFILL_METHODS.map((method) => ({
          label: t(`widgets.generate.compositingOptions.infillMethods.${method}`),
          value: method,
        })),
      }),
    [t]
  );

  const coherenceCollection = useMemo(
    () =>
      createListCollection({
        items: COHERENCE_MODES.map((mode) => ({
          label: t(`widgets.generate.compositingOptions.coherenceModes.${mode}`),
          value: mode,
        })),
      }),
    [t]
  );

  const infillValue = useMemo(() => [settings.infillMethod], [settings.infillMethod]);
  const coherenceValue = useMemo(() => [settings.coherenceMode], [settings.coherenceMode]);

  const handleInfillChange = useCallback(
    ({ value }: SelectValueChangeDetails) => {
      const method = value[0] as CanvasInfillMethod | undefined;
      if (method) {
        patch({ [CANVAS_COMPOSITING_KEYS.infillMethod]: method });
      }
    },
    [patch]
  );

  const handleCoherenceModeChange = useCallback(
    ({ value }: SelectValueChangeDetails) => {
      const mode = value[0] as CanvasCoherenceMode | undefined;
      if (mode) {
        patch({ [CANVAS_COMPOSITING_KEYS.coherenceMode]: mode });
      }
    },
    [patch]
  );

  const handleEdgeSizeChange = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        patch({ [CANVAS_COMPOSITING_KEYS.coherenceEdgeSize]: valueAsNumber });
      }
    },
    [patch]
  );

  const handleMinDenoiseChange = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        patch({ [CANVAS_COMPOSITING_KEYS.coherenceMinDenoise]: valueAsNumber });
      }
    },
    [patch]
  );

  const handleMaskBlurChange = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        patch({ [CANVAS_COMPOSITING_KEYS.maskBlur]: valueAsNumber });
      }
    },
    [patch]
  );

  const opt = (key: string) => t(`widgets.generate.compositingOptions.${key}`);

  return (
    <GenerationSettingsSection label={t('widgets.generate.compositing')}>
      <Stack gap="2" p="2">
        <Field label={opt('infillMethod')}>
          <Select
            aria-label={opt('infillMethod')}
            collection={infillCollection}
            positioning={SELECT_POSITIONING}
            size="xs"
            value={infillValue}
            valueText={opt(`infillMethods.${settings.infillMethod}`)}
            onValueChange={handleInfillChange}
          />
        </Field>
        <Field label={opt('coherenceMode')}>
          <Select
            aria-label={opt('coherenceMode')}
            collection={coherenceCollection}
            positioning={SELECT_POSITIONING}
            size="xs"
            value={coherenceValue}
            valueText={opt(`coherenceModes.${settings.coherenceMode}`)}
            onValueChange={handleCoherenceModeChange}
          />
        </Field>
        <Field label={opt('coherenceEdgeSize')}>
          <NumberInput.Root
            max={CANVAS_COHERENCE_EDGE_SIZE_MAX}
            min={0}
            size="xs"
            step={1}
            value={String(settings.coherenceEdgeSize)}
            onValueChange={handleEdgeSizeChange}
          >
            <NumberInput.Control />
            <NumberInput.Input aria-label={opt('coherenceEdgeSize')} />
          </NumberInput.Root>
        </Field>
        <Field label={opt('coherenceMinDenoise')}>
          <NumberInput.Root
            max={1}
            min={0}
            size="xs"
            step={0.01}
            value={settings.coherenceMinDenoise.toFixed(2)}
            onValueChange={handleMinDenoiseChange}
          >
            <NumberInput.Control />
            <NumberInput.Input aria-label={opt('coherenceMinDenoise')} />
          </NumberInput.Root>
        </Field>
        <Field label={opt('maskBlur')}>
          <NumberInput.Root
            max={CANVAS_MASK_BLUR_MAX}
            min={0}
            size="xs"
            step={1}
            value={String(settings.maskBlur)}
            onValueChange={handleMaskBlurChange}
          >
            <NumberInput.Control />
            <NumberInput.Input aria-label={opt('maskBlur')} />
          </NumberInput.Root>
        </Field>
      </Stack>
    </GenerationSettingsSection>
  );
};
