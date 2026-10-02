/**
 * Expose canvas-only infill/coherence/mask controls in Generate. Persist them in canvas values for
 * prepareCanvasInvocation and graph compilation.
 */

import type { NumberInput as ChakraNumberInput, SelectValueChangeDetails } from '@chakra-ui/react';
import type { CanvasCoherenceMode, CanvasInfillMethod } from '@workbench/widgets/canvas/invoke/canvasCompositing';

import { Badge, createListCollection, NumberInput, Stack } from '@chakra-ui/react';
import { GenerationSettingsSection } from '@features/generation/components';
import { ColorPicker, Field, formatHexColor, parseHexColor, Select } from '@platform/ui';
import { getCanvasEngine } from '@workbench/canvas-operations/api';
import {
  CANVAS_COHERENCE_EDGE_SIZE_MAX,
  CANVAS_COMPOSITING_KEYS,
  CANVAS_MASK_BLUR_MAX,
  readCanvasCompositingSettings,
} from '@workbench/widgets/canvas/invoke/canvasCompositing';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectId, useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const INFILL_METHODS: readonly CanvasInfillMethod[] = ['patchmatch', 'lama', 'cv2', 'color', 'tile'];
const COHERENCE_MODES: readonly CanvasCoherenceMode[] = ['Gaussian Blur', 'Box Blur', 'Staged'];

// Match full-width menus to the trigger width and start edge.
const SELECT_POSITIONING = { placement: 'bottom-start', sameWidth: true } as const;

const selectCanvasValues = (project: Parameters<typeof getProjectWidgetValues>[0]): Record<string, unknown> =>
  getProjectWidgetValues(project, 'canvas');

export const GenerateCanvasCompositingSection = () => {
  const { t } = useTranslation();
  const { widgets } = useWorkbenchCommands();
  const projectId = useActiveProjectId();
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

  const infillColor = formatHexColor(settings.infillColorValue, { alpha: true });

  const handleInfillColorChange = useCallback(
    (color: string) => {
      patch({ [CANVAS_COMPOSITING_KEYS.infillColorValue]: parseHexColor(color) });
    },
    [patch]
  );

  // The generate form outlives the canvas widget's engine lease, so the sampler
  // peeks non-owningly at click time instead of holding an engine alive.
  const handleSampleInfillColor = useCallback(
    () => getCanvasEngine(projectId)?.tools.requestColorSample() ?? Promise.resolve(null),
    [projectId]
  );

  const opt = (key: string) => t(`widgets.generate.compositingOptions.${key}`);

  const badges = (
    <>
      <Badge size="xs">{t(`widgets.generate.compositingOptions.infillMethods.${settings.infillMethod}`)}</Badge>
      <Badge size="xs">{settings.maskBlur}px</Badge>
    </>
  );

  return (
    <GenerationSettingsSection
      badges={badges}
      defaultOpen={false}
      label={t('widgets.generate.compositing')}
      sectionId="canvas-fill"
    >
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
        {settings.infillMethod === 'color' ? (
          <Field label={opt('infillColorValue')}>
            <ColorPicker
              aria-label={opt('infillColorValue')}
              value={infillColor}
              withAlpha
              onSampleColor={handleSampleInfillColor}
              onValueChange={handleInfillColorChange}
            />
          </Field>
        ) : null}
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
