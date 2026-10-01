import type { SelectValueChangeDetails } from '@chakra-ui/react';
import type { ArchitectureCapabilitiesSnapshot } from '@features/generation/runtime';
import type {
  CanvasMaskContract,
  CanvasMaskFillContract,
  CanvasRegionalGuidanceLayerContract,
} from '@workbench/canvas-engine/api';
import type { CanvasStructuralEngine } from '@workbench/widgets/layers/layerOps';
import type { ChangeEvent, FocusEvent } from 'react';

import { createListCollection, HStack, IconButton, Stack, Switch, Text } from '@chakra-ui/react';
import { PROMPT_ATTENTION_TARGET_PROPS, PromptTextarea } from '@features/generation/components';
import { getRegionalGuidanceSupport } from '@features/generation/graph';
import { getArchitectureCapabilitiesSnapshot, subscribeArchitectureCapabilities } from '@features/generation/runtime';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { Button, ColorPicker, Field, Select, Tooltip } from '@platform/ui';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { useNotify } from '@workbench/useNotify';
import { armMaskTintTarget } from '@workbench/widgets/canvas/color-system/maskTintTarget';
import { type ColorSamplerEngine, useColorSampler } from '@workbench/widgets/canvas/useColorSampler';
import {
  type CanvasPreparedEngine,
  reportMaskEdit,
  useStructuralPreview,
} from '@workbench/widgets/canvas/useStructuralCommit';
import { PaletteIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { useSelectedModelBase } from './useSelectedModelBase';

/** The regional-guidance fields patchable via `updateCanvasLayerConfig`. */
interface RegionalConfigPatch {
  mask?: Partial<CanvasMaskContract>;
  positivePrompt?: string | null;
  negativePrompt?: string | null;
  autoNegative?: boolean;
}

/** The six mask fill styles, matching `CanvasMaskFillContract['style']` / legacy `zFillStyle`. */
const MASK_FILL_STYLES: readonly CanvasMaskFillContract['style'][] = [
  'solid',
  'grid',
  'crosshatch',
  'diagonal',
  'horizontal',
  'vertical',
];

const SELECT_POSITIONING = { placement: 'bottom-end', sameWidth: true } as const;

const REGIONAL_PROMPT_HEIGHT_PX = 72;

interface RegionalGuidanceSettingsProps {
  engine: (CanvasStructuralEngine & CanvasPreparedEngine & ColorSamplerEngine) | null;
  layer: CanvasRegionalGuidanceLayerContract;
}

/**
 * Edit regional prompts/toggles/fill through undoable config patches and invert through engine pixels. Reference
 * images have dedicated tree-child editors.
 */
export const RegionalGuidanceSettings = ({ engine, layer }: RegionalGuidanceSettingsProps) => {
  const { t } = useTranslation();
  const { commit: commitPrepared, preview: previewStructural } = useStructuralPreview(engine);
  const sampleColor = useColorSampler(engine);
  const base = useSelectedModelBase();
  const showSyntaxHighlighting = useWorkbenchPreferenceSelector(
    (preferences) => preferences.showPromptSyntaxHighlighting
  );
  const fillBeforeRef = useRef<CanvasMaskFillContract | null>(null);
  const [positivePrompt, setPositivePrompt] = useState(layer.positivePrompt ?? '');
  const [negativePrompt, setNegativePrompt] = useState(layer.negativePrompt ?? '');

  const fill = layer.mask.fill;
  // Resolve support inside the capability selector to update memoized answers. Distinguish unavailable policy from
  // unsupported models.
  const { hasCapabilities, support } = useExternalStoreSelector(
    subscribeArchitectureCapabilities,
    getArchitectureCapabilitiesSnapshot,
    useCallback(
      (snapshot: ArchitectureCapabilitiesSnapshot) => ({
        hasCapabilities: snapshot.revision > 0,
        support: getRegionalGuidanceSupport(base),
      }),
      [base]
    )
  );
  // Hide negatives only when backend policy rejects them, retaining values for other models; offer controls while
  // no model/policy is known.
  const showNegativeControls = !hasCapabilities || support?.negativePrompt !== false;
  const unsupportedModel = hasCapabilities && base !== null && support === null;

  const commitConfig = useCallback(
    (label: string, next: RegionalConfigPatch, before: RegionalConfigPatch) => {
      commitPrepared(label, (model) =>
        model.prepare({
          before: { layerType: 'regional_guidance', ...before },
          config: { layerType: 'regional_guidance', ...next },
          id: layer.id,
          type: 'patch-config',
        })
      );
    },
    [commitPrepared, layer.id]
  );

  const handlePositiveBlur = useCallback(
    (event: FocusEvent<HTMLTextAreaElement>) => {
      const value = event.target.value;
      const next = value.length > 0 ? value : null;
      if (next !== layer.positivePrompt) {
        commitConfig(
          t('widgets.layers.regionalGuidance.positivePrompt'),
          { positivePrompt: next },
          {
            positivePrompt: layer.positivePrompt,
          }
        );
      }
    },
    [commitConfig, layer.positivePrompt, t]
  );

  const handlePositiveChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) => setPositivePrompt(event.currentTarget.value),
    []
  );

  const handleNegativeBlur = useCallback(
    (event: FocusEvent<HTMLTextAreaElement>) => {
      const value = event.target.value;
      const next = value.length > 0 ? value : null;
      if (next !== layer.negativePrompt) {
        commitConfig(
          t('widgets.layers.regionalGuidance.negativePrompt'),
          { negativePrompt: next },
          {
            negativePrompt: layer.negativePrompt,
          }
        );
      }
    },
    [commitConfig, layer.negativePrompt, t]
  );

  const handleNegativeChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) => setNegativePrompt(event.currentTarget.value),
    []
  );

  const handleAutoNegative = useCallback(
    (details: { checked: boolean }) => {
      commitConfig(
        t('widgets.layers.regionalGuidance.autoNegative'),
        { autoNegative: details.checked },
        { autoNegative: layer.autoNegative }
      );
    },
    [commitConfig, layer.autoNegative, t]
  );

  const styleCollection = useMemo(
    () =>
      createListCollection({
        items: MASK_FILL_STYLES.map((style) => ({
          label: t(`widgets.layers.maskFill.styles.${style}`),
          value: style,
        })),
      }),
    [t]
  );

  const commitFill = useCallback(
    (next: CanvasMaskFillContract, before: CanvasMaskFillContract) => {
      commitConfig(t('widgets.layers.maskFill.fill'), { mask: { fill: next } }, { mask: { fill: before } });
    },
    [commitConfig, t]
  );

  const handleColorChange = useCallback(
    (hex: string) => {
      if (
        !previewStructural({
          config: { layerType: 'regional_guidance', mask: { fill: { ...fill, color: hex } } },
          id: layer.id,
          type: 'updateCanvasLayerConfig',
        })
      ) {
        return;
      }
      if (fillBeforeRef.current === null) {
        fillBeforeRef.current = fill;
      }
    },
    [previewStructural, fill, layer.id]
  );

  const handleArmTint = useCallback(() => armMaskTintTarget(layer.id), [layer.id]);
  const handleColorChangeEnd = useCallback(
    (hex: string) => {
      const before = fillBeforeRef.current ?? fill;
      fillBeforeRef.current = null;
      commitFill({ ...before, color: hex }, before);
    },
    [commitFill, fill]
  );

  const handleStyleChange = useCallback(
    ({ value }: SelectValueChangeDetails) => {
      const style = value[0] as CanvasMaskFillContract['style'] | undefined;
      if (style && style !== fill.style) {
        commitFill({ ...fill, style }, fill);
      }
    },
    [commitFill, fill]
  );

  const notify = useNotify();
  const handleInvert = useCallback(() => {
    if (engine) {
      reportMaskEdit(engine.layers.invertMask(layer.id), notify.error, t);
    }
  }, [engine, layer.id, notify, t]);

  const styleValue = useMemo(() => [fill.style], [fill.style]);

  return (
    <Stack gap="2">
      {unsupportedModel ? (
        <Text color="fg.warning" fontSize="2xs" role="alert">
          {t('widgets.layers.regionalGuidance.unsupportedModel')}
        </Text>
      ) : null}
      <Field label={t('widgets.layers.regionalGuidance.positivePrompt')}>
        <PromptTextarea
          {...PROMPT_ATTENTION_TARGET_PROPS}
          aria-label={t('widgets.layers.regionalGuidance.positivePrompt')}
          defaultHeightPx={REGIONAL_PROMPT_HEIGHT_PX}
          minHeightPx={REGIONAL_PROMPT_HEIGHT_PX}
          placeholder={t('widgets.layers.regionalGuidance.positivePromptPlaceholder')}
          resizeHandleAriaLabel={t('widgets.layers.regionalGuidance.positivePrompt')}
          showSyntaxHighlighting={showSyntaxHighlighting}
          size="sm"
          value={positivePrompt}
          onBlur={handlePositiveBlur}
          onChange={handlePositiveChange}
        />
      </Field>
      {showNegativeControls && (
        <Field label={t('widgets.layers.regionalGuidance.negativePrompt')}>
          <PromptTextarea
            {...PROMPT_ATTENTION_TARGET_PROPS}
            aria-label={t('widgets.layers.regionalGuidance.negativePrompt')}
            defaultHeightPx={REGIONAL_PROMPT_HEIGHT_PX}
            minHeightPx={REGIONAL_PROMPT_HEIGHT_PX}
            placeholder={t('widgets.layers.regionalGuidance.negativePromptPlaceholder')}
            resizeHandleAriaLabel={t('widgets.layers.regionalGuidance.negativePrompt')}
            showSyntaxHighlighting={showSyntaxHighlighting}
            size="sm"
            value={negativePrompt}
            onBlur={handleNegativeBlur}
            onChange={handleNegativeChange}
          />
        </Field>
      )}
      {showNegativeControls && (
        <Switch.Root checked={layer.autoNegative} size="sm" onCheckedChange={handleAutoNegative}>
          <Switch.HiddenInput />
          <Switch.Control>
            <Switch.Thumb />
          </Switch.Control>
          <Switch.Label>
            <Text fontSize="xs">{t('widgets.layers.regionalGuidance.autoNegative')}</Text>
          </Switch.Label>
        </Switch.Root>
      )}

      <HStack gap="2">
        <Field flexShrink="0" label={t('widgets.layers.maskFill.color')}>
          <ColorPicker
            aria-label={t('widgets.layers.maskFill.color')}
            value={fill.color}
            onSampleColor={sampleColor}
            onValueChange={handleColorChange}
            onValueChangeEnd={handleColorChangeEnd}
          />
        </Field>
        <Tooltip content={t('widgets.layers.maskFill.editInColorPane')}>
          <IconButton
            aria-label={t('widgets.layers.maskFill.editInColorPane')}
            alignSelf="flex-end"
            color="fg.muted"
            size="2xs"
            variant="ghost"
            onClick={handleArmTint}
          >
            <PaletteIcon size={14} />
          </IconButton>
        </Tooltip>
        <Field flex="1" label={t('widgets.layers.maskFill.style')} minW="0">
          <Select
            aria-label={t('widgets.layers.maskFill.style')}
            collection={styleCollection}
            positioning={SELECT_POSITIONING}
            size="xs"
            value={styleValue}
            valueText={t(`widgets.layers.maskFill.styles.${fill.style}`)}
            onValueChange={handleStyleChange}
          />
        </Field>
      </HStack>
      <Button disabled={!engine} size="xs" variant="outline" onClick={handleInvert}>
        {t('widgets.layers.maskFill.invert')}
      </Button>
    </Stack>
  );
};
