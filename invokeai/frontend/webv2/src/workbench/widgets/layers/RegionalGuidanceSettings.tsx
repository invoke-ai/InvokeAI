import type { SelectValueChangeDetails } from '@chakra-ui/react';
import type { ArchitectureCapabilitiesSnapshot } from '@features/generation/runtime';
import type {
  CanvasMaskContract,
  CanvasMaskFillContract,
  CanvasRegionalGuidanceLayerContract,
  CanvasRegionalLayoutContract,
} from '@workbench/canvas-engine/api';
import type { CanvasStructuralEngine } from '@workbench/widgets/layers/layerOps';
import type { ChangeEvent } from 'react';

import {
  createListCollection,
  HStack,
  IconButton,
  Stack,
  Switch,
  Text,
  Textarea,
  VisuallyHidden,
} from '@chakra-ui/react';
import { PromptTextarea } from '@features/generation/components';
import { getLayoutRegionIssue, getPromptSyntaxProfile, getRegionalGuidanceSupport } from '@features/generation/graph';
import { getArchitectureCapabilitiesSnapshot, subscribeArchitectureCapabilities } from '@features/generation/runtime';
import { useRegisterDraftFlusher } from '@platform/react/draftRegistry';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { Button, ColorPicker, Field, Select, Tooltip } from '@platform/ui';
import { SegmentedControl } from '@platform/ui/SegmentedControl';
import { CANVAS_REGIONAL_LAYOUT_KINDS } from '@workbench/canvas-engine/api';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { useNotify } from '@workbench/useNotify';
import { armMaskTintTarget } from '@workbench/widgets/canvas/color-system/maskTintTarget';
import { type ColorSamplerEngine, useColorSampler } from '@workbench/widgets/canvas/useColorSampler';
import {
  baselineConfig,
  type CanvasPreparedEngine,
  type PreparedCommitOutcome,
  reportMaskEdit,
  useStructuralPreview,
} from '@workbench/widgets/canvas/useStructuralCommit';
import { PaletteIcon } from 'lucide-react';
import { useCallback, useId, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { useSelectedModelBase } from './useSelectedModelBase';

/** The regional-guidance fields patchable via `updateCanvasLayerConfig`. */
interface RegionalConfigPatch {
  mask?: Partial<CanvasMaskContract>;
  positivePrompt?: string | null;
  negativePrompt?: string | null;
  autoNegative?: boolean;
  layout?: CanvasRegionalLayoutContract | null;
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
  // Null while not editing, so undo and edits from elsewhere show through and only typed text is ever committed.
  const [positiveDraft, setPositiveDraft] = useState<string | null>(null);
  const [negativeDraft, setNegativeDraft] = useState<string | null>(null);
  const positivePrompt = positiveDraft ?? layer.positivePrompt ?? '';
  const negativePrompt = negativeDraft ?? layer.negativePrompt ?? '';

  const syntaxProfile = getPromptSyntaxProfile(base);
  const attentionSyntax = syntaxProfile === 'none' ? 'none' : 'edit';
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

  // A previewed fill records from where its preview started; any other change from the live layer.
  const commitConfig = useCallback(
    (label: string, next: RegionalConfigPatch): PreparedCommitOutcome => {
      const config = { layerType: 'regional_guidance', ...next } as const;
      return commitPrepared(label, (model, baseline) =>
        model.prepare({ before: baselineConfig(baseline, config), config, id: layer.id, type: 'patch-config' })
      );
    },
    [commitPrepared, layer.id]
  );

  // Commits typed prompt drafts that differ from the layer, on blur and when Invoke (Ctrl+Enter included) flushes.
  const commitPrompts = useCallback(() => {
    if (positiveDraft === null && negativeDraft === null) {
      return;
    }
    const positive = positiveDraft === null ? layer.positivePrompt : positiveDraft || null;
    const negative = negativeDraft === null ? layer.negativePrompt : negativeDraft || null;
    const patch: RegionalConfigPatch = {
      ...(positive === layer.positivePrompt ? {} : { positivePrompt: positive }),
      ...(negative === layer.negativePrompt ? {} : { negativePrompt: negative }),
    };
    const outcome =
      'positivePrompt' in patch
        ? commitConfig(t('widgets.layers.regionalGuidance.positivePrompt'), patch)
        : 'negativePrompt' in patch
          ? commitConfig(t('widgets.layers.regionalGuidance.negativePrompt'), patch)
          : null;
    // A refused edit (already reported) keeps the typed text in the field instead of discarding it.
    if (outcome === null || outcome.status === 'committed' || outcome.status === 'unchanged') {
      setPositiveDraft(null);
      setNegativeDraft(null);
    }
  }, [commitConfig, layer.negativePrompt, layer.positivePrompt, negativeDraft, positiveDraft, t]);
  useRegisterDraftFlusher(commitPrompts);

  const handlePositiveChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) => setPositiveDraft(event.currentTarget.value),
    []
  );

  const handleNegativeChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) => setNegativeDraft(event.currentTarget.value),
    []
  );

  const handleAutoNegative = useCallback(
    (details: { checked: boolean }) => {
      commitConfig(t('widgets.layers.regionalGuidance.autoNegative'), { autoNegative: details.checked });
    },
    [commitConfig, t]
  );

  // Layout bases read the region's role and, for text, its exact text; the stored role survives model switches.
  const isLayout = support?.mode === 'layout';
  const layoutKind = layer.layout?.kind ?? 'object';
  const storedVisibleText = layer.layout?.text;
  // Null while not editing, so undo and edits from elsewhere show through.
  const [visibleTextDraft, setVisibleTextDraft] = useState<string | null>(null);
  const visibleText = visibleTextDraft ?? storedVisibleText ?? '';
  const layoutKindFieldId = useId();
  const promptIssueId = useId();
  const visibleTextIssueId = useId();

  const commitLayout = useCallback(
    (label: string, next: CanvasRegionalLayoutContract) =>
      // A plain object role is stored as absence, like a layer that never had one.
      commitConfig(label, { layout: next.kind === 'object' && next.text === undefined ? null : next }),
    [commitConfig]
  );

  const layoutKindOptions = useMemo(
    () =>
      CANVAS_REGIONAL_LAYOUT_KINDS.map((kind) => ({
        label: t(`widgets.layers.regionalGuidance.layoutKinds.${kind}`),
        value: kind,
      })),
    [t]
  );

  const handleLayoutKindChange = useCallback(
    (value: string) => {
      const kind = CANVAS_REGIONAL_LAYOUT_KINDS.find((candidate) => candidate === value);
      if (kind && kind !== layoutKind) {
        commitLayout(t('widgets.layers.regionalGuidance.layoutKind'), {
          kind,
          ...(storedVisibleText === undefined ? {} : { text: storedVisibleText }),
        });
      }
    },
    [commitLayout, layoutKind, storedVisibleText, t]
  );

  const commitVisibleText = useCallback(() => {
    if (visibleTextDraft === null) {
      return;
    }
    const text = visibleTextDraft.length > 0 ? visibleTextDraft : undefined;
    if (text === storedVisibleText) {
      setVisibleTextDraft(null);
      return;
    }
    const outcome = commitLayout(t('widgets.layers.regionalGuidance.visibleText'), {
      kind: layoutKind,
      ...(text === undefined ? {} : { text }),
    });
    // A refused edit (already reported) keeps the typed text in the field instead of discarding it.
    if (outcome.status === 'committed' || outcome.status === 'unchanged') {
      setVisibleTextDraft(null);
    }
  }, [commitLayout, layoutKind, storedVisibleText, t, visibleTextDraft]);
  // Invoke flushes drafts first, so text typed but not yet blurred still reaches the graph.
  useRegisterDraftFlusher(commitVisibleText);

  const handleVisibleTextChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) => setVisibleTextDraft(event.currentTarget.value),
    []
  );

  // Generation skips a region the layout cannot place. The warning sits under the missing value and describes its
  // control; a status region that exists before any warning announces each change.
  const layoutIssue =
    isLayout && support
      ? getLayoutRegionIssue({ layout: { kind: layoutKind, text: visibleText }, positivePrompt })
      : null;
  const layoutIssueMessage = layoutIssue
    ? t(
        layoutIssue === 'missing_prompt'
          ? 'widgets.layers.regionalGuidance.layoutMissingPrompt'
          : 'widgets.layers.regionalGuidance.layoutMissingText',
        { model: support?.label }
      )
    : null;

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
    (next: CanvasMaskFillContract) => commitConfig(t('widgets.layers.maskFill.fill'), { mask: { fill: next } }),
    [commitConfig, t]
  );

  const handleColorChange = useCallback(
    (hex: string) => {
      previewStructural({
        config: { layerType: 'regional_guidance', mask: { fill: { ...fill, color: hex } } },
        id: layer.id,
        type: 'updateCanvasLayerConfig',
      });
    },
    [previewStructural, fill, layer.id]
  );

  const handleArmTint = useCallback(() => armMaskTintTarget(layer.id), [layer.id]);
  const handleColorChangeEnd = useCallback((hex: string) => commitFill({ ...fill, color: hex }), [commitFill, fill]);

  const handleStyleChange = useCallback(
    ({ value }: SelectValueChangeDetails) => {
      const style = value[0] as CanvasMaskFillContract['style'] | undefined;
      if (style && style !== fill.style) {
        commitFill({ ...fill, style });
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
        <Text color="fg.warning" fontSize="xs" role="alert">
          {t('widgets.layers.regionalGuidance.unsupportedModel')}
        </Text>
      ) : null}
      {isLayout && support ? (
        <Field
          helpText={t('widgets.layers.regionalGuidance.layoutKindHelp', { model: support.label })}
          id={layoutKindFieldId}
          label={t('widgets.layers.regionalGuidance.layoutKind')}
        >
          <SegmentedControl
            aria-describedby={`${layoutKindFieldId}-help`}
            aria-labelledby={`${layoutKindFieldId}-label`}
            options={layoutKindOptions}
            value={layoutKind}
            onChange={handleLayoutKindChange}
          />
        </Field>
      ) : null}
      <Field label={t('widgets.layers.regionalGuidance.positivePrompt')}>
        <PromptTextarea
          aria-describedby={layoutIssue === 'missing_prompt' ? promptIssueId : undefined}
          aria-label={t('widgets.layers.regionalGuidance.positivePrompt')}
          attentionSyntax={attentionSyntax}
          syntaxProfile={syntaxProfile}
          defaultHeightPx={REGIONAL_PROMPT_HEIGHT_PX}
          minHeightPx={REGIONAL_PROMPT_HEIGHT_PX}
          placeholder={t('widgets.layers.regionalGuidance.positivePromptPlaceholder')}
          resizeHandleAriaLabel={t('widgets.layers.regionalGuidance.positivePrompt')}
          showSyntaxHighlighting={showSyntaxHighlighting}
          size="lg"
          value={positivePrompt}
          onBlur={commitPrompts}
          onChange={handlePositiveChange}
        />
      </Field>
      {layoutIssue === 'missing_prompt' ? (
        <Text color="fg.warning" fontSize="xs" id={promptIssueId}>
          {layoutIssueMessage}
        </Text>
      ) : null}
      {isLayout && layoutKind === 'text' ? (
        <Field label={t('widgets.layers.regionalGuidance.visibleText')}>
          <Textarea
            aria-describedby={layoutIssue === 'missing_text' ? visibleTextIssueId : undefined}
            autoComplete="off"
            autoresize
            placeholder={t('widgets.layers.regionalGuidance.visibleTextPlaceholder')}
            rows={1}
            value={visibleText}
            onBlur={commitVisibleText}
            onChange={handleVisibleTextChange}
          />
        </Field>
      ) : null}
      {layoutIssue === 'missing_text' ? (
        <Text color="fg.warning" fontSize="xs" id={visibleTextIssueId}>
          {layoutIssueMessage}
        </Text>
      ) : null}
      {isLayout ? <VisuallyHidden role="status">{layoutIssueMessage}</VisuallyHidden> : null}
      {showNegativeControls && (
        <Field label={t('widgets.layers.regionalGuidance.negativePrompt')}>
          <PromptTextarea
            aria-label={t('widgets.layers.regionalGuidance.negativePrompt')}
            attentionSyntax={attentionSyntax}
            syntaxProfile={syntaxProfile}
            defaultHeightPx={REGIONAL_PROMPT_HEIGHT_PX}
            minHeightPx={REGIONAL_PROMPT_HEIGHT_PX}
            placeholder={t('widgets.layers.regionalGuidance.negativePromptPlaceholder')}
            resizeHandleAriaLabel={t('widgets.layers.regionalGuidance.negativePrompt')}
            showSyntaxHighlighting={showSyntaxHighlighting}
            size="lg"
            value={negativePrompt}
            onBlur={commitPrompts}
            onChange={handleNegativeChange}
          />
        </Field>
      )}
      {showNegativeControls && (
        <Switch.Root checked={layer.autoNegative} onCheckedChange={handleAutoNegative}>
          <Switch.HiddenInput />
          <Switch.Control>
            <Switch.Thumb />
          </Switch.Control>
          <Switch.Label>
            <Text fontSize="md">{t('widgets.layers.regionalGuidance.autoNegative')}</Text>
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
            size="sm"
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
            value={styleValue}
            valueText={t(`widgets.layers.maskFill.styles.${fill.style}`)}
            onValueChange={handleStyleChange}
          />
        </Field>
      </HStack>
      <Button disabled={!engine} variant="outline" onClick={handleInvert}>
        {t('widgets.layers.maskFill.invert')}
      </Button>
    </Stack>
  );
};
