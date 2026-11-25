import type { SelectValueChangeDetails } from '@chakra-ui/react';
import type { ArchitectureCapabilitiesSnapshot } from '@features/models/architecturesStore';
import type {
  CanvasMaskContract,
  CanvasMaskFillContract,
  CanvasMaskGuidanceContract,
  CanvasRegionalGuidanceLayerContract,
  CanvasRegionalLayoutContract,
} from '@workbench/canvas-engine/api';
import type { CanvasStructuralEngine } from '@workbench/widgets/layers/layerOps';
import type { ChangeEvent, ComponentProps } from 'react';

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
import { AddNegativePromptButton, PromptTextarea, RemoveNegativePromptButton } from '@features/generation/components';
import {
  areRegionalNegativesActive,
  getEffectiveLayoutKind,
  getLayoutRegionIssue,
  getRegionalGuidanceSupport,
  supportsPromptAttentionSyntax,
} from '@features/generation/graph';
import {
  getArchitectureCapabilitiesSnapshot,
  subscribeArchitectureCapabilities,
} from '@features/models/architecturesStore';
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
  usePreparedCommit,
  useStructuralPreview,
} from '@workbench/widgets/canvas/useStructuralCommit';
import { PaletteIcon } from 'lucide-react';
import { useCallback, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { useSelectedCfgScale, useSelectedModelBase } from './useSelectedModelBase';

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

interface RegionalPromptValue {
  positivePrompt: string | null;
  negativePrompt: string | null;
  autoNegative: boolean;
}

type RegionalPromptCommit = (label: string, patch: Partial<RegionalPromptValue>) => PreparedCommitOutcome;

type PromptFieldProps = ComponentProps<typeof PromptTextarea>;

interface RegionalNegativeControlsProps {
  addRef: (element: HTMLElement | null) => void;
  autoNegative: boolean;
  disabled: boolean;
  /** Null until the region has a negative prompt; its add button stands in. */
  field: PromptFieldProps | null;
  ignoredAtCfg: boolean;
  onAdd: () => void;
  onAutoNegativeChange: (details: { checked: boolean }) => void;
  onRemove: () => void;
}

/** Prompt drafts, their commits and the selected model's regional support, for a guidance layer and a guided mask. */
const useRegionalPromptEditor = ({
  commit,
  disabled,
  value,
}: {
  /** One undo step; a refusal (already reported) keeps the typed text. */
  commit: RegionalPromptCommit;
  disabled: boolean;
  value: RegionalPromptValue;
}) => {
  const { t } = useTranslation();
  const base = useSelectedModelBase();
  const cfgScale = useSelectedCfgScale();
  const showSyntaxHighlighting = useWorkbenchPreferenceSelector(
    (preferences) => preferences.showPromptSyntaxHighlighting
  );
  // Null while not editing, so undo and edits from elsewhere show through and only typed text is ever committed.
  const [positiveDraft, setPositiveDraft] = useState<string | null>(null);
  const [negativeDraft, setNegativeDraft] = useState<string | null>(null);
  const positivePrompt = positiveDraft ?? value.positivePrompt ?? '';

  // Resolve support inside the capability selector to update memoized answers.
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
  // Hide negatives only when backend policy rejects them, retaining values for other models; offer controls while no
  // model/policy is known.
  const showNegativeControls = !hasCapabilities || support?.negativePrompt !== false;
  const unsupportedModel = hasCapabilities && base !== null && support === null;
  // Controls stay editable so values carry over; the hint says why they do nothing at this CFG Scale.
  const negativesIgnoredAtCfg =
    support !== null && cfgScale !== null && support.negativePrompt && !areRegionalNegativesActive(support, cfgScale);

  // Commits typed prompt drafts that differ from the stored ones, on blur and when Invoke (Ctrl+Enter included) flushes.
  const commitPrompts = useCallback(() => {
    if (positiveDraft === null && negativeDraft === null) {
      return;
    }
    const positive = positiveDraft === null ? value.positivePrompt : positiveDraft || null;
    // Clearing the text keeps the negative prompt; only its remove control takes it out.
    const negative = negativeDraft === null || value.negativePrompt === null ? value.negativePrompt : negativeDraft;
    const patch: Partial<RegionalPromptValue> = {
      ...(positive === value.positivePrompt ? {} : { positivePrompt: positive }),
      ...(negative === value.negativePrompt ? {} : { negativePrompt: negative }),
    };
    const outcome =
      'positivePrompt' in patch
        ? commit(t('widgets.layers.regionalGuidance.positivePrompt'), patch)
        : 'negativePrompt' in patch
          ? commit(t('widgets.layers.regionalGuidance.negativePrompt'), patch)
          : null;
    // A refused edit (already reported) keeps the typed text in the field instead of discarding it.
    if (outcome === null || outcome.status === 'committed' || outcome.status === 'unchanged') {
      setPositiveDraft(null);
      setNegativeDraft(null);
    }
  }, [commit, negativeDraft, positiveDraft, t, value.negativePrompt, value.positivePrompt]);
  useRegisterDraftFlusher(commitPrompts);

  const handlePositiveChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) => setPositiveDraft(event.currentTarget.value),
    []
  );

  const handleNegativeChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) => setNegativeDraft(event.currentTarget.value),
    []
  );

  // Adding or removing the prompt swaps the field and its add button; focus follows to whichever one mounts.
  const pendingNegativeFocus = useRef<'add' | 'field' | null>(null);
  const requestNegativeFocus = useCallback((target: 'add' | 'field') => {
    pendingNegativeFocus.current = target;
    requestAnimationFrame(() => {
      if (pendingNegativeFocus.current === target) {
        pendingNegativeFocus.current = null;
      }
    });
  }, []);
  const focusWhenPending = useCallback(
    (target: 'add' | 'field') => (element: HTMLElement | null) => {
      if (element && pendingNegativeFocus.current === target) {
        pendingNegativeFocus.current = null;
        element.focus();
      }
    },
    []
  );
  const negativeFieldRef = useMemo(() => focusWhenPending('field'), [focusWhenPending]);
  const addNegativeRef = useMemo(() => focusWhenPending('add'), [focusWhenPending]);

  const handleAddNegative = useCallback(() => {
    requestNegativeFocus('field');
    const outcome = commit(t('widgets.layers.regionalGuidance.addNegativePrompt'), { negativePrompt: '' });
    if (outcome.status === 'committed') {
      // Text left unsaved in a prompt that was undone away belongs to that prompt, not this new empty one.
      setNegativeDraft(null);
    } else {
      pendingNegativeFocus.current = null;
    }
  }, [commit, requestNegativeFocus, t]);

  const handleRemoveNegative = useCallback(() => {
    requestNegativeFocus('add');
    const outcome = commit(t('widgets.layers.regionalGuidance.removeNegativePrompt'), { negativePrompt: null });
    if (outcome.status === 'committed') {
      setNegativeDraft(null);
    } else {
      pendingNegativeFocus.current = null;
    }
  }, [commit, requestNegativeFocus, t]);

  const handleAutoNegative = useCallback(
    (details: { checked: boolean }) => {
      commit(t('widgets.layers.regionalGuidance.autoNegative'), { autoNegative: details.checked });
    },
    [commit, t]
  );

  const fieldProps = {
    attentionSyntax: supportsPromptAttentionSyntax(base) ? 'edit' : 'none',
    defaultHeightPx: REGIONAL_PROMPT_HEIGHT_PX,
    minHeightPx: REGIONAL_PROMPT_HEIGHT_PX,
    showSyntaxHighlighting,
    size: 'lg',
    onBlur: commitPrompts,
  } satisfies Partial<PromptFieldProps>;
  const negative: RegionalNegativeControlsProps = {
    addRef: addNegativeRef,
    autoNegative: value.autoNegative,
    disabled,
    // A region has a negative prompt once one is added, even while it is empty; null means none, and generation reads
    // an empty one as none too.
    field:
      value.negativePrompt === null
        ? null
        : {
            ...fieldProps,
            'aria-label': t('widgets.layers.regionalGuidance.negativePrompt'),
            placeholder: t('widgets.layers.regionalGuidance.negativePromptPlaceholder'),
            resizeHandleAriaLabel: t('widgets.layers.regionalGuidance.negativePrompt'),
            textareaRef: negativeFieldRef,
            value: negativeDraft ?? value.negativePrompt,
            onChange: handleNegativeChange,
          },
    ignoredAtCfg: negativesIgnoredAtCfg,
    onAdd: handleAddNegative,
    onAutoNegativeChange: handleAutoNegative,
    onRemove: handleRemoveNegative,
  };
  const positiveField: PromptFieldProps = {
    ...fieldProps,
    'aria-label': t('widgets.layers.regionalGuidance.positivePrompt'),
    placeholder: t('widgets.layers.regionalGuidance.positivePromptPlaceholder'),
    resizeHandleAriaLabel: t('widgets.layers.regionalGuidance.positivePrompt'),
    value: positivePrompt,
    onChange: handlePositiveChange,
  };

  return { negative, positiveField, positivePrompt, showNegativeControls, support, unsupportedModel };
};

/** The negative prompt, or its add button, then Auto-Negative: siblings of the caller's other fields. */
const RegionalNegativeControls = ({
  addRef,
  autoNegative,
  disabled,
  field,
  ignoredAtCfg,
  onAdd,
  onAutoNegativeChange,
  onRemove,
}: RegionalNegativeControlsProps) => {
  const { t } = useTranslation();
  const fieldId = useId();
  const helpId = ignoredAtCfg ? `${fieldId}-help` : undefined;
  const labelEnd = useMemo(
    () => (
      <RemoveNegativePromptButton
        disabled={disabled}
        label={t('widgets.layers.regionalGuidance.removeNegativePrompt')}
        onClick={onRemove}
      />
    ),
    [disabled, onRemove, t]
  );

  return (
    <>
      {field ? (
        <Field
          helpText={ignoredAtCfg ? t('widgets.layers.regionalGuidance.negativesNeedCfg') : undefined}
          id={fieldId}
          label={t('widgets.layers.regionalGuidance.negativePrompt')}
          labelEnd={labelEnd}
        >
          <PromptTextarea aria-describedby={helpId} {...field} />
        </Field>
      ) : (
        <Stack gap="1.5">
          <AddNegativePromptButton
            ref={addRef}
            aria-describedby={helpId}
            aria-label={t('widgets.layers.regionalGuidance.addNegativePrompt')}
            disabled={disabled}
            label={t('widgets.layers.regionalGuidance.negativePromptAdd')}
            onClick={onAdd}
          />
          {/* The field's help line while there is no field, so Auto-Negative keeps its description. */}
          {ignoredAtCfg ? (
            <Text color="fg.muted" fontSize="xs" id={helpId}>
              {t('widgets.layers.regionalGuidance.negativesNeedCfg')}
            </Text>
          ) : null}
        </Stack>
      )}
      <Switch.Root checked={autoNegative} onCheckedChange={onAutoNegativeChange}>
        <Switch.HiddenInput aria-describedby={helpId} />
        <Switch.Control>
          <Switch.Thumb />
        </Switch.Control>
        <Switch.Label>
          <Text fontSize="md">{t('widgets.layers.regionalGuidance.autoNegative')}</Text>
        </Switch.Label>
      </Switch.Root>
    </>
  );
};

interface RegionalGuidanceSettingsProps {
  engine: (CanvasStructuralEngine & CanvasPreparedEngine & ColorSamplerEngine) | null;
  layer: CanvasRegionalGuidanceLayerContract;
}

/** Prompts, toggles and fill use config patches; invert uses engine pixels. */
export const RegionalGuidanceSettings = ({ engine, layer }: RegionalGuidanceSettingsProps) => {
  const { t } = useTranslation();
  const { commit: commitPrepared, preview: previewStructural } = useStructuralPreview(engine);
  const sampleColor = useColorSampler(engine);
  const fill = layer.mask.fill;

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

  const prompts = useRegionalPromptEditor({ commit: commitConfig, disabled: !engine, value: layer });
  const { positivePrompt, support } = prompts;

  // Layout bases read the region's role and, for text, its exact text; the stored role survives model switches, and a
  // base without it places the region as an object. A base with only one role offers no choice.
  const isLayout = support?.mode === 'layout';
  const layoutKinds = isLayout ? support.layoutKinds : [];
  const showLayoutKinds = layoutKinds.length > 1;
  const layoutKind = layer.layout?.kind ?? 'object';
  const effectiveLayoutKind = getEffectiveLayoutKind(layoutKinds, layer.layout);
  const storedVisibleText = layer.layout?.text;
  // Null while not editing, so undo and edits from elsewhere show through.
  const [visibleTextDraft, setVisibleTextDraft] = useState<string | null>(null);
  const visibleText = visibleTextDraft ?? storedVisibleText ?? '';
  const layoutKindFieldId = useId();
  const promptFieldId = useId();
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

  // Generation skips a region the layout cannot place.
  const layoutIssue = isLayout
    ? getLayoutRegionIssue({ layout: { kind: layoutKind, text: visibleText }, positivePrompt }, layoutKinds)
    : null;
  // The prompt field explains the box on every layout base; the region type's help explains only the roles.
  const promptHelp = isLayout
    ? t('widgets.layers.regionalGuidance.layoutBoxHelp', { model: support.label })
    : undefined;
  const promptDescription =
    [promptHelp ? `${promptFieldId}-help` : null, layoutIssue === 'missing_prompt' ? promptIssueId : null]
      .filter(Boolean)
      .join(' ') || undefined;
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
      {prompts.unsupportedModel ? (
        <Text color="fg.warning" fontSize="xs" role="alert">
          {t('widgets.layers.regionalGuidance.unsupportedModel')}
        </Text>
      ) : null}
      {isLayout && showLayoutKinds ? (
        <Field
          helpText={t('widgets.layers.regionalGuidance.layoutKindHelp')}
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
      <Field helpText={promptHelp} id={promptFieldId} label={t('widgets.layers.regionalGuidance.positivePrompt')}>
        <PromptTextarea aria-describedby={promptDescription} {...prompts.positiveField} />
      </Field>
      {layoutIssue === 'missing_prompt' ? (
        <Text color="fg.warning" fontSize="xs" id={promptIssueId}>
          {layoutIssueMessage}
        </Text>
      ) : null}
      {isLayout && effectiveLayoutKind === 'text' ? (
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
      {prompts.showNegativeControls ? <RegionalNegativeControls {...prompts.negative} /> : null}

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

// The mask's Add button unmounts as this editor replaces its settings; focus follows to the prompt.
let promptFocusPending = false;

export const focusMaskGuidancePromptOnMount = (): void => {
  promptFocusPending = true;
  requestAnimationFrame(() => {
    promptFocusPending = false;
  });
};

const focusPendingPrompt = (element: HTMLElement | null): void => {
  if (element && promptFocusPending) {
    promptFocusPending = false;
    element.focus();
  }
};

/** An inpaint mask's regional prompts; its tree row owns enabling and removal. */
export const MaskGuidanceSettings = ({
  engine,
  guidance,
  layerId,
}: {
  engine: CanvasPreparedEngine | null;
  guidance: CanvasMaskGuidanceContract;
  layerId: string;
}) => {
  const { t } = useTranslation();
  const commitPrepared = usePreparedCommit(engine);

  const commit = useCallback<RegionalPromptCommit>(
    (label, patch) =>
      commitPrepared(label, (model) => {
        const live = model.getLayer(layerId);
        const current = live?.type === 'inpaint_mask' ? live.guidance : undefined;
        // Removed since this editor rendered: committing would bring it back.
        if (!current) {
          return { ids: [layerId], status: 'missing' };
        }
        return model.prepare({
          config: { guidance: { ...current, ...patch }, layerType: 'inpaint_mask' },
          id: layerId,
          type: 'patch-config',
        });
      }),
    [commitPrepared, layerId]
  );

  const prompts = useRegionalPromptEditor({ commit, disabled: !engine, value: guidance });
  const { support } = prompts;
  const helpId = useId();
  const promptFieldId = useId();
  const promptIssueId = useId();

  // A guided mask is always an object region: a layout base needs only its prompt.
  const layoutLabel = support?.mode === 'layout' ? support.label : null;
  const missingPrompt =
    layoutLabel !== null && getLayoutRegionIssue({ positivePrompt: prompts.positivePrompt }, []) !== null;
  const missingPromptMessage = missingPrompt
    ? t('widgets.layers.modifiers.guidanceLayoutMissingPrompt', { model: layoutLabel })
    : null;
  const promptHelp =
    layoutLabel !== null ? t('widgets.layers.modifiers.guidanceLayoutBoxHelp', { model: layoutLabel }) : undefined;
  const promptDescription = [helpId, promptHelp ? `${promptFieldId}-help` : null, missingPrompt ? promptIssueId : null]
    .filter(Boolean)
    .join(' ');

  return (
    <Stack gap="2">
      {prompts.unsupportedModel ? (
        <Text color="fg.warning" fontSize="xs" role="alert">
          {t('widgets.layers.modifiers.guidanceUnsupportedModel')}
        </Text>
      ) : null}
      <Text color="fg.muted" fontSize="xs" id={helpId}>
        {t('widgets.layers.modifiers.guidanceHelp')}
      </Text>
      <Field helpText={promptHelp} id={promptFieldId} label={t('widgets.layers.regionalGuidance.positivePrompt')}>
        <PromptTextarea
          aria-describedby={promptDescription}
          textareaRef={focusPendingPrompt}
          {...prompts.positiveField}
        />
      </Field>
      {missingPrompt ? (
        <Text color="fg.warning" fontSize="xs" id={promptIssueId}>
          {missingPromptMessage}
        </Text>
      ) : null}
      {layoutLabel !== null ? <VisuallyHidden role="status">{missingPromptMessage}</VisuallyHidden> : null}
      {prompts.showNegativeControls ? <RegionalNegativeControls {...prompts.negative} /> : null}
    </Stack>
  );
};
