import type { FlexProps } from '@chakra-ui/react';
/* oxlint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { SelectObjectSaveTarget } from '@workbench/canvas-engine/engine';
import type { SamSessionError, SamSessionErrorCode, SamSessionSnapshot } from '@workbench/canvas-engine/engineStores';
import type { SamModel } from '@workbench/generation/canvas/samGraph';
import type { ChangeEvent } from 'react';

import {
  Flex,
  Group,
  Heading,
  IconButton,
  Input,
  Menu,
  NativeSelect,
  Portal,
  Spinner,
  Stack,
  Switch,
  Text,
  VisuallyHidden,
} from '@chakra-ui/react';
import { Button, MenuContent, Tooltip } from '@workbench/components/ui';
import { makeImageDurable } from '@workbench/gallery/api';
import { isSamDocumentInputValid } from '@workbench/generation/canvas/samGraph';
import { useSamSession } from '@workbench/widgets/canvas/engineStoreHooks';
import { ChevronDownIcon, InfoIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import type { ToolOptionsComponentProps } from './ToolOptionsBar';

import { CanvasOperationPanel } from './CanvasOperationPanel';

export interface SamActionEligibility {
  canApply: boolean;
  canCancel: boolean;
  canEditInputs: boolean;
  canProcess: boolean;
  canReset: boolean;
  canSave: boolean;
}

export interface SamPanelViewModel {
  bboxActive: boolean;
  excludeCount: number;
  includeCount: number;
  sourceLabel: string;
  sourceSummary: string;
}

export const SAM_COMPACT_CONTROL_LAYOUT = { h: '8', minH: '8', size: 'xs' } as const;
export const SAM_COMPACT_BUTTON_LAYOUT = { ...SAM_COMPACT_CONTROL_LAYOUT, px: '2' } as const;
export const SAM_COMPACT_FOOTER_LAYOUT = { flexWrap: 'wrap', gap: '1' } satisfies FlexProps;
export const SAM_COMPACT_GROUP_LAYOUT = { maxW: 'full', minW: '0' } as const;
export const SAM_MODEL_SELECT_LAYOUT = { flex: '0 1 11rem', maxW: 'full', minW: '0', w: '11rem' } as const;

const SAM_SEGMENT_BUTTON_LAYOUT = {
  ...SAM_COMPACT_BUTTON_LAYOUT,
  flex: '1 1 0',
  minW: '0',
  overflow: 'hidden',
} as const;

const SAM_PROMPT_GUIDANCE_ID = 'sam-prompt-guidance';
const SAM_VISUAL_GUIDANCE_ID = 'sam-visual-guidance';

const SAM_STATUS_TRANSLATION_KEYS: Record<SamSessionSnapshot['status'], string> = {
  committing: 'widgets.layers.selectObject.statusCommitting',
  error: 'widgets.layers.selectObject.statusError',
  'preparing-composite': 'widgets.layers.selectObject.statusPreparingComposite',
  'processing-sam': 'widgets.layers.selectObject.statusProcessingSam',
  ready: 'widgets.layers.selectObject.statusReady',
  'rendering-preview': 'widgets.layers.selectObject.statusRenderingPreview',
  scheduled: 'widgets.layers.selectObject.statusScheduled',
  uploading: 'widgets.layers.selectObject.statusUploading',
};

const SAM_ERROR_TRANSLATION_KEYS: Record<SamSessionErrorCode, string> = {
  decode: 'widgets.layers.selectObject.errorDecode',
  empty: 'widgets.layers.selectObject.errorEmpty',
  invalid: 'widgets.layers.selectObject.errorInvalid',
  locked: 'widgets.layers.selectObject.errorLocked',
  'no-output': 'widgets.layers.selectObject.errorNoOutput',
  'not-ready': 'widgets.layers.selectObject.errorNotReady',
  'output-dimension': 'widgets.layers.selectObject.errorOutputDimension',
  queue: 'widgets.layers.selectObject.errorQueue',
  reconcile: 'widgets.layers.selectObject.errorReconcile',
  unknown: 'widgets.layers.selectObject.errorUnknown',
  upload: 'widgets.layers.selectObject.errorUpload',
};

const SAVE_TARGETS: readonly SelectObjectSaveTarget[] = ['raster', 'control', 'inpaint_mask', 'regional_guidance'];

export const getSamStatusTranslationKey = (status: SamSessionSnapshot['status']): string =>
  SAM_STATUS_TRANSLATION_KEYS[status];

export const getSamErrorTranslationKey = (code: SamSessionErrorCode): string => SAM_ERROR_TRANSLATION_KEYS[code];

export const getSamPanelViewModel = (
  session: SamSessionSnapshot,
  formatSourceSummary: (width: number, height: number) => string,
  formatSourceLabel: (width: number, height: number) => string
): SamPanelViewModel => ({
  bboxActive: session.input.type === 'visual' && session.input.bbox !== null,
  excludeCount: session.input.type === 'visual' ? session.input.excludePoints.length : 0,
  includeCount: session.input.type === 'visual' ? session.input.includePoints.length : 0,
  sourceLabel: formatSourceLabel(session.sourceRect.width, session.sourceRect.height),
  sourceSummary: formatSourceSummary(session.sourceRect.width, session.sourceRect.height),
});

export const SamProcessFeedback = ({
  error,
  errorText,
  isBusy,
  statusText,
  technicalDetailsLabel,
}: {
  error: SamSessionError | null;
  errorText: string | null;
  isBusy: boolean;
  statusText: string;
  technicalDetailsLabel: string;
}) => {
  const detail = error?.detail?.trim();
  if (error && errorText) {
    return (
      <Flex align="center" gap="1" minW="0">
        <span
          aria-live="assertive"
          role="alert"
          style={{ WebkitBoxOrient: 'vertical', WebkitLineClamp: 2, display: '-webkit-box', overflow: 'hidden' }}
        >
          {errorText}
        </span>
        {detail && detail !== errorText ? (
          <Tooltip content={detail}>
            <IconButton aria-label={technicalDetailsLabel} flexShrink="0" size="xs" tabIndex={0} variant="ghost">
              <InfoIcon />
            </IconButton>
          </Tooltip>
        ) : null}
      </Flex>
    );
  }
  if (isBusy) {
    return (
      <Flex align="center" aria-live="polite" gap="2" role="status">
        <Spinner boxSize="5" flexShrink="0" />
        <span>{statusText}</span>
      </Flex>
    );
  }
  return null;
};

export const getSamActionHandlers = (engine: ToolOptionsComponentProps['engine']) => ({
  apply: () => void engine.applySelectObjectSession(),
  cancel: () => engine.cancelSelectObjectSession(),
  process: () => void engine.processSelectObjectSession(),
  reset: () => engine.resetSelectObjectSession(),
  save: (target: SelectObjectSaveTarget) => void engine.saveSelectObjectSession(target, makeImageDurable),
});

export const getSamActionEligibility = (
  session: SamSessionSnapshot,
  isExternalInteractionLocked = false
): SamActionEligibility => {
  const isProcessing =
    session.status === 'preparing-composite' ||
    session.status === 'uploading' ||
    session.status === 'processing-sam' ||
    session.status === 'rendering-preview';
  const actionsBlocked = session.status === 'committing' || isExternalInteractionLocked;
  const hasReadyPreview = session.hasPreview && !isProcessing && !actionsBlocked;
  return {
    canApply: hasReadyPreview,
    canCancel: true,
    canEditInputs: !actionsBlocked,
    canProcess: !isProcessing && !actionsBlocked && isSamDocumentInputValid(session.input),
    canReset: !actionsBlocked,
    canSave: hasReadyPreview,
  };
};

export const SamSwitch = ({
  accessibleLabel,
  checked,
  disabled,
  label,
  onChange,
}: {
  accessibleLabel: string;
  checked: boolean;
  disabled?: boolean;
  label: string;
  onChange(checked: boolean): void;
}) => (
  <Tooltip content={accessibleLabel}>
    <Switch.Root
      checked={checked}
      disabled={disabled}
      minW="0"
      {...SAM_COMPACT_CONTROL_LAYOUT}
      onCheckedChange={({ checked: next }) => onChange(next)}
    >
      <Switch.HiddenInput aria-label={accessibleLabel} />
      <Switch.Control flexShrink="0">
        <Switch.Thumb />
      </Switch.Control>
      <Switch.Label fontSize="xs" whiteSpace="nowrap">
        {label}
      </Switch.Label>
    </Switch.Root>
  </Tooltip>
);

export const SamModeToggle = ({
  disabled,
  groupLabel = 'Selection mode',
  mode,
  onPrompt,
  onVisual,
  promptLabel,
  visualLabel,
}: {
  disabled: boolean;
  groupLabel?: string;
  mode: SamSessionSnapshot['input']['type'];
  onPrompt(): void;
  onVisual(): void;
  promptLabel: string;
  visualLabel: string;
}) => (
  <Group aria-label={groupLabel} attached flex="1 1 10rem" role="group" {...SAM_COMPACT_GROUP_LAYOUT}>
    <Button
      aria-pressed={mode === 'visual'}
      disabled={disabled}
      {...SAM_SEGMENT_BUTTON_LAYOUT}
      variant={mode === 'visual' ? 'solid' : 'ghost'}
      onClick={onVisual}
    >
      {visualLabel}
    </Button>
    <Button
      aria-pressed={mode === 'prompt'}
      disabled={disabled}
      {...SAM_SEGMENT_BUTTON_LAYOUT}
      variant={mode === 'prompt' ? 'solid' : 'ghost'}
      onClick={onPrompt}
    >
      {promptLabel}
    </Button>
  </Group>
);

export const SamVisualBody = ({
  disabled,
  session,
  viewModel,
  onExclude,
  onInclude,
}: {
  disabled: boolean;
  session: SamSessionSnapshot & { input: Extract<SamSessionSnapshot['input'], { type: 'visual' }> };
  viewModel: SamPanelViewModel;
  onExclude(): void;
  onInclude(): void;
}) => {
  const { t } = useTranslation();
  return (
    <Flex
      align="center"
      aria-describedby={SAM_VISUAL_GUIDANCE_ID}
      aria-label={t('widgets.layers.selectObject.pointType')}
      flexWrap="wrap"
      gap="2"
      role="group"
    >
      <VisuallyHidden id={SAM_VISUAL_GUIDANCE_ID}>{t('widgets.layers.selectObject.visualGuidance')}</VisuallyHidden>
      <Group attached flex="1 1 10rem" {...SAM_COMPACT_GROUP_LAYOUT}>
        <Button
          aria-pressed={session.pointLabel === 'include'}
          disabled={disabled}
          {...SAM_SEGMENT_BUTTON_LAYOUT}
          variant={session.pointLabel === 'include' ? 'solid' : 'outline'}
          onClick={onInclude}
        >
          <Text as="span" fontVariantNumeric="tabular-nums" overflow="hidden" textOverflow="ellipsis">
            {t('widgets.layers.selectObject.includeCount', { count: viewModel.includeCount })}
          </Text>
        </Button>
        <Button
          aria-pressed={session.pointLabel === 'exclude'}
          disabled={disabled}
          {...SAM_SEGMENT_BUTTON_LAYOUT}
          variant={session.pointLabel === 'exclude' ? 'solid' : 'outline'}
          onClick={onExclude}
        >
          <Text as="span" fontVariantNumeric="tabular-nums" overflow="hidden" textOverflow="ellipsis">
            {t('widgets.layers.selectObject.excludeCount', { count: viewModel.excludeCount })}
          </Text>
        </Button>
      </Group>
      <Text color={viewModel.bboxActive ? 'fg' : 'fg.muted'} flex="0 1 auto" fontSize="xs" fontWeight="medium" minW="0">
        {viewModel.bboxActive
          ? t('widgets.layers.selectObject.bboxActive')
          : t('widgets.layers.selectObject.bboxInactive')}
      </Text>
    </Flex>
  );
};

export const SamPromptBody = ({
  disabled,
  prompt,
  onChange,
}: {
  disabled: boolean;
  prompt: string;
  onChange(event: ChangeEvent<HTMLInputElement>): void;
}) => {
  const { t } = useTranslation();
  return (
    <>
      <Input
        aria-describedby={SAM_PROMPT_GUIDANCE_ID}
        aria-label={t('widgets.layers.selectObject.prompt')}
        autoComplete="off"
        disabled={disabled}
        minW="0"
        placeholder={t('widgets.layers.selectObject.promptGuidance')}
        value={prompt}
        {...SAM_COMPACT_CONTROL_LAYOUT}
        onChange={onChange}
      />
      <VisuallyHidden id={SAM_PROMPT_GUIDANCE_ID}>{t('widgets.layers.selectObject.promptGuidance')}</VisuallyHidden>
    </>
  );
};

const SamSaveItem = ({
  disabled,
  onSave,
  target,
}: {
  disabled: boolean;
  onSave(target: SelectObjectSaveTarget): void;
  target: SelectObjectSaveTarget;
}) => {
  const { t } = useTranslation();
  return (
    <Menu.Item disabled={disabled} value={target} onClick={() => onSave(target)}>
      <Menu.ItemText>{t(`widgets.layers.selectObject.saveAs_${target}`)}</Menu.ItemText>
    </Menu.Item>
  );
};

export const SamOptionsPanel = ({
  engine,
  session,
  isExternalInteractionLocked = false,
}: ToolOptionsComponentProps & {
  isExternalInteractionLocked?: boolean;
  session: SamSessionSnapshot;
}) => {
  const { t } = useTranslation();
  const eligibility = getSamActionEligibility(session, isExternalInteractionLocked);
  const viewModel = getSamPanelViewModel(
    session,
    (width, height) => t('widgets.layers.selectObject.sourceDimensions', { height, width }),
    (width, height) => t('widgets.layers.selectObject.sourceDimensionsLabel', { height, width })
  );
  const actions = getSamActionHandlers(engine);
  const isProcessing =
    session.status === 'preparing-composite' ||
    session.status === 'uploading' ||
    session.status === 'processing-sam' ||
    session.status === 'rendering-preview';
  const setBoolean = (key: 'applyPolygonRefinement' | 'autoProcess' | 'invert' | 'isolatedPreview', value: boolean) =>
    engine.updateSelectObjectSession({ [key]: value });

  return (
    <CanvasOperationPanel.Root aria-labelledby="sam-operation-title" operation="select-object">
      <CanvasOperationPanel.Header>
        <Flex align="center" flexWrap="wrap" gap="2" justify="space-between">
          <Flex align="baseline" gap="1" minW="0">
            <Heading fontSize="sm" id="sam-operation-title" whiteSpace="nowrap">
              {t('widgets.layers.selectObject.title')}
            </Heading>
            <Text
              aria-label={viewModel.sourceLabel}
              color="fg.muted"
              fontSize="xs"
              fontVariantNumeric="tabular-nums"
              whiteSpace="nowrap"
            >
              {viewModel.sourceSummary}
            </Text>
          </Flex>
          <Flex align="center" flexWrap="wrap" gap="2" minW="0">
            <SamSwitch
              accessibleLabel={t('widgets.layers.selectObject.autoProcess')}
              checked={session.autoProcess}
              disabled={!eligibility.canEditInputs}
              label={t('widgets.layers.selectObject.autoProcessCompact')}
              onChange={(value) => setBoolean('autoProcess', value)}
            />
            <SamSwitch
              accessibleLabel={t('widgets.layers.selectObject.isolatedPreview')}
              checked={session.isolatedPreview}
              disabled={!eligibility.canEditInputs}
              label={t('widgets.layers.selectObject.isolatedPreviewCompact')}
              onChange={(value) => setBoolean('isolatedPreview', value)}
            />
          </Flex>
        </Flex>
      </CanvasOperationPanel.Header>
      <CanvasOperationPanel.Body>
        <Stack gap="2">
          <Flex align="center" flexWrap="wrap" gap="1" justify="space-between" minW="0">
            <SamModeToggle
              disabled={!eligibility.canEditInputs}
              groupLabel={t('widgets.layers.selectObject.mode')}
              mode={session.input.type}
              promptLabel={t('widgets.layers.selectObject.promptMode')}
              visualLabel={t('widgets.layers.selectObject.visual')}
              onPrompt={() => engine.updateSelectObjectSession({ input: { prompt: '', type: 'prompt' } })}
              onVisual={() =>
                engine.updateSelectObjectSession({
                  input: { bbox: null, excludePoints: [], includePoints: [], type: 'visual' },
                })
              }
            />
            <SamSwitch
              accessibleLabel={t('widgets.layers.selectObject.invert')}
              checked={session.invert}
              disabled={!eligibility.canEditInputs}
              label={t('widgets.layers.selectObject.invert')}
              onChange={(value) => setBoolean('invert', value)}
            />
          </Flex>
          {session.input.type === 'visual' ? (
            <SamVisualBody
              disabled={!eligibility.canEditInputs}
              session={{ ...session, input: session.input }}
              viewModel={viewModel}
              onExclude={() => engine.updateSelectObjectSession({ pointLabel: 'exclude' })}
              onInclude={() => engine.updateSelectObjectSession({ pointLabel: 'include' })}
            />
          ) : (
            <SamPromptBody
              disabled={!eligibility.canEditInputs}
              prompt={session.input.prompt}
              onChange={(event) =>
                engine.updateSelectObjectSession({ input: { prompt: event.currentTarget.value, type: 'prompt' } })
              }
            />
          )}
          <Flex align="center" flexWrap="wrap" gap="1" minW="0">
            <Text asChild fontSize="xs" fontWeight="semibold">
              <label htmlFor="sam-model">{t('widgets.layers.selectObject.model')}</label>
            </Text>
            <NativeSelect.Root
              disabled={isProcessing || !eligibility.canEditInputs}
              {...SAM_COMPACT_CONTROL_LAYOUT}
              {...SAM_MODEL_SELECT_LAYOUT}
            >
              <NativeSelect.Field
                aria-label={t('widgets.layers.selectObject.model')}
                id="sam-model"
                value={session.model}
                onChange={(event) => engine.updateSelectObjectSession({ model: event.currentTarget.value as SamModel })}
              >
                <option value="segment-anything-2-large">{t('widgets.layers.selectObject.modelSam2Large')}</option>
                <option value="segment-anything-huge">{t('widgets.layers.selectObject.modelHuge')}</option>
              </NativeSelect.Field>
              <NativeSelect.Indicator />
            </NativeSelect.Root>
            <SamSwitch
              accessibleLabel={t('widgets.layers.selectObject.refine')}
              checked={session.applyPolygonRefinement}
              disabled={isProcessing || !eligibility.canEditInputs}
              label={t('widgets.layers.selectObject.refine')}
              onChange={(value) => setBoolean('applyPolygonRefinement', value)}
            />
          </Flex>
        </Stack>
      </CanvasOperationPanel.Body>
      {session.error || isProcessing || session.status === 'scheduled' || session.status === 'committing' ? (
        <CanvasOperationPanel.Feedback color={session.error ? 'fg.error' : 'fg.muted'} fontSize="xs">
          <SamProcessFeedback
            error={session.error}
            errorText={session.error ? t(getSamErrorTranslationKey(session.error.code)) : null}
            isBusy={!session.error}
            statusText={t(getSamStatusTranslationKey(session.status))}
            technicalDetailsLabel={t('widgets.layers.selectObject.technicalDetails')}
          />
        </CanvasOperationPanel.Feedback>
      ) : null}
      <CanvasOperationPanel.Footer {...SAM_COMPACT_FOOTER_LAYOUT}>
        <Button
          disabled={!eligibility.canProcess}
          loading={isProcessing}
          {...SAM_COMPACT_BUTTON_LAYOUT}
          onClick={actions.process}
        >
          {t('widgets.layers.selectObject.process')}
        </Button>
        <Button
          colorPalette="accent"
          disabled={!eligibility.canApply}
          {...SAM_COMPACT_BUTTON_LAYOUT}
          onClick={actions.apply}
        >
          {t('common.apply')}
        </Button>
        <Button disabled={!eligibility.canReset} variant="ghost" {...SAM_COMPACT_BUTTON_LAYOUT} onClick={actions.reset}>
          {t('widgets.layers.selectObject.reset')}
        </Button>
        <Menu.Root>
          <Menu.Trigger asChild>
            <Button disabled={!eligibility.canSave} variant="outline" {...SAM_COMPACT_BUTTON_LAYOUT}>
              {t('widgets.layers.selectObject.saveAs')} <ChevronDownIcon size={14} />
            </Button>
          </Menu.Trigger>
          <Portal>
            <Menu.Positioner>
              <MenuContent minW="11rem" py="1">
                {SAVE_TARGETS.map((target) => (
                  <SamSaveItem key={target} disabled={!eligibility.canSave} target={target} onSave={actions.save} />
                ))}
              </MenuContent>
            </Menu.Positioner>
          </Portal>
        </Menu.Root>
        <Button
          disabled={!eligibility.canCancel}
          variant="ghost"
          {...SAM_COMPACT_BUTTON_LAYOUT}
          onClick={actions.cancel}
        >
          {t('common.cancel')}
        </Button>
      </CanvasOperationPanel.Footer>
    </CanvasOperationPanel.Root>
  );
};

export const SamOptions = ({
  engine,
  isExternalInteractionLocked = false,
}: ToolOptionsComponentProps & { isExternalInteractionLocked?: boolean }) => {
  const session = useSamSession(engine);
  if (!session) {
    return null;
  }
  return (
    <SamOptionsPanel engine={engine} isExternalInteractionLocked={isExternalInteractionLocked} session={session} />
  );
};
