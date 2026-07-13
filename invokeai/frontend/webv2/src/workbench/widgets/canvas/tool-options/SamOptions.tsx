/* oxlint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { SelectObjectSaveTarget } from '@workbench/canvas-engine/engine';
import type { SamSessionError, SamSessionErrorCode, SamSessionSnapshot } from '@workbench/canvas-engine/engineStores';
import type { SamModel } from '@workbench/generation/canvas/samGraph';
import type { ChangeEvent, CSSProperties } from 'react';

import {
  Flex,
  Group,
  HStack,
  IconButton,
  Input,
  Menu,
  NativeSelect,
  Popover,
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
import { CanvasFloatingBar, CanvasFloatingBarDivider } from '@workbench/widgets/canvas/CanvasFloatingBar';
import { useSamSession } from '@workbench/widgets/canvas/engineStoreHooks';
import { ChevronDownIcon, InfoIcon, SettingsIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import type { ToolOptionsComponentProps } from './ToolOptionsBar';

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
}

const SAM_PROMPT_GUIDANCE_ID = 'sam-prompt-guidance';
const SAM_VISUAL_GUIDANCE_ID = 'sam-visual-guidance';

const SAM_UPWARD_POSITIONING = { placement: 'top-end' } as const;

const SAM_ERROR_CLAMP_STYLE: CSSProperties = {
  display: '-webkit-box',
  overflow: 'hidden',
  WebkitBoxOrient: 'vertical',
  WebkitLineClamp: 2,
};

const SAM_STATUS_TRANSLATION_KEYS: Record<SamSessionSnapshot['status'], string> = {
  committing: 'widgets.layers.selectObject.statusCommitting',
  error: 'widgets.layers.selectObject.statusError',
  'preparing-source': 'widgets.layers.selectObject.statusPreparingSource',
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

const SAVE_TARGETS: readonly SelectObjectSaveTarget[] = [
  'selection',
  'raster',
  'control',
  'inpaint_mask',
  'regional_guidance',
];

const isSamProcessingStatus = (status: SamSessionSnapshot['status']): boolean =>
  status === 'preparing-source' ||
  status === 'uploading' ||
  status === 'processing-sam' ||
  status === 'rendering-preview';

export const getSamStatusTranslationKey = (status: SamSessionSnapshot['status']): string =>
  SAM_STATUS_TRANSLATION_KEYS[status];

export const getSamErrorTranslationKey = (code: SamSessionErrorCode): string => SAM_ERROR_TRANSLATION_KEYS[code];

export const getSamPanelViewModel = (
  session: SamSessionSnapshot,
  formatSourceLabel: (layerName: string, width: number, height: number) => string
): SamPanelViewModel => ({
  bboxActive: session.input.type === 'visual' && session.input.bbox !== null,
  excludeCount: session.input.type === 'visual' ? session.input.excludePoints.length : 0,
  includeCount: session.input.type === 'visual' ? session.input.includePoints.length : 0,
  sourceLabel: formatSourceLabel(session.layerName, session.sourceRect.width, session.sourceRect.height),
});

/**
 * The always-mounted status slot: reserves its width so status/error text
 * appearing never shifts the surrounding controls, and keeps the polite live
 * region in the tree before content arrives so announcements are reliable.
 */
export const SamStatusSlot = ({
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
  return (
    <Flex
      align="center"
      color={error ? 'fg.error' : 'fg.muted'}
      flex="0 1 auto"
      fontSize="xs"
      gap="1"
      maxW="16rem"
      minW="8rem"
    >
      {error && errorText ? (
        <>
          <span aria-live="assertive" role="alert" style={SAM_ERROR_CLAMP_STYLE}>
            {errorText}
          </span>
          {detail && detail !== errorText ? (
            <Tooltip content={detail}>
              <IconButton aria-label={technicalDetailsLabel} flexShrink="0" size="xs" tabIndex={0} variant="ghost">
                <InfoIcon />
              </IconButton>
            </Tooltip>
          ) : null}
        </>
      ) : (
        <Flex align="center" aria-live="polite" gap="2" minW="0" role="status">
          {isBusy ? (
            <>
              <Spinner flexShrink="0" size="xs" />
              <span>{statusText}</span>
            </>
          ) : null}
        </Flex>
      )}
    </Flex>
  );
};

export const getSamActionHandlers = (engine: ToolOptionsComponentProps['engine']) => ({
  apply: () => void engine.applySelectObjectSession(makeImageDurable),
  cancel: () => engine.cancelSelectObjectSession(),
  process: () => void engine.processSelectObjectSession(),
  reset: () => engine.resetSelectObjectSession(),
  save: (target: SelectObjectSaveTarget) => void engine.saveSelectObjectSession(target, makeImageDurable),
});

export const getSamActionEligibility = (
  session: SamSessionSnapshot,
  isExternalInteractionLocked = false
): SamActionEligibility => {
  const isProcessing = isSamProcessingStatus(session.status);
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
  <Group aria-label={groupLabel} attached flexShrink="0" role="group">
    <Button
      aria-pressed={mode === 'visual'}
      disabled={disabled}
      size="xs"
      variant={mode === 'visual' ? 'solid' : 'ghost'}
      onClick={onVisual}
    >
      {visualLabel}
    </Button>
    <Button
      aria-pressed={mode === 'prompt'}
      disabled={disabled}
      size="xs"
      variant={mode === 'prompt' ? 'solid' : 'ghost'}
      onClick={onPrompt}
    >
      {promptLabel}
    </Button>
  </Group>
);

export const SamVisualInput = ({
  disabled,
  pointLabel,
  viewModel,
  onExclude,
  onInclude,
}: {
  disabled: boolean;
  pointLabel: SamSessionSnapshot['pointLabel'];
  viewModel: SamPanelViewModel;
  onExclude(): void;
  onInclude(): void;
}) => {
  const { t } = useTranslation();
  const bboxText = viewModel.bboxActive
    ? t('widgets.layers.selectObject.bboxActive')
    : t('widgets.layers.selectObject.bboxInactive');
  return (
    <Flex
      align="center"
      aria-describedby={SAM_VISUAL_GUIDANCE_ID}
      aria-label={t('widgets.layers.selectObject.pointType')}
      gap="1"
      minW="0"
      role="group"
    >
      <VisuallyHidden id={SAM_VISUAL_GUIDANCE_ID}>{t('widgets.layers.selectObject.visualGuidance')}</VisuallyHidden>
      <Group attached flexShrink="0">
        <Button
          aria-pressed={pointLabel === 'include'}
          disabled={disabled}
          size="xs"
          variant={pointLabel === 'include' ? 'solid' : 'outline'}
          onClick={onInclude}
        >
          <Text as="span" fontVariantNumeric="tabular-nums">
            {t('widgets.layers.selectObject.includeCount', { count: viewModel.includeCount })}
          </Text>
        </Button>
        <Button
          aria-pressed={pointLabel === 'exclude'}
          disabled={disabled}
          size="xs"
          variant={pointLabel === 'exclude' ? 'solid' : 'outline'}
          onClick={onExclude}
        >
          <Text as="span" fontVariantNumeric="tabular-nums">
            {t('widgets.layers.selectObject.excludeCount', { count: viewModel.excludeCount })}
          </Text>
        </Button>
      </Group>
      <Tooltip content={bboxText}>
        <Text
          color={viewModel.bboxActive ? 'fg' : 'fg.subtle'}
          fontSize="2xs"
          fontWeight="medium"
          px="1"
          whiteSpace="nowrap"
        >
          <span aria-hidden="true">{t('widgets.layers.selectObject.bbox')}</span>
          <VisuallyHidden>{bboxText}</VisuallyHidden>
        </Text>
      </Tooltip>
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
        flex="0 1 13rem"
        h="8"
        minW="6rem"
        placeholder={t('widgets.layers.selectObject.promptGuidance')}
        size="xs"
        value={prompt}
        w="13rem"
        onChange={onChange}
      />
      <VisuallyHidden id={SAM_PROMPT_GUIDANCE_ID}>{t('widgets.layers.selectObject.promptGuidance')}</VisuallyHidden>
    </>
  );
};

const SamSettingsSwitch = ({
  checked,
  disabled,
  label,
  onChange,
}: {
  checked: boolean;
  disabled?: boolean;
  label: string;
  onChange(checked: boolean): void;
}) => (
  <Switch.Root
    checked={checked}
    disabled={disabled}
    justifyContent="space-between"
    size="sm"
    w="full"
    onCheckedChange={({ checked: next }) => onChange(next)}
  >
    <Switch.Label fontSize="xs">{label}</Switch.Label>
    <Switch.HiddenInput />
    <Switch.Control>
      <Switch.Thumb />
    </Switch.Control>
  </Switch.Root>
);

/** Set-once session settings (model, refinement, preview behavior) demoted out of the bar. */
export const SamSettingsPopover = ({
  eligibility,
  isProcessing,
  session,
  onModelChange,
  onToggle,
}: {
  eligibility: SamActionEligibility;
  isProcessing: boolean;
  session: SamSessionSnapshot;
  onModelChange(model: SamModel): void;
  onToggle(key: 'applyPolygonRefinement' | 'autoProcess' | 'isolatedPreview', value: boolean): void;
}) => {
  const { t } = useTranslation();
  return (
    <Popover.Root lazyMount positioning={SAM_UPWARD_POSITIONING} unmountOnExit>
      <Popover.Trigger asChild>
        <Tooltip content={t('widgets.layers.selectObject.settings')}>
          <IconButton aria-label={t('widgets.layers.selectObject.settings')} size="xs" variant="ghost">
            <SettingsIcon />
          </IconButton>
        </Tooltip>
      </Popover.Trigger>
      <Portal>
        <Popover.Positioner>
          <Popover.Content bg="bg.muted" borderColor="border.emphasized" borderWidth="1px" w="16rem">
            <Popover.Body p="2.5">
              <Stack gap="2.5">
                <Stack gap="1">
                  <Text asChild fontSize="xs" fontWeight="semibold">
                    <label htmlFor="sam-model">{t('widgets.layers.selectObject.model')}</label>
                  </Text>
                  <NativeSelect.Root disabled={isProcessing || !eligibility.canEditInputs} size="xs">
                    <NativeSelect.Field
                      id="sam-model"
                      value={session.model}
                      onChange={(event) => onModelChange(event.currentTarget.value as SamModel)}
                    >
                      <option value="segment-anything-2-large">
                        {t('widgets.layers.selectObject.modelSam2Large')}
                      </option>
                      <option value="segment-anything-huge">{t('widgets.layers.selectObject.modelHuge')}</option>
                    </NativeSelect.Field>
                    <NativeSelect.Indicator />
                  </NativeSelect.Root>
                </Stack>
                <SamSettingsSwitch
                  checked={session.applyPolygonRefinement}
                  disabled={isProcessing || !eligibility.canEditInputs}
                  label={t('widgets.layers.selectObject.refine')}
                  onChange={(value) => onToggle('applyPolygonRefinement', value)}
                />
                <SamSettingsSwitch
                  checked={session.autoProcess}
                  disabled={!eligibility.canEditInputs}
                  label={t('widgets.layers.selectObject.autoProcess')}
                  onChange={(value) => onToggle('autoProcess', value)}
                />
                <SamSettingsSwitch
                  checked={session.isolatedPreview}
                  disabled={!eligibility.canEditInputs}
                  label={t('widgets.layers.selectObject.isolatedPreview')}
                  onChange={(value) => onToggle('isolatedPreview', value)}
                />
              </Stack>
            </Popover.Body>
          </Popover.Content>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
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

export const SamOptionsBar = ({
  engine,
  session,
  isExternalInteractionLocked = false,
}: ToolOptionsComponentProps & {
  isExternalInteractionLocked?: boolean;
  session: SamSessionSnapshot;
}) => {
  const { t } = useTranslation();
  const eligibility = getSamActionEligibility(session, isExternalInteractionLocked);
  const viewModel = getSamPanelViewModel(session, (layerName, width, height) =>
    t('widgets.layers.selectObject.sourceLayerLabel', {
      height,
      name: layerName,
      type: t(`widgets.layers.selectObject.saveAs_${session.layerType}`),
      width,
    })
  );
  const actions = getSamActionHandlers(engine);
  const isProcessing = isSamProcessingStatus(session.status);
  const isBusy = !session.error && (isProcessing || session.status === 'scheduled' || session.status === 'committing');
  const setBoolean = (key: 'applyPolygonRefinement' | 'autoProcess' | 'invert' | 'isolatedPreview', value: boolean) =>
    engine.updateSelectObjectSession({ [key]: value });

  return (
    <CanvasFloatingBar maxW="full">
      <Flex
        align="center"
        aria-label={t('widgets.layers.selectObject.title')}
        flexWrap="wrap"
        gap="1"
        minW="0"
        role="group"
      >
        <Tooltip content={viewModel.sourceLabel}>
          <Text flexShrink="0" fontSize="xs" fontWeight="semibold" px="1" whiteSpace="nowrap">
            {t('widgets.layers.selectObject.title')}
            <VisuallyHidden>{viewModel.sourceLabel}</VisuallyHidden>
          </Text>
        </Tooltip>
        <CanvasFloatingBarDivider />
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
        {session.input.type === 'visual' ? (
          <SamVisualInput
            disabled={!eligibility.canEditInputs}
            pointLabel={session.pointLabel}
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
        <CanvasFloatingBarDivider />
        <Button
          aria-pressed={session.invert}
          disabled={!eligibility.canEditInputs}
          size="xs"
          variant={session.invert ? 'solid' : 'ghost'}
          onClick={() => setBoolean('invert', !session.invert)}
        >
          {t('widgets.layers.selectObject.invert')}
        </Button>
        <CanvasFloatingBarDivider />
        <SamStatusSlot
          error={session.error}
          errorText={session.error ? t(getSamErrorTranslationKey(session.error.code)) : null}
          isBusy={isBusy}
          statusText={t(getSamStatusTranslationKey(session.status))}
          technicalDetailsLabel={t('widgets.layers.selectObject.technicalDetails')}
        />
        <SamSettingsPopover
          eligibility={eligibility}
          isProcessing={isProcessing}
          session={session}
          onModelChange={(model) => engine.updateSelectObjectSession({ model })}
          onToggle={setBoolean}
        />
        <CanvasFloatingBarDivider />
        <HStack flexShrink="0" gap="1">
          <Button disabled={!eligibility.canProcess} loading={isProcessing} size="xs" onClick={actions.process}>
            {t('widgets.layers.selectObject.process')}
          </Button>
          <Button disabled={!eligibility.canReset} size="xs" variant="ghost" onClick={actions.reset}>
            {t('widgets.layers.selectObject.reset')}
          </Button>
          <Menu.Root positioning={SAM_UPWARD_POSITIONING}>
            <Group attached>
              <Button
                colorPalette="accent"
                disabled={!eligibility.canApply}
                roundedEnd="none"
                size="xs"
                onClick={actions.apply}
              >
                {t('common.apply')}
              </Button>
              <Menu.Trigger asChild>
                <IconButton
                  aria-label={t('widgets.layers.selectObject.saveAs')}
                  colorPalette="accent"
                  disabled={!eligibility.canSave}
                  minW="0"
                  roundedStart="none"
                  size="xs"
                  w="6"
                >
                  <ChevronDownIcon />
                </IconButton>
              </Menu.Trigger>
            </Group>
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
          <Button disabled={!eligibility.canCancel} size="xs" variant="ghost" onClick={actions.cancel}>
            {t('common.cancel')}
          </Button>
        </HStack>
      </Flex>
    </CanvasFloatingBar>
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
  return <SamOptionsBar engine={engine} isExternalInteractionLocked={isExternalInteractionLocked} session={session} />;
};
