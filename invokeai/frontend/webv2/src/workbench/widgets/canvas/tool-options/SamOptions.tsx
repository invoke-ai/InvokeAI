import type {
  OperationPropertyForm,
  ToolFormProps,
  ToolFooterProps,
} from '@workbench/widgets/canvas/tool-presentation/toolFormContracts';
/* oxlint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ChangeEvent } from 'react';

import {
  Box,
  createListCollection,
  Flex,
  Icon,
  Input,
  Menu,
  Portal,
  Stack,
  Switch,
  Text,
  VisuallyHidden,
} from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { Group } from '@platform/ui/Group';
import { MenuActionItem, MenuContent } from '@platform/ui/Menu';
import { Select } from '@platform/ui/Select';
import { Tooltip } from '@platform/ui/Tooltip';
import {
  getCanvasOperations,
  isSamDocumentInputValid,
  type CanvasOperationCapability,
  type SamSessionError,
  type SamSessionErrorCode,
  type SamSessionSnapshot,
  type SamModel,
  type SelectObjectSaveTarget,
} from '@workbench/canvas-operations/api';
import { useSamSession } from '@workbench/widgets/canvas/engineStoreHooks';
import { PropertySwitchRow } from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import { ChevronDownIcon, SquareMinusIcon, SquarePlusIcon } from 'lucide-react';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { OperationStatusChip, OperationStatusSlot } from './OperationStatusSlot';

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

/** Reserve OperationStatusSlot width so SAM status/error text cannot shift controls. */
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
}) => (
  <OperationStatusSlot
    errorDetail={error?.detail ?? null}
    errorText={error && errorText ? errorText : null}
    isBusy={isBusy}
    minW="0"
    statusText={statusText}
    technicalDetailsLabel={technicalDetailsLabel}
  />
);

/** Legacy parity: canvas adoption keeps the SAM result intermediate and out of the gallery. */
export const keepSamImageIntermediate = (_imageName: string): Promise<void> => Promise.resolve();

export const getSamActionHandlers = (operations: CanvasOperationCapability) => ({
  apply: () => void operations.applySelectObjectSession(keepSamImageIntermediate),
  cancel: () => operations.cancelSelectObjectSession(),
  process: () => void operations.processSelectObjectSession(),
  reset: () => operations.resetSelectObjectSession(),
  save: (target: SelectObjectSaveTarget) => void operations.saveSelectObjectSession(target, keepSamImageIntermediate),
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
  const includeLabel = t('widgets.layers.selectObject.includeCount', { count: viewModel.includeCount });
  const excludeLabel = t('widgets.layers.selectObject.excludeCount', { count: viewModel.excludeCount });
  return (
    <Group
      aria-describedby={SAM_VISUAL_GUIDANCE_ID}
      aria-label={t('widgets.layers.selectObject.pointType')}
      attached
      flexShrink="0"
      role="group"
    >
      <VisuallyHidden id={SAM_VISUAL_GUIDANCE_ID}>{t('widgets.layers.selectObject.visualGuidance')}</VisuallyHidden>
      <Tooltip content={includeLabel}>
        <Button
          aria-label={includeLabel}
          aria-pressed={pointLabel === 'include'}
          disabled={disabled}
          px="1.5"
          size="xs"
          variant={pointLabel === 'include' ? 'solid' : 'outline'}
          onClick={onInclude}
        >
          <SquarePlusIcon />
          <Text as="span" fontVariantNumeric="tabular-nums">
            {viewModel.includeCount}
          </Text>
        </Button>
      </Tooltip>
      <Tooltip content={excludeLabel}>
        <Button
          aria-label={excludeLabel}
          aria-pressed={pointLabel === 'exclude'}
          disabled={disabled}
          px="1.5"
          size="xs"
          variant={pointLabel === 'exclude' ? 'solid' : 'outline'}
          onClick={onExclude}
        >
          <SquareMinusIcon />
          <Text as="span" fontVariantNumeric="tabular-nums">
            {viewModel.excludeCount}
          </Text>
        </Button>
      </Tooltip>
    </Group>
  );
};

/** Whether a visual box is placed; lives in the More menu beside the settings. */
export const SamBboxIndicator = ({ viewModel }: { viewModel: SamPanelViewModel }) => {
  const { t } = useTranslation();
  const bboxText = viewModel.bboxActive
    ? t('widgets.layers.selectObject.bboxActive')
    : t('widgets.layers.selectObject.bboxInactive');
  return (
    <Text color={viewModel.bboxActive ? 'fg' : 'fg.subtle'} fontSize="xs" fontWeight="medium">
      {bboxText}
    </Text>
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
        flexShrink={0}
        placeholder={t('widgets.layers.selectObject.promptGuidance')}
        size="xs"
        value={prompt}
        w="7.5rem"
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

/** Set-once session settings (model, refinement, preview behavior), shown in the More menu. */
export const SamSettings = ({
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
  const modelValue = useMemo(() => [session.model], [session.model]);
  const modelCollection = useMemo(
    () =>
      createListCollection({
        items: [
          { label: t('widgets.layers.selectObject.modelSam2Large'), value: 'segment-anything-2-large' },
          { label: t('widgets.layers.selectObject.modelHuge'), value: 'segment-anything-huge' },
        ] as const,
      }),
    [t]
  );
  return (
    <Stack aria-label={t('widgets.layers.selectObject.settings')} gap="2" role="group" w="full">
      <Stack gap="1">
        <Text asChild fontSize="xs" fontWeight="semibold">
          <label htmlFor="sam-model">{t('widgets.layers.selectObject.model')}</label>
        </Text>
        <Select
          collection={modelCollection}
          disabled={isProcessing || !eligibility.canEditInputs}
          ids={{ trigger: 'sam-model' }}
          size="xs"
          value={modelValue}
          onValueChange={({ value }) => {
            const model = value[0];
            if (model === 'segment-anything-2-large' || model === 'segment-anything-huge') {
              onModelChange(model);
            }
          }}
        />
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
  );
};

/** Input mode, the visual point labels or the prompt, and invert: what changes between previews. */

/** The session's inputs: visual points or a prompt, plus the invert switch. */
const SamInputSettings = ({ engine, isSurfaceInteractionLocked }: ToolFormProps) => {
  const { t } = useTranslation();
  const session = useSamSession(engine);
  const operations = getCanvasOperations(engine);
  if (!session) {
    return null;
  }
  const eligibility = getSamActionEligibility(session, isSurfaceInteractionLocked);
  const viewModel = getSamPanelViewModel(session, (layerName, width, height) =>
    t('widgets.layers.selectObject.sourceLayerLabel', {
      height,
      name: layerName,
      type: t(`widgets.layers.selectObject.saveAs_${session.layerType}`),
      width,
    })
  );
  return (
    <>
      <SamModeToggle
        disabled={!eligibility.canEditInputs}
        groupLabel={t('widgets.layers.selectObject.mode')}
        mode={session.input.type}
        promptLabel={t('widgets.layers.selectObject.promptMode')}
        visualLabel={t('widgets.layers.selectObject.visual')}
        onPrompt={() => operations.updateSelectObjectSession({ input: { prompt: '', type: 'prompt' } })}
        onVisual={() =>
          operations.updateSelectObjectSession({
            input: { bbox: null, excludePoints: [], includePoints: [], type: 'visual' },
          })
        }
      />
      {session.input.type === 'visual' ? (
        <>
          <SamVisualInput
            disabled={!eligibility.canEditInputs}
            pointLabel={session.pointLabel}
            viewModel={viewModel}
            onExclude={() => operations.updateSelectObjectSession({ pointLabel: 'exclude' })}
            onInclude={() => operations.updateSelectObjectSession({ pointLabel: 'include' })}
          />
          <SamBboxIndicator viewModel={viewModel} />
        </>
      ) : (
        <SamPromptBody
          disabled={!eligibility.canEditInputs}
          prompt={session.input.prompt}
          onChange={(event) =>
            operations.updateSelectObjectSession({ input: { prompt: event.currentTarget.value, type: 'prompt' } })
          }
        />
      )}
      <PropertySwitchRow
        checked={session.invert}
        disabled={!eligibility.canEditInputs}
        label={t('widgets.layers.selectObject.invert')}
        onCheckedChange={(invert) => operations.updateSelectObjectSession({ invert })}
      />
    </>
  );
};

/** Set-once session settings: model, refinement, preview behavior. */
const SamSettingsGroup = ({ engine, isSurfaceInteractionLocked }: ToolFormProps) => {
  const session = useSamSession(engine);
  const operations = getCanvasOperations(engine);
  if (!session) {
    return null;
  }
  const eligibility = getSamActionEligibility(session, isSurfaceInteractionLocked);
  return (
    <SamSettings
      eligibility={eligibility}
      isProcessing={isSamProcessingStatus(session.status)}
      session={session}
      onModelChange={(model) => operations.updateSelectObjectSession({ model })}
      onToggle={(key, value) => operations.updateSelectObjectSession({ [key]: value })}
    />
  );
};

const APPLY_MENU_POSITIONING = { placement: 'top-end' } as const;

/** The sticky footer: status chip, Reset, Process, Apply with its save-as menu, Cancel. */
const SamFooter = ({ engine, isExternalInteractionLocked }: ToolFooterProps) => {
  const { t } = useTranslation();
  const session = useSamSession(engine);
  const actions = useMemo(() => getSamActionHandlers(getCanvasOperations(engine)), [engine]);
  if (!session) {
    return null;
  }
  const eligibility = getSamActionEligibility(session, isExternalInteractionLocked);
  const isProcessing = isSamProcessingStatus(session.status);
  const isBusy = !session.error && (isProcessing || session.status === 'scheduled' || session.status === 'committing');
  const sourceLabel = t('widgets.layers.selectObject.sourceLayerLabel', {
    height: session.sourceRect.height,
    name: session.layerName,
    type: t(`widgets.layers.selectObject.saveAs_${session.layerType}`),
    width: session.sourceRect.width,
  });
  return (
    <Flex align="center" gap="1" minW="0" w="full">
      {/* No nowrap: the chip's own two-line clamp must keep the details button visible. */}
      <Box flex="1" minW="0" overflow="hidden">
        <OperationStatusChip
          errorDetail={session.error?.detail ?? null}
          errorText={session.error ? t(getSamErrorTranslationKey(session.error.code)) : null}
          isBusy={isBusy}
          sourceLabel={sourceLabel}
          statusText={t(getSamStatusTranslationKey(session.status))}
          technicalDetailsLabel={t('widgets.layers.selectObject.technicalDetails')}
          title={t('widgets.layers.selectObject.title')}
        />
      </Box>
      <Button disabled={!eligibility.canReset} flexShrink={0} size="xs" variant="ghost" onClick={actions.reset}>
        {t('widgets.layers.selectObject.reset')}
      </Button>
      <Button
        disabled={!eligibility.canProcess}
        flexShrink={0}
        loading={isProcessing}
        size="xs"
        onClick={actions.process}
      >
        {t('widgets.layers.selectObject.process')}
      </Button>
      <Flex flexShrink={0}>
        <Button
          data-pane-action="apply"
          disabled={!eligibility.canApply}
          loading={session.status === 'committing'}
          roundedRight="none"
          size="xs"
          variant="solid"
          onClick={actions.apply}
        >
          {t('common.apply')}
        </Button>
        <Menu.Root positioning={APPLY_MENU_POSITIONING}>
          <Menu.Trigger asChild>
            <Button
              aria-label={t('widgets.layers.selectObject.saveAs')}
              disabled={!eligibility.canSave}
              px="1"
              roundedLeft="none"
              size="xs"
              variant="solid"
            >
              <Icon as={ChevronDownIcon} boxSize="3.5" />
            </Button>
          </Menu.Trigger>
          <Portal>
            <Menu.Positioner>
              <MenuContent minW="12rem" py="1">
                {SAVE_TARGETS.map((target) => (
                  <MenuActionItem
                    key={target}
                    disabled={!eligibility.canSave}
                    label={t(`widgets.layers.selectObject.saveAs_${target}`)}
                    value={`save-${target}`}
                    onSelect={() => actions.save(target)}
                  />
                ))}
              </MenuContent>
            </Menu.Positioner>
          </Portal>
        </Menu.Root>
      </Flex>
      <Button
        data-pane-action="cancel"
        disabled={!eligibility.canCancel}
        flexShrink={0}
        size="xs"
        variant="ghost"
        onClick={actions.cancel}
      >
        {t('common.cancel')}
      </Button>
    </Flex>
  );
};

export const selectObjectOperationForm: OperationPropertyForm = {
  footer: SamFooter,
  groups: [
    { body: SamInputSettings, id: 'sam-input', labelKey: 'widgets.properties.groups.input' },
    {
      body: SamSettingsGroup,
      collapsible: 'collapsed',
      id: 'sam-settings',
      labelKey: 'widgets.properties.groups.settings',
    },
  ],
  kind: 'select-object',
};
