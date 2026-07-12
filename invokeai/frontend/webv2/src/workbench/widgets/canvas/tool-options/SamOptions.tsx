/* oxlint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { SelectObjectSaveTarget } from '@workbench/canvas-engine/engine';
import type { SamSessionError, SamSessionErrorCode, SamSessionSnapshot } from '@workbench/canvas-engine/engineStores';
import type { SamModel } from '@workbench/generation/canvas/samGraph';
import type { ChangeEvent } from 'react';

import {
  Box,
  Grid,
  Heading,
  HStack,
  Menu,
  NativeSelect,
  Portal,
  Stack,
  Switch,
  Text,
  Textarea,
} from '@chakra-ui/react';
import { Button, Field, MenuContent } from '@workbench/components/ui';
import { makeImageDurable } from '@workbench/gallery/api';
import { isSamDocumentInputValid } from '@workbench/generation/canvas/samGraph';
import { useSamSession } from '@workbench/widgets/canvas/engineStoreHooks';
import { ChevronDownIcon } from 'lucide-react';
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
  sourceSummary: string;
}

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

export const getSamPanelViewModel = (session: SamSessionSnapshot): SamPanelViewModel => ({
  bboxActive: session.input.type === 'visual' && session.input.bbox !== null,
  excludeCount: session.input.type === 'visual' ? session.input.excludePoints.length : 0,
  includeCount: session.input.type === 'visual' ? session.input.includePoints.length : 0,
  sourceSummary: `${session.sourceRect.width} × ${session.sourceRect.height} generation area`,
});

export const SamProcessFeedback = ({
  error,
  errorText,
  statusText,
}: {
  error: SamSessionError | null;
  errorText: string | null;
  statusText: string;
}) => {
  const detail = error?.detail?.trim();
  const showDetail = Boolean(detail && detail !== errorText);
  return (
    <span>
      <span aria-live="polite" role="status">
        {statusText}
      </span>
      {error && errorText ? (
        <span aria-live="assertive" role="alert">
          <span>{errorText}</span>
          {showDetail ? <span>{detail}</span> : null}
        </span>
      ) : null}
    </span>
  );
};

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

const SamSwitch = ({
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
    minH="10"
    size="sm"
    onCheckedChange={({ checked: next }) => onChange(next)}
  >
    <Switch.HiddenInput />
    <Switch.Control>
      <Switch.Thumb />
    </Switch.Control>
    <Switch.Label fontSize="xs">{label}</Switch.Label>
  </Switch.Root>
);

const SamVisualBody = ({
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
    <Stack aria-labelledby="sam-visual-tab" gap="3" id="sam-visual-panel" role="tabpanel">
      <Box>
        <Text fontSize="xs" fontWeight="semibold" mb="2">
          {t('widgets.layers.selectObject.pointType')}
        </Text>
        <Grid gap="2" templateColumns={{ base: '1fr', sm: 'repeat(2, minmax(0, 1fr))' }}>
          <Button
            aria-pressed={session.pointLabel === 'include'}
            disabled={disabled}
            minH="10"
            variant={session.pointLabel === 'include' ? 'solid' : 'outline'}
            onClick={onInclude}
          >
            {t('widgets.layers.selectObject.include')}{' '}
            <Text as="span" fontVariantNumeric="tabular-nums">
              {viewModel.includeCount}
            </Text>
          </Button>
          <Button
            aria-pressed={session.pointLabel === 'exclude'}
            disabled={disabled}
            minH="10"
            variant={session.pointLabel === 'exclude' ? 'solid' : 'outline'}
            onClick={onExclude}
          >
            {t('widgets.layers.selectObject.exclude')}{' '}
            <Text as="span" fontVariantNumeric="tabular-nums">
              {viewModel.excludeCount}
            </Text>
          </Button>
        </Grid>
      </Box>
      <Text color={viewModel.bboxActive ? 'fg' : 'fg.muted'} fontSize="xs" fontWeight="medium">
        {viewModel.bboxActive
          ? t('widgets.layers.selectObject.bboxActive')
          : t('widgets.layers.selectObject.bboxInactive')}
      </Text>
      <Text color="fg.muted" fontSize="xs">
        {t('widgets.layers.selectObject.visualGuidance')}
      </Text>
    </Stack>
  );
};

const SamPromptBody = ({
  disabled,
  prompt,
  onChange,
}: {
  disabled: boolean;
  prompt: string;
  onChange(event: ChangeEvent<HTMLTextAreaElement>): void;
}) => {
  const { t } = useTranslation();
  return (
    <Stack aria-labelledby="sam-prompt-tab" gap="2" id="sam-prompt-panel" role="tabpanel">
      <Field label={t('widgets.layers.selectObject.prompt')}>
        <Textarea
          aria-label={t('widgets.layers.selectObject.prompt')}
          autoComplete="off"
          disabled={disabled}
          minH="24"
          resize="vertical"
          value={prompt}
          onChange={onChange}
        />
      </Field>
      <Text color="fg.muted" fontSize="xs">
        {t('widgets.layers.selectObject.promptGuidance')}
      </Text>
    </Stack>
  );
};

const SamSaveItem = ({
  disabled,
  engine,
  target,
}: ToolOptionsComponentProps & { disabled: boolean; target: SelectObjectSaveTarget }) => {
  const { t } = useTranslation();
  return (
    <Menu.Item
      disabled={disabled}
      value={target}
      onClick={() => void engine.saveSelectObjectSession(target, makeImageDurable)}
    >
      <Menu.ItemText>{t(`widgets.layers.selectObject.saveAs_${target}`)}</Menu.ItemText>
    </Menu.Item>
  );
};

export const SamOptions = ({
  engine,
  isExternalInteractionLocked = false,
}: ToolOptionsComponentProps & { isExternalInteractionLocked?: boolean }) => {
  const { t } = useTranslation();
  const session = useSamSession(engine);
  if (!session) {
    return null;
  }

  const eligibility = getSamActionEligibility(session, isExternalInteractionLocked);
  const viewModel = getSamPanelViewModel(session);
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
        <Stack gap="3">
          <Box>
            <Heading fontSize="md" id="sam-operation-title">
              {t('widgets.layers.selectObject.title')}
            </Heading>
            <Text color="fg.muted" fontSize="xs" fontVariantNumeric="tabular-nums">
              {viewModel.sourceSummary}
            </Text>
          </Box>
          <Grid gap="2" templateColumns={{ base: '1fr', sm: 'repeat(2, minmax(0, 1fr))' }}>
            <SamSwitch
              checked={session.autoProcess}
              disabled={!eligibility.canEditInputs}
              label={t('widgets.layers.selectObject.autoProcess')}
              onChange={(value) => setBoolean('autoProcess', value)}
            />
            <SamSwitch
              checked={session.isolatedPreview}
              disabled={!eligibility.canEditInputs}
              label={t('widgets.layers.selectObject.isolatedPreview')}
              onChange={(value) => setBoolean('isolatedPreview', value)}
            />
          </Grid>
        </Stack>
      </CanvasOperationPanel.Header>
      <CanvasOperationPanel.Body>
        <Stack gap="4">
          <Stack gap="2">
            <HStack gap="1" role="tablist">
              <Button
                aria-controls="sam-visual-panel"
                aria-selected={session.input.type === 'visual'}
                disabled={!eligibility.canEditInputs}
                flex="1"
                id="sam-visual-tab"
                minH="10"
                role="tab"
                variant={session.input.type === 'visual' ? 'solid' : 'ghost'}
                onClick={() =>
                  engine.updateSelectObjectSession({
                    input: { bbox: null, excludePoints: [], includePoints: [], type: 'visual' },
                  })
                }
              >
                {t('widgets.layers.selectObject.visual')}
              </Button>
              <Button
                aria-controls="sam-prompt-panel"
                aria-selected={session.input.type === 'prompt'}
                disabled={!eligibility.canEditInputs}
                flex="1"
                id="sam-prompt-tab"
                minH="10"
                role="tab"
                variant={session.input.type === 'prompt' ? 'solid' : 'ghost'}
                onClick={() => engine.updateSelectObjectSession({ input: { prompt: '', type: 'prompt' } })}
              >
                {t('widgets.layers.selectObject.promptMode')}
              </Button>
            </HStack>
            <SamSwitch
              checked={session.invert}
              disabled={!eligibility.canEditInputs}
              label={t('widgets.layers.selectObject.invert')}
              onChange={(value) => setBoolean('invert', value)}
            />
          </Stack>
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
          <Stack borderTopWidth="1px" gap="3" pt="4">
            <Heading fontSize="sm">{t('widgets.layers.selectObject.modelAndRefinement')}</Heading>
            <Field label={t('widgets.layers.selectObject.model')}>
              <NativeSelect.Root disabled={isProcessing || !eligibility.canEditInputs} minH="10" size="sm">
                <NativeSelect.Field
                  aria-label={t('widgets.layers.selectObject.model')}
                  value={session.model}
                  onChange={(event) =>
                    engine.updateSelectObjectSession({ model: event.currentTarget.value as SamModel })
                  }
                >
                  <option value="segment-anything-2-large">{t('widgets.layers.selectObject.modelSam2Large')}</option>
                  <option value="segment-anything-huge">{t('widgets.layers.selectObject.modelHuge')}</option>
                </NativeSelect.Field>
                <NativeSelect.Indicator />
              </NativeSelect.Root>
            </Field>
            <SamSwitch
              checked={session.applyPolygonRefinement}
              disabled={isProcessing || !eligibility.canEditInputs}
              label={t('widgets.layers.selectObject.refine')}
              onChange={(value) => setBoolean('applyPolygonRefinement', value)}
            />
          </Stack>
        </Stack>
      </CanvasOperationPanel.Body>
      <CanvasOperationPanel.Feedback color={session.error ? 'fg.error' : 'fg.muted'} fontSize="xs">
        <SamProcessFeedback
          error={session.error}
          errorText={session.error ? t(getSamErrorTranslationKey(session.error.code)) : null}
          statusText={t(getSamStatusTranslationKey(session.status))}
        />
      </CanvasOperationPanel.Feedback>
      <CanvasOperationPanel.Footer>
        <Button
          disabled={!eligibility.canProcess}
          loading={isProcessing}
          minH="10"
          onClick={() => void engine.processSelectObjectSession()}
        >
          {t('widgets.layers.selectObject.process')}
        </Button>
        <Button
          disabled={!eligibility.canReset}
          minH="10"
          variant="outline"
          onClick={() => engine.resetSelectObjectSession()}
        >
          {t('widgets.layers.selectObject.reset')}
        </Button>
        <Button disabled={!eligibility.canApply} minH="10" onClick={() => void engine.applySelectObjectSession()}>
          {t('common.apply')}
        </Button>
        <Menu.Root>
          <Menu.Trigger asChild>
            <Button disabled={!eligibility.canSave} minH="10" variant="outline">
              {t('widgets.layers.selectObject.saveAs')} <ChevronDownIcon size={14} />
            </Button>
          </Menu.Trigger>
          <Portal>
            <Menu.Positioner>
              <MenuContent minW="11rem" py="1">
                {SAVE_TARGETS.map((target) => (
                  <SamSaveItem key={target} disabled={!eligibility.canSave} engine={engine} target={target} />
                ))}
              </MenuContent>
            </Menu.Positioner>
          </Portal>
        </Menu.Root>
        <Button
          disabled={!eligibility.canCancel}
          minH="10"
          variant="ghost"
          onClick={() => engine.cancelSelectObjectSession()}
        >
          {t('common.cancel')}
        </Button>
      </CanvasOperationPanel.Footer>
    </CanvasOperationPanel.Root>
  );
};
