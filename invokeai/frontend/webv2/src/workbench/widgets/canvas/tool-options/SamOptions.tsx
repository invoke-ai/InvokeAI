import type {
  OperationPropertyForm,
  OperationStatus,
  ToolFormProps,
} from '@workbench/widgets/canvas/tool-presentation/toolFormContracts';
/* oxlint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-array-as-prop */
import type { KeyboardEvent } from 'react';

import { createListCollection, Input, Text, VisuallyHidden } from '@chakra-ui/react';
import { isImeComposing } from '@platform/browser/imeComposition';
import { Button } from '@platform/ui/Button';
import { Select } from '@platform/ui/Select';
import {
  getCanvasOperations,
  isSamDocumentInputValid,
  type CanvasOperationCapability,
  type SamSessionErrorCode,
  type SamSessionSnapshot,
  type SelectObjectSaveTarget,
} from '@workbench/canvas-operations/api';
import { patchWorkbenchPreferences } from '@workbench/settings/store';
import { useSamSession } from '@workbench/widgets/canvas/engineStoreHooks';
import {
  PropertyControlRow,
  PropertySegmentedRow,
  PropertySwitchRow,
} from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import { useId, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { OperationActions } from './OperationActions';

export interface SamActionEligibility {
  canApply: boolean;
  canCancel: boolean;
  canEditInputs: boolean;
  canProcess: boolean;
  canReset: boolean;
  canSave: boolean;
}

type SamStatus = SamSessionSnapshot['status'];

/** The phases shown while work is in flight; Ready and the error state show none. */
const SAM_BUSY_TRANSLATION_KEYS: Partial<Record<SamStatus, string>> = {
  committing: 'widgets.layers.selectObject.statusCommitting',
  'preparing-source': 'widgets.layers.selectObject.statusPreparingSource',
  'processing-sam': 'widgets.layers.selectObject.statusProcessingSam',
  'rendering-preview': 'widgets.layers.selectObject.statusRenderingPreview',
  scheduled: 'widgets.layers.selectObject.statusScheduled',
  uploading: 'widgets.layers.selectObject.statusUploading',
};

const SAM_ERROR_TRANSLATION_KEYS: Record<SamSessionErrorCode, string> = {
  'adjustment-error': 'widgets.layers.selectObject.errorAdjustment',
  'adjustment-over-budget': 'widgets.layers.selectObject.errorAdjustmentOverBudget',
  'adjustment-too-large': 'widgets.layers.selectObject.errorAdjustmentTooLarge',
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
  unreadable: 'widgets.layers.selectObject.errorUnreadable',
  upload: 'widgets.layers.selectObject.errorUpload',
};

const SAVE_TARGETS: readonly SelectObjectSaveTarget[] = [
  'selection',
  'raster',
  'control',
  'inpaint_mask',
  'regional_guidance',
];

const isSamProcessingStatus = (status: SamStatus): boolean =>
  status === 'preparing-source' ||
  status === 'uploading' ||
  status === 'processing-sam' ||
  status === 'rendering-preview';

export const getSamBusyTranslationKey = (status: SamStatus): string | null => SAM_BUSY_TRANSLATION_KEYS[status] ?? null;

export const getSamErrorTranslationKey = (code: SamSessionErrorCode): string => SAM_ERROR_TRANSLATION_KEYS[code];

/** Legacy parity: canvas adoption keeps the SAM result intermediate and out of the gallery. */
export const keepSamImageIntermediate = (_imageName: string): Promise<void> => Promise.resolve();

export const getSamActionHandlers = (operations: CanvasOperationCapability) => ({
  apply: () => void operations.applySelectObjectSession(keepSamImageIntermediate),
  cancel: () => operations.cancelSelectObjectSession(),
  process: () => void operations.processSelectObjectSession(),
  reset: () => operations.resetSelectObjectSession(),
  save: (target: SelectObjectSaveTarget) => void operations.saveSelectObjectSession(target, keepSamImageIntermediate),
});

/** Auto-update is also the account's preference, so the next session starts from the person's last choice. */
export const toggleSamSetting = (
  operations: Pick<CanvasOperationCapability, 'updateSelectObjectSession'>,
  key: 'applyPolygonRefinement' | 'autoProcess' | 'isolatedPreview',
  value: boolean
): void => {
  const result = operations.updateSelectObjectSession({ [key]: value });
  if (key === 'autoProcess' && result === 'updated') {
    void patchWorkbenchPreferences({ selectObjectAutoUpdate: value });
  }
};

export const getSamActionEligibility = (session: SamSessionSnapshot): SamActionEligibility => {
  const isProcessing = isSamProcessingStatus(session.status);
  const actionsBlocked = session.status === 'committing';
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

const PointCount = ({ children }: { children: string }) => (
  <Text as="span" fontVariantNumeric="tabular-nums">
    {children}
  </Text>
);

/** What to select: points and a box drawn on the canvas, or a prompt; plus invert. */
const SamInputSettings = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const session = useSamSession(engine);
  const operations = getCanvasOperations(engine);
  const promptLabelId = useId();
  const gestureGuidanceId = useId();
  if (!session) {
    return null;
  }
  const eligibility = getSamActionEligibility(session);
  const disabled = !eligibility.canEditInputs;
  const { input } = session;
  const runOnEnter = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' && !isImeComposing(event.nativeEvent) && eligibility.canProcess) {
      event.preventDefault();
      void operations.processSelectObjectSession();
    }
  };
  return (
    <>
      <PropertySegmentedRow
        disabled={disabled}
        label={t('widgets.layers.selectObject.mode')}
        options={[
          { label: t('widgets.layers.selectObject.visual'), value: 'visual' },
          { label: t('widgets.layers.selectObject.promptMode'), value: 'prompt' },
        ]}
        value={input.type}
        onValueChange={(inputType) => operations.updateSelectObjectSession({ inputType })}
      />
      {input.type === 'prompt' ? (
        <PropertyControlRow label={t('widgets.layers.selectObject.prompt')} labelId={promptLabelId}>
          <Input
            aria-labelledby={promptLabelId}
            autoComplete="off"
            disabled={disabled}
            gridColumn="2 / -1"
            placeholder={t('widgets.layers.selectObject.promptGuidance')}
            value={input.prompt}
            onChange={(event) =>
              operations.updateSelectObjectSession({ input: { prompt: event.currentTarget.value, type: 'prompt' } })
            }
            onKeyDown={runOnEnter}
          />
        </PropertyControlRow>
      ) : (
        <>
          {/* The canvas shows these gestures as hints; this keeps them with the control for assistive tech. */}
          <VisuallyHidden id={gestureGuidanceId}>{t('widgets.layers.selectObject.visualGuidance')}</VisuallyHidden>
          <PropertySegmentedRow
            describedBy={gestureGuidanceId}
            disabled={disabled}
            label={t('widgets.layers.selectObject.pointType')}
            options={[
              {
                label: (
                  <PointCount>
                    {t('widgets.layers.selectObject.includeCount', { count: input.includePoints.length })}
                  </PointCount>
                ),
                value: 'include',
              },
              {
                label: (
                  <PointCount>
                    {t('widgets.layers.selectObject.excludeCount', { count: input.excludePoints.length })}
                  </PointCount>
                ),
                value: 'exclude',
              },
            ]}
            value={session.pointLabel}
            onValueChange={(pointLabel) => operations.updateSelectObjectSession({ pointLabel })}
          />
          <PropertyControlRow label={t('widgets.layers.selectObject.bbox')}>
            <Text color={input.bbox ? undefined : 'fg.muted'} minW="0" truncate>
              {input.bbox ? t('widgets.layers.selectObject.bboxActive') : t('widgets.layers.selectObject.bboxInactive')}
            </Text>
            {/* Stays mounted, disabled without a box, so setting or clearing a box never reflows the row. */}
            <Button
              aria-label={t('widgets.layers.selectObject.clearBox')}
              disabled={disabled || !input.bbox}
              variant="ghost"
              onClick={() => operations.updateSelectObjectSession({ input: { ...input, bbox: null } })}
            >
              {t('common.clear')}
            </Button>
          </PropertyControlRow>
        </>
      )}
      <PropertySwitchRow
        checked={session.invert}
        disabled={disabled}
        label={t('widgets.layers.selectObject.invert')}
        onCheckedChange={(invert) => operations.updateSelectObjectSession({ invert })}
      />
    </>
  );
};

/** Set-once session settings: model, refinement, preview behavior. */
const SamSettingsGroup = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const session = useSamSession(engine);
  const operations = getCanvasOperations(engine);
  const modelLabelId = useId();
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
  if (!session) {
    return null;
  }
  const eligibility = getSamActionEligibility(session);
  const settingsLocked = !eligibility.canEditInputs;
  const processingLocked = settingsLocked || isSamProcessingStatus(session.status);
  return (
    <>
      <PropertyControlRow label={t('widgets.layers.selectObject.model')} labelId={modelLabelId}>
        <Select
          collection={modelCollection}
          disabled={processingLocked}
          gridColumn="2 / -1"
          ids={{ label: modelLabelId }}
          value={[session.model]}
          onValueChange={({ value }) => {
            const model = value[0];
            if (model === 'segment-anything-2-large' || model === 'segment-anything-huge') {
              operations.updateSelectObjectSession({ model });
            }
          }}
        />
      </PropertyControlRow>
      <PropertySwitchRow
        checked={session.applyPolygonRefinement}
        disabled={processingLocked}
        label={t('widgets.layers.selectObject.refine')}
        onCheckedChange={(value) => toggleSamSetting(operations, 'applyPolygonRefinement', value)}
      />
      <PropertySwitchRow
        checked={session.autoProcess}
        disabled={settingsLocked}
        label={t('widgets.layers.selectObject.autoProcess')}
        onCheckedChange={(value) => toggleSamSetting(operations, 'autoProcess', value)}
      />
      <PropertySwitchRow
        checked={session.isolatedPreview}
        disabled={settingsLocked}
        label={t('widgets.layers.selectObject.isolatedPreview')}
        onCheckedChange={(value) => toggleSamSetting(operations, 'isolatedPreview', value)}
      />
    </>
  );
};

const useSamStatus = (engine: ToolFormProps['engine']): OperationStatus | null => {
  const { t } = useTranslation();
  const session = useSamSession(engine);
  if (!session) {
    return null;
  }
  const busyKey = getSamBusyTranslationKey(session.status);
  return {
    busyText: busyKey ? t(busyKey) : null,
    error: session.error
      ? { detail: session.error.detail ?? null, text: t(getSamErrorTranslationKey(session.error.code)) }
      : null,
    source: t('widgets.layers.selectObject.sourceLayerLabel', {
      height: session.sourceRect.height,
      name: session.layerName,
      type: t(`widgets.layers.selectObject.saveAs_${session.layerType}`),
      width: session.sourceRect.width,
    }),
  };
};

const SamActions = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const session = useSamSession(engine);
  const actions = useMemo(() => getSamActionHandlers(getCanvasOperations(engine)), [engine]);
  if (!session) {
    return null;
  }
  const eligibility = getSamActionEligibility(session);
  return (
    <OperationActions
      applying={session.status === 'committing'}
      canApply={eligibility.canApply}
      canCancel={eligibility.canCancel}
      canReset={eligibility.canReset}
      canSave={eligibility.canSave}
      run={
        session.autoProcess
          ? null
          : {
              disabled: !eligibility.canProcess,
              label: t('widgets.layers.selectObject.findObject'),
              loading: isSamProcessingStatus(session.status),
              onRun: actions.process,
            }
      }
      saveAsLabel={t('widgets.layers.selectObject.saveAs')}
      saveTargets={SAVE_TARGETS.map((target) => ({
        disabled: !eligibility.canSave,
        label: t(`widgets.layers.selectObject.saveAs_${target}`),
        value: `save-${target}`,
        onSelect: () => actions.save(target),
      }))}
      onApply={actions.apply}
      onCancel={actions.cancel}
      onReset={actions.reset}
    />
  );
};

export const selectObjectOperationForm: OperationPropertyForm = {
  actions: SamActions,
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
  useStatus: useSamStatus,
};
