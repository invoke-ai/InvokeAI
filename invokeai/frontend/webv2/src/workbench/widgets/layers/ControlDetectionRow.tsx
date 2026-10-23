import type { SelectValueChangeDetails } from '@chakra-ui/react';

import { createListCollection, HStack, Icon, Spinner, Stack, Text, VisuallyHidden } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { Select } from '@platform/ui/Select';
import { getFilterDefinition } from '@workbench/canvas-operations/api';
import {
  setPropertyGroupCollapsed,
  usePropertyGroupCollapsed,
} from '@workbench/widgets/canvas/tool-presentation/propertyGroupStore';
import { PropertyControlRow } from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import { CircleAlertIcon, RotateCcwIcon, ScanSearchIcon, SlidersHorizontalIcon } from 'lucide-react';
import { useCallback, useId, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { DETECTION_NONE, formatPanelText, type ControlPanelModel } from './controlPanelModel';
import { LayerFilterControls } from './LayerFilterControls';
import { OptionNameSeparator } from './OptionNameSeparator';

const SELECT_POSITIONING = { placement: 'bottom-end', sameWidth: true } as const;
const SETTINGS_GROUP_ID = 'control-detection-settings';
const ACTION_SIZE = '7';

interface DetectionItem {
  disabled: boolean;
  hint: string | null;
  label: string;
  value: string;
}

const groupDetection = (item: DetectionItem): string => (item.value === DETECTION_NONE ? 'none' : 'detectors');
// Neither group needs a header: the detectors open the list and None follows a divider.
const noGroupLabel = (): null => null;
const ignoreFilterTypeChange = (): void => undefined;

/**
 * The Detection row: the type's detectors and "None: use the image as it is", a spinner while one runs, Run again
 * after a failure, Detect again for an out-of-date map, and the detector's settings inline under the row.
 */
export const ControlDetectionRow = ({
  disabled,
  model,
  onDetect,
  onDetectorChange,
  onSettingsChange,
}: {
  disabled: boolean;
  model: ControlPanelModel['detection'];
  onDetect(): void;
  onDetectorChange(detector: string | null): void;
  onSettingsChange(settings: Record<string, unknown>): void;
}) => {
  const { i18n, t } = useTranslation();
  const label = t('widgets.layers.control.rows.detection');
  const regionId = useId();
  const collapsed = usePropertyGroupCollapsed(SETTINGS_GROUP_ID, true);
  const format = useCallback(
    (text: Parameters<typeof formatPanelText>[1]) => formatPanelText(t, text, i18n.language),
    [i18n.language, t]
  );
  const detectorLabel = useCallback(
    (detector: string | null) =>
      detector ? t(`widgets.layers.control.filters.${detector}`, detector) : format(model.valueText),
    [format, model.valueText, t]
  );

  const collection = useMemo(
    () =>
      createListCollection<DetectionItem>({
        items: model.options.map((option) => ({
          disabled: option.disabled,
          hint: option.hint ? format(option.hint) : null,
          label: format(option.label),
          value: option.value,
        })),
      }),
    [format, model.options]
  );
  const value = useMemo(() => (model.value ? [model.value] : []), [model.value]);
  const handleChange = useCallback(
    ({ value: next }: SelectValueChangeDetails) => {
      const choice = next[0];
      if (choice && choice !== model.value) {
        onDetectorChange(choice === DETECTION_NONE ? null : choice);
      }
    },
    [model.value, onDetectorChange]
  );

  const hasParams = model.settings ? (getFilterDefinition(model.settings.detector)?.params.length ?? 0) > 0 : false;
  const settingsUnavailable = model.settingsUnavailable
    ? format(model.settingsUnavailable)
    : hasParams
      ? undefined
      : t('widgets.layers.control.detectionNoSettings', { detector: format(model.valueText) });
  const open = !collapsed && settingsUnavailable === undefined;
  const toggleSettings = useCallback(() => setPropertyGroupCollapsed(SETTINGS_GROUP_ID, !collapsed), [collapsed]);

  // The last run this row saw, so its end can be announced once the status settles.
  const [lastRun, setLastRun] = useState<string | null>(null);
  const running = model.busy?.detector ?? null;
  if (running !== null && running !== lastRun) {
    setLastRun(running);
  } else if (model.failed && lastRun !== null) {
    setLastRun(null);
  }
  const announcement = running
    ? t('widgets.layers.control.detecting', { detector: detectorLabel(running) })
    : model.failed
      ? t('widgets.layers.control.detectionFailed', {
          detector: format(model.valueText),
          reason: t(`widgets.layers.control.detectionErrors.${model.failed.code}`),
        })
      : lastRun
        ? t('widgets.layers.control.detectionDone', { detector: detectorLabel(lastRun) })
        : '';

  const valueText = useMemo(
    () => (
      <HStack as="span" gap="1.5" minW="0">
        {model.busy ? <Spinner boxSize="3" flexShrink={0} /> : null}
        {model.failed ? <Icon as={CircleAlertIcon} boxSize="3.5" color="fg.error" flexShrink={0} /> : null}
        <Text
          as="span"
          color={model.tone === 'muted' ? 'fg.muted' : model.tone === 'error' ? 'fg.error' : undefined}
          truncate
        >
          {format(model.valueText)}
        </Text>
      </HStack>
    ),
    [format, model.busy, model.failed, model.tone, model.valueText]
  );

  const actionLabel =
    model.action === 'retry'
      ? t('widgets.layers.control.detectionRetry')
      : model.action === 'detect'
        ? t('widgets.layers.control.detectionRun')
        : null;

  return (
    <>
      <PropertyControlRow label={label}>
        <Select
          aria-label={label}
          collection={collection}
          disabled={disabled || model.disabled}
          groupBy={groupDetection}
          minW="0"
          positioning={SELECT_POSITIONING}
          renderGroupLabel={noGroupLabel}
          renderItem={renderDetectionItem}
          value={value}
          valueText={valueText}
          onValueChange={handleChange}
        />
        <HStack gap="0.5">
          {actionLabel ? (
            <IconButton
              aria-label={actionLabel}
              disabled={disabled}
              minW={ACTION_SIZE}
              size="sm"
              tooltip={model.action === 'detect' ? t('widgets.layers.control.detectionStale') : actionLabel}
              variant="ghost"
              w={ACTION_SIZE}
              onClick={onDetect}
            >
              {model.action === 'retry' ? <RotateCcwIcon /> : <ScanSearchIcon />}
            </IconButton>
          ) : null}
          <IconButton
            aria-controls={open ? regionId : undefined}
            aria-expanded={open}
            aria-label={t('widgets.layers.control.detectionSettings')}
            minW={ACTION_SIZE}
            size="sm"
            tooltip={t('widgets.layers.control.detectionSettings')}
            unavailableReason={settingsUnavailable}
            variant={open ? 'subtle' : 'ghost'}
            w={ACTION_SIZE}
            onClick={toggleSettings}
          >
            <SlidersHorizontalIcon />
          </IconButton>
        </HStack>
      </PropertyControlRow>
      <VisuallyHidden aria-live="polite" role="status">
        {announcement}
      </VisuallyHidden>
      {open && model.settings ? (
        <Stack
          aria-label={t('widgets.layers.control.detectionSettings')}
          borderColor="border.subtle"
          borderLeftWidth="1px"
          gap="2"
          id={regionId}
          ms="1"
          ps="3"
          role="group"
        >
          <LayerFilterControls
            disabled={disabled}
            enumPresentation="segmented"
            filterType={model.settings.detector}
            focusFilter={false}
            parts="params"
            settings={model.settings.values}
            onFilterTypeChange={ignoreFilterTypeChange}
            onSettingsChange={onSettingsChange}
          />
        </Stack>
      ) : null}
    </>
  );
};

const renderDetectionItem = (item: DetectionItem) => (
  <Stack gap="0" minW="0">
    <Text truncate>
      {item.label}
      {item.hint ? <OptionNameSeparator /> : null}
    </Text>
    {item.hint ? (
      <Text color="fg.muted" fontSize="sm" truncate>
        {item.hint}
      </Text>
    ) : null}
  </Stack>
);
