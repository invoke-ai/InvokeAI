import type { FieldInputTemplate, NodeFieldFormElement, ProjectGraphState } from '@features/workflow/contracts';
import type {
  NodeFieldFormSettings,
  NodeFieldNumberSettings,
  NodeFieldSettingsIssue,
  NodeFieldStringOption,
  NodeFieldStringSettings,
} from '@features/workflow/utility';

import { HStack, Icon, Input, Popover, Portal, Stack, Text } from '@chakra-ui/react';
import { isSeedInputField } from '@features/workflow/graph';
import { useFocusedDraft } from '@features/workflow/ui/fields/NumericInput';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import {
  getDefaultNodeFieldFormSettings,
  getNodeFieldSettingsIssue,
  hasNodeFieldFormSettings,
  serializeNodeFieldFormSettings,
} from '@features/workflow/utility';
import { Button, IconButton } from '@platform/ui/Button';
import { PopoverContent } from '@platform/ui/Popover';
import { SegmentedControl } from '@platform/ui/SegmentedControl';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { PlusIcon, SlidersHorizontalIcon, XIcon } from 'lucide-react';
import { useCallback, useId, useMemo, type ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

import { useNodeFieldBinding } from './NodeFieldControl';

const POPOVER_POSITIONING = { placement: 'bottom-end' } as const;

const SETTINGS_ISSUE_KEYS = {
  inverted: 'widgets.workflow.formBuilder.invertedBounds',
  noTrack: 'widgets.workflow.formBuilder.sliderNeedsBounds',
  noValue: 'widgets.workflow.formBuilder.noValidValue',
  outOfRange: 'widgets.workflow.formBuilder.boundsOutsideNode',
  stepIgnored: 'widgets.workflow.formBuilder.stepOutsideNode',
} as const satisfies Record<NodeFieldSettingsIssue, string>;

const toOptionalNumber = (raw: string): number | undefined => {
  const parsed = Number(raw);

  return raw.trim() !== '' && Number.isFinite(parsed) ? parsed : undefined;
};

/** One authored bound: the text stays as typed while focused; a blank or unparseable entry clears the bound. */
const BoundInput = ({
  describedBy,
  label,
  onChange,
  placeholder,
  value,
}: {
  describedBy: string;
  label: string;
  onChange: (value: number | undefined) => void;
  placeholder: string;
  value: number | undefined;
}) => {
  const [draft, setDraft, clearDraft] = useFocusedDraft();
  const onInputChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      setDraft(event.currentTarget.value);
      onChange(toOptionalNumber(event.currentTarget.value));
    },
    [onChange, setDraft]
  );

  return (
    <Stack flex="1" gap="0.5" minW="0">
      <Text color="fg.subtle" fontSize="xs">
        {label}
      </Text>
      <Input
        aria-describedby={describedBy}
        aria-label={label}
        className="nodrag"
        fontVariantNumeric="tabular-nums"
        placeholder={placeholder}
        size="sm"
        type="number"
        value={draft ?? (value === undefined ? '' : String(value))}
        onBlur={clearDraft}
        onChange={onInputChange}
      />
    </Stack>
  );
};

const NumberSettingsBody = ({
  onChange,
  settings,
  template,
  templateBounds,
}: {
  onChange: (settings: NodeFieldFormSettings) => void;
  settings: NodeFieldNumberSettings;
  template: FieldInputTemplate;
  templateBounds: { max: string; min: string; step: string };
}) => {
  const { t } = useTranslation();
  const controlOptions = useMemo(
    () => [
      { label: t('widgets.workflow.formBuilder.controlNumberInput'), value: 'number-input' },
      { label: t('widgets.workflow.formBuilder.controlSlider'), value: 'slider' },
      { label: t('widgets.workflow.formBuilder.controlNumberInputAndSlider'), value: 'number-input-and-slider' },
    ],
    [t]
  );
  const onComponentChange = useCallback(
    (component: string) => onChange({ ...settings, component: component as NodeFieldNumberSettings['component'] }),
    [onChange, settings]
  );
  const onMinChange = useCallback((min: number | undefined) => onChange({ ...settings, min }), [onChange, settings]);
  const onMaxChange = useCallback((max: number | undefined) => onChange({ ...settings, max }), [onChange, settings]);
  const onStepChange = useCallback(
    (step: number | undefined) => onChange({ ...settings, step: step !== undefined && step > 0 ? step : undefined }),
    [onChange, settings]
  );
  const issue = getNodeFieldSettingsIssue(template, settings);
  const hintId = useId();

  return (
    <Stack gap="2">
      <SegmentedControl
        ariaLabel={t('widgets.workflow.formBuilder.control')}
        options={controlOptions}
        size="sm"
        value={settings.component}
        onChange={onComponentChange}
      />
      <HStack alignItems="start" gap="2">
        <BoundInput
          describedBy={hintId}
          label={t('widgets.workflow.formBuilder.minimum')}
          placeholder={templateBounds.min}
          value={settings.min}
          onChange={onMinChange}
        />
        <BoundInput
          describedBy={hintId}
          label={t('widgets.workflow.formBuilder.maximum')}
          placeholder={templateBounds.max}
          value={settings.max}
          onChange={onMaxChange}
        />
        <BoundInput
          describedBy={hintId}
          label={t('widgets.workflow.formBuilder.step')}
          placeholder={templateBounds.step}
          value={settings.step}
          onChange={onStepChange}
        />
      </HStack>
      <Text aria-live="polite" color={issue ? 'fg.warning' : 'fg.subtle'} fontSize="xs" id={hintId} lineHeight={1.3}>
        {t(issue ? SETTINGS_ISSUE_KEYS[issue] : 'widgets.workflow.formBuilder.boundsHint')}
      </Text>
    </Stack>
  );
};

const OptionRow = ({
  index,
  onChange,
  onRemove,
  option,
}: {
  index: number;
  onChange: (index: number, option: NodeFieldStringOption) => void;
  onRemove: (index: number) => void;
  option: NodeFieldStringOption;
}) => {
  const { t } = useTranslation();
  const onLabelChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => onChange(index, { ...option, label: event.currentTarget.value }),
    [index, onChange, option]
  );
  const onValueChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => onChange(index, { ...option, value: event.currentTarget.value }),
    [index, onChange, option]
  );
  const onRemoveClick = useCallback(() => onRemove(index), [index, onRemove]);
  const removeLabel = t('widgets.workflow.formBuilder.removeOption', { index: index + 1 });

  return (
    <HStack gap="1">
      <Input
        aria-label={t('widgets.workflow.formBuilder.optionLabel', { index: index + 1 })}
        className="nodrag"
        placeholder={t('widgets.workflow.formBuilder.optionLabelPlaceholder')}
        size="sm"
        value={option.label}
        onChange={onLabelChange}
      />
      <Input
        aria-label={t('widgets.workflow.formBuilder.optionValue', { index: index + 1 })}
        className="nodrag"
        fontFamily="mono"
        placeholder={t('widgets.workflow.formBuilder.optionValuePlaceholder')}
        size="sm"
        value={option.value}
        onChange={onValueChange}
      />
      <Tooltip content={removeLabel}>
        <IconButton aria-label={removeLabel} color="fg.subtle" size="sm" variant="ghost" onClick={onRemoveClick}>
          <Icon as={XIcon} boxSize="3" />
        </IconButton>
      </Tooltip>
    </HStack>
  );
};

const StringSettingsBody = ({
  onChange,
  settings,
}: {
  onChange: (settings: NodeFieldFormSettings) => void;
  settings: NodeFieldStringSettings;
}) => {
  const { t } = useTranslation();
  const controlOptions = useMemo(
    () => [
      { label: t('widgets.workflow.formBuilder.controlTextInput'), value: 'input' },
      { label: t('widgets.workflow.formBuilder.controlTextarea'), value: 'textarea' },
      { label: t('widgets.workflow.formBuilder.controlDropdown'), value: 'dropdown' },
    ],
    [t]
  );
  const onComponentChange = useCallback(
    (component: string) => onChange({ ...settings, component: component as NodeFieldStringSettings['component'] }),
    [onChange, settings]
  );
  const onOptionChange = useCallback(
    (index: number, option: NodeFieldStringOption) =>
      onChange({ ...settings, options: settings.options.map((current, i) => (i === index ? option : current)) }),
    [onChange, settings]
  );
  const onOptionRemove = useCallback(
    (index: number) => onChange({ ...settings, options: settings.options.filter((_, i) => i !== index) }),
    [onChange, settings]
  );
  const onAddOption = useCallback(
    () => onChange({ ...settings, options: [...settings.options, { label: '', value: '' }] }),
    [onChange, settings]
  );

  return (
    <Stack gap="2">
      <SegmentedControl
        ariaLabel={t('widgets.workflow.formBuilder.control')}
        options={controlOptions}
        size="sm"
        value={settings.component}
        onChange={onComponentChange}
      />
      {settings.component === 'dropdown' ? (
        <Stack gap="1">
          <Text color="fg.subtle" fontSize="xs">
            {t('widgets.workflow.formBuilder.dropdownOptions')}
          </Text>
          {settings.options.length === 0 ? (
            <Text color="fg.warning" fontSize="xs" lineHeight={1.3}>
              {t('widgets.workflow.formBuilder.noOptions')}
            </Text>
          ) : null}
          {settings.options.map((option, index) => (
            <OptionRow
              // Rows have no identity of their own; the index is the row.
              key={index}
              index={index}
              option={option}
              onChange={onOptionChange}
              onRemove={onOptionRemove}
            />
          ))}
          <Button size="sm" variant="outline" w="fit-content" onClick={onAddOption}>
            <Icon as={PlusIcon} boxSize="3" />
            {t('widgets.workflow.formBuilder.addOption')}
          </Button>
        </Stack>
      ) : null}
    </Stack>
  );
};

const formatBound = (value: number | null): string => (value === null ? '' : String(value));

/** The builder's per-field control settings: which control the form shows and, for numbers, its bounds. */
export const NodeFieldSettingsPopover = ({
  element,
  projectGraph,
}: {
  element: NodeFieldFormElement;
  projectGraph: ProjectGraphState;
}) => {
  const { t } = useTranslation();
  const { editGraph } = useProjectGraphCommands();
  const triggerIds = useTooltipTriggerIds();
  const { formSettings, template } = useNodeFieldBinding(element, projectGraph);
  const onSettingsChange = useCallback(
    (settings: NodeFieldFormSettings) => {
      if (template) {
        editGraph({
          elementId: element.id,
          settings: serializeNodeFieldFormSettings(settings, template, element.data.showShuffle, element.data.settings),
          type: 'setNodeFieldSettings',
        });
      }
    },
    [editGraph, element.data.settings, element.data.showShuffle, element.id, template]
  );
  const templateBounds = useMemo(
    () =>
      template
        ? {
            max: formatBound(template.maximum ?? template.exclusiveMaximum),
            min: formatBound(template.minimum ?? template.exclusiveMinimum),
            step: formatBound(template.multipleOf),
          }
        : { max: '', min: '', step: '' },
    [template]
  );

  if (!template || !hasNodeFieldFormSettings(template) || isSeedInputField(template)) {
    return null;
  }

  const settings = formSettings ?? getDefaultNodeFieldFormSettings(template);

  if (!settings) {
    return null;
  }

  const label = t('widgets.workflow.formBuilder.fieldSettings');

  return (
    <Popover.Root ids={triggerIds} lazyMount positioning={POPOVER_POSITIONING}>
      <Tooltip content={label} ids={triggerIds}>
        <Popover.Trigger asChild>
          <IconButton
            aria-label={label}
            className="nodrag"
            color={formSettings ? 'accent.solid' : 'fg.subtle'}
            size="sm"
            variant="ghost"
          >
            <Icon as={SlidersHorizontalIcon} boxSize="3" />
          </IconButton>
        </Popover.Trigger>
      </Tooltip>
      <Portal>
        <Popover.Positioner>
          <PopoverContent w="20rem">
            <Popover.Body p="2">
              <Stack gap="2">
                <Text color="fg.subtle" fontSize="xs" lineHeight={1.3}>
                  {t('widgets.workflow.formBuilder.fieldSettingsHint')}
                </Text>
                {settings.kind === 'number' ? (
                  <NumberSettingsBody
                    settings={settings}
                    template={template}
                    templateBounds={templateBounds}
                    onChange={onSettingsChange}
                  />
                ) : (
                  <StringSettingsBody settings={settings} onChange={onSettingsChange} />
                )}
              </Stack>
            </Popover.Body>
          </PopoverContent>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};
