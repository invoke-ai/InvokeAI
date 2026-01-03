import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import {
  Combobox,
  Flex,
  FormControl,
  FormLabel,
  IconButton,
  Input,
  Radio,
  Select,
  Switch,
  Text,
  Textarea,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { skipToken } from '@reduxjs/toolkit/query';
import { logger } from 'app/logging/logger';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { UploadImageIconButton } from 'common/hooks/useImageUploadButton';
import {
  canvasWorkflowIntegrationFieldValueChanged,
  canvasWorkflowIntegrationImageFieldSelected,
  selectCanvasWorkflowIntegrationFieldValues,
  selectCanvasWorkflowIntegrationSelectedImageFieldKey,
  selectCanvasWorkflowIntegrationSelectedWorkflowId,
} from 'features/controlLayers/store/canvasWorkflowIntegrationSlice';
import { DndImage } from 'features/dnd/DndImage';
import { ModelFieldCombobox } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ModelFieldCombobox';
import { $templates } from 'features/nodes/store/nodesSlice';
import type { NodeFieldElement } from 'features/nodes/types/workflow';
import { SCHEDULER_OPTIONS } from 'features/parameters/types/constants';
import { isParameterScheduler } from 'features/parameters/types/parameterSchemas';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import { modelConfigsAdapterSelectors, useGetModelConfigsQuery } from 'services/api/endpoints/models';
import { useGetWorkflowQuery } from 'services/api/endpoints/workflows';
import type { AnyModelConfig, ImageDTO } from 'services/api/types';

const log = logger('canvas-workflow-integration');

interface WorkflowFieldRendererProps {
  el: NodeFieldElement;
}

export const WorkflowFieldRenderer = memo(({ el }: WorkflowFieldRendererProps) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const selectedWorkflowId = useAppSelector(selectCanvasWorkflowIntegrationSelectedWorkflowId);
  const fieldValues = useAppSelector(selectCanvasWorkflowIntegrationFieldValues);
  const selectedImageFieldKey = useAppSelector(selectCanvasWorkflowIntegrationSelectedImageFieldKey);
  const templates = useStore($templates);

  const { data: workflow } = useGetWorkflowQuery(selectedWorkflowId!, {
    skip: !selectedWorkflowId,
  });

  // Load boards and models for BoardField and ModelIdentifierField
  const { data: boardsData } = useListAllBoardsQuery({ include_archived: true });
  const { data: modelsData, isLoading: isLoadingModels } = useGetModelConfigsQuery();

  const { fieldIdentifier } = el.data;
  const fieldKey = `${fieldIdentifier.nodeId}.${fieldIdentifier.fieldName}`;

  log.debug({ fieldIdentifier, fieldKey }, 'Rendering workflow field');

  // Get the node, field instance, and field template
  const { field, fieldTemplate } = useMemo(() => {
    if (!workflow?.workflow.nodes) {
      log.warn('No workflow nodes found');
      return { field: null, fieldTemplate: null };
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const foundNode = workflow.workflow.nodes.find((n: any) => n.data.id === fieldIdentifier.nodeId);
    if (!foundNode) {
      log.warn({ nodeId: fieldIdentifier.nodeId }, 'Node not found');
      return { field: null, fieldTemplate: null };
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const foundField = (foundNode.data as any).inputs[fieldIdentifier.fieldName];
    if (!foundField) {
      log.warn({ nodeId: fieldIdentifier.nodeId, fieldName: fieldIdentifier.fieldName }, 'Field not found in node');
      return { field: null, fieldTemplate: null };
    }

    // Get the field template from the invocation templates
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const nodeType = (foundNode.data as any).type;
    const template = templates[nodeType];
    if (!template) {
      log.warn({ nodeType }, 'No template found for node type');
      return { field: foundField, fieldTemplate: null };
    }

    const foundFieldTemplate = template.inputs[fieldIdentifier.fieldName];
    if (!foundFieldTemplate) {
      log.warn({ nodeType, fieldName: fieldIdentifier.fieldName }, 'Field template not found');
      return { field: foundField, fieldTemplate: null };
    }

    return { field: foundField, fieldTemplate: foundFieldTemplate };
  }, [workflow, fieldIdentifier, templates]);

  // Get the current value from Redux or fallback to field default
  const currentValue = useMemo(() => {
    if (fieldValues && fieldKey in fieldValues) {
      return fieldValues[fieldKey];
    }

    return field?.value ?? fieldTemplate?.default ?? '';
  }, [fieldValues, fieldKey, field, fieldTemplate]);

  // Get field type from the template
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fieldType = fieldTemplate ? (fieldTemplate as any).type?.name : null;

  const handleChange = useCallback(
    (value: unknown) => {
      dispatch(canvasWorkflowIntegrationFieldValueChanged({ fieldName: fieldKey, value }));
    },
    [dispatch, fieldKey]
  );

  const handleStringChange = useCallback(
    (e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      handleChange(e.target.value);
    },
    [handleChange]
  );

  const handleNumberChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const val = fieldType === 'IntegerField' ? parseInt(e.target.value, 10) : parseFloat(e.target.value);
      handleChange(isNaN(val) ? 0 : val);
    },
    [handleChange, fieldType]
  );

  const handleBooleanChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      handleChange(e.target.checked);
    },
    [handleChange]
  );

  const handleSelectChange = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      handleChange(e.target.value);
    },
    [handleChange]
  );

  // SchedulerField handlers
  const handleSchedulerChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isParameterScheduler(v?.value)) {
        return;
      }
      handleChange(v.value);
    },
    [handleChange]
  );

  const schedulerValue = useMemo(() => SCHEDULER_OPTIONS.find((o) => o.value === currentValue), [currentValue]);

  // BoardField handlers
  const handleBoardChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!v) {
        return;
      }
      const value = v.value === 'auto' || v.value === 'none' ? v.value : { board_id: v.value };
      handleChange(value);
    },
    [handleChange]
  );

  const boardOptions = useMemo<ComboboxOption[]>(() => {
    const _options: ComboboxOption[] = [
      { label: t('common.auto'), value: 'auto' },
      { label: `${t('common.none')} (${t('boards.uncategorized')})`, value: 'none' },
    ];
    if (boardsData) {
      for (const board of boardsData) {
        _options.push({
          label: board.board_name,
          value: board.board_id,
        });
      }
    }
    return _options;
  }, [boardsData, t]);

  const boardValue = useMemo(() => {
    const _value = currentValue;
    const autoOption = boardOptions[0];
    const noneOption = boardOptions[1];
    if (!_value || _value === 'auto') {
      return autoOption;
    }
    if (_value === 'none') {
      return noneOption;
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const boardId = typeof _value === 'object' ? (_value as any).board_id : _value;
    const boardOption = boardOptions.find((o) => o.value === boardId);
    return boardOption ?? autoOption;
  }, [currentValue, boardOptions]);

  const noOptionsMessage = useCallback(() => t('boards.noMatching'), [t]);

  // ModelIdentifierField handlers
  const handleModelChange = useCallback(
    (value: AnyModelConfig | null) => {
      if (!value) {
        return;
      }
      handleChange(value);
    },
    [handleChange]
  );

  const modelConfigs = useMemo(() => {
    if (!modelsData) {
      return EMPTY_ARRAY;
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ui_model_base = fieldTemplate ? (fieldTemplate as any)?.ui_model_base : null;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ui_model_type = fieldTemplate ? (fieldTemplate as any)?.ui_model_type : null;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ui_model_variant = fieldTemplate ? (fieldTemplate as any)?.ui_model_variant : null;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ui_model_format = fieldTemplate ? (fieldTemplate as any)?.ui_model_format : null;

    if (!ui_model_base && !ui_model_type) {
      return modelConfigsAdapterSelectors.selectAll(modelsData);
    }

    return modelConfigsAdapterSelectors.selectAll(modelsData).filter((config) => {
      if (ui_model_base && !ui_model_base.includes(config.base)) {
        return false;
      }
      if (ui_model_type && !ui_model_type.includes(config.type)) {
        return false;
      }
      if (ui_model_variant && 'variant' in config && config.variant && !ui_model_variant.includes(config.variant)) {
        return false;
      }
      if (ui_model_format && !ui_model_format.includes(config.format)) {
        return false;
      }
      return true;
    });
  }, [modelsData, fieldTemplate]);

  // ImageField handler
  const handleImageFieldSelect = useCallback(() => {
    dispatch(canvasWorkflowIntegrationImageFieldSelected({ fieldKey }));
  }, [dispatch, fieldKey]);

  if (!field || !fieldTemplate) {
    log.warn({ fieldIdentifier }, 'Field or template is null - not rendering');
    return null;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const label = (field as any)?.label || (fieldTemplate as any)?.title || fieldIdentifier.fieldName;

  // Log the entire field structure to understand its shape
  log.debug(
    { fieldType, label, currentValue, fieldStructure: field, fieldTemplateStructure: fieldTemplate },
    'Field info'
  );

  // ImageField - allow user to select which one receives the canvas image
  if (fieldType === 'ImageField') {
    return (
      <ImageFieldComponent
        label={label}
        fieldKey={fieldKey}
        currentValue={currentValue}
        selectedImageFieldKey={selectedImageFieldKey}
        fieldTemplate={fieldTemplate}
        handleImageFieldSelect={handleImageFieldSelect}
        handleChange={handleChange}
      />
    );
  }

  // Render different input types based on field type
  if (fieldType === 'StringField') {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const isTextarea = (fieldTemplate as any)?.ui_component === 'textarea';
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const isRequired = (fieldTemplate as any)?.required ?? false;

    if (isTextarea) {
      return (
        <FormControl isRequired={isRequired}>
          <FormLabel>{label}</FormLabel>
          <Textarea
            value={String(currentValue)}
            onChange={handleStringChange}
            placeholder={label}
            rows={3}
            isRequired={isRequired}
          />
        </FormControl>
      );
    }

    return (
      <FormControl isRequired={isRequired}>
        <FormLabel>{label}</FormLabel>
        <Input value={String(currentValue)} onChange={handleStringChange} placeholder={label} isRequired={isRequired} />
      </FormControl>
    );
  }

  if (fieldType === 'IntegerField' || fieldType === 'FloatField') {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const min = (fieldTemplate as any)?.minimum;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const max = (fieldTemplate as any)?.maximum;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const step = fieldType === 'IntegerField' ? 1 : ((fieldTemplate as any)?.multipleOf ?? 0.01);

    return (
      <FormControl>
        <FormLabel>{label}</FormLabel>
        <Flex gap={2} alignItems="center">
          <Input
            type="number"
            value={Number(currentValue)}
            onChange={handleNumberChange}
            min={min}
            max={max}
            step={step}
          />
        </Flex>
      </FormControl>
    );
  }

  if (fieldType === 'BooleanField') {
    return (
      <FormControl>
        <FormLabel>{label}</FormLabel>
        <Switch isChecked={Boolean(currentValue)} onChange={handleBooleanChange} />
      </FormControl>
    );
  }

  if (fieldType === 'EnumField') {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const options = (fieldTemplate as any)?.options ?? (fieldTemplate as any)?.ui_choice_labels ?? [];
    const optionsList = Array.isArray(options) ? options : Object.keys(options);

    return (
      <FormControl>
        <FormLabel>{label}</FormLabel>
        <Select value={String(currentValue)} onChange={handleSelectChange}>
          {optionsList.map((option: string) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </Select>
      </FormControl>
    );
  }

  if (fieldType === 'SchedulerField') {
    return (
      <FormControl>
        <FormLabel>{label}</FormLabel>
        <Combobox value={schedulerValue} options={SCHEDULER_OPTIONS} onChange={handleSchedulerChange} />
      </FormControl>
    );
  }

  if (fieldType === 'BoardField') {
    return (
      <FormControl>
        <FormLabel>{label}</FormLabel>
        <Combobox
          value={boardValue}
          options={boardOptions}
          onChange={handleBoardChange}
          placeholder={t('boards.selectBoard')}
          noOptionsMessage={noOptionsMessage}
        />
      </FormControl>
    );
  }

  if (fieldType === 'ModelIdentifierField') {
    return (
      <FormControl>
        <FormLabel>{label}</FormLabel>
        <ModelFieldCombobox
          value={currentValue}
          modelConfigs={modelConfigs}
          isLoadingConfigs={isLoadingModels}
          onChange={handleModelChange}
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          required={(fieldTemplate as any).required}
          groupByType
        />
      </FormControl>
    );
  }

  // For other field types, show a read-only message
  log.warn(`Unsupported field type "${fieldType}" for field "${label}" - showing as read-only`);
  return (
    <FormControl>
      <FormLabel>{label}</FormLabel>
      <Input value={`${fieldType} (read-only)`} isReadOnly />
    </FormControl>
  );
});

WorkflowFieldRenderer.displayName = 'WorkflowFieldRenderer';

// Separate component for ImageField to avoid conditional hooks
interface ImageFieldComponentProps {
  label: string;
  fieldKey: string;
  currentValue: unknown;
  selectedImageFieldKey: string | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  fieldTemplate: any;
  handleImageFieldSelect: () => void;
  handleChange: (value: unknown) => void;
}

const ImageFieldComponent = memo(
  ({
    label,
    fieldKey,
    currentValue,
    selectedImageFieldKey,
    fieldTemplate,
    handleImageFieldSelect,
    handleChange,
  }: ImageFieldComponentProps) => {
    const { t } = useTranslation();

    const isSelected = selectedImageFieldKey === fieldKey;

    // Get image from field values (uploaded image) or from workflow field (default/saved image)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const imageValue = currentValue as any;
    const imageName = imageValue?.image_name;

    const { currentData: imageDTO } = useGetImageDTOQuery(imageName ?? skipToken);

    const handleImageUpload = useCallback(
      (uploadedImage: ImageDTO) => {
        handleChange(uploadedImage);
      },
      [handleChange]
    );

    const handleImageClear = useCallback(() => {
      handleChange(undefined);
    }, [handleChange]);

    return (
      <FormControl>
        <Flex alignItems="center" gap={2} mb={2}>
          <Radio isChecked={isSelected} onChange={handleImageFieldSelect} />
          <FormLabel mb={0} cursor="pointer" onClick={handleImageFieldSelect}>
            {label}
          </FormLabel>
        </Flex>
        <Text fontSize="sm" color="base.400" ml={6} mb={2}>
          {isSelected
            ? t('controlLayers.workflowIntegration.imageFieldSelected', 'This field will receive the canvas image')
            : t('controlLayers.workflowIntegration.imageFieldNotSelected', 'Click to use this field for canvas image')}
        </Text>

        {/* Show image upload/preview for non-selected fields */}
        {!isSelected && (
          <Flex ml={6} position="relative" w="full" h={32} alignItems="stretch">
            {!imageDTO && (
              <UploadImageIconButton
                w="full"
                h="auto"
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                isError={(fieldTemplate as any)?.required && !imageValue}
                onUpload={handleImageUpload}
                fontSize={24}
              />
            )}
            {imageDTO && (
              <Flex gap={2} alignItems="center">
                <Flex borderRadius="base" borderWidth={1} borderStyle="solid" overflow="hidden" position="relative">
                  <DndImage imageDTO={imageDTO} asThumbnail />
                  <Text
                    position="absolute"
                    background="base.900"
                    color="base.50"
                    fontSize="sm"
                    fontWeight="semibold"
                    insetInlineEnd={1}
                    insetBlockEnd={1}
                    opacity={0.7}
                    px={2}
                    borderRadius="base"
                    pointerEvents="none"
                  >{`${imageDTO.width}x${imageDTO.height}`}</Text>
                </Flex>
                <IconButton
                  aria-label={t('common.clearImage', 'Clear image')}
                  icon={<PiTrashSimpleBold />}
                  onClick={handleImageClear}
                  size="sm"
                  variant="ghost"
                  colorScheme="error"
                />
              </Flex>
            )}
          </Flex>
        )}
      </FormControl>
    );
  }
);

ImageFieldComponent.displayName = 'ImageFieldComponent';
