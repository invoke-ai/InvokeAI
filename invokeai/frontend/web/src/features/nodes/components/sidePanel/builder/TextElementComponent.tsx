import type { SystemStyleObject, TextProps } from '@invoke-ai/ui-library';
import { Flex, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEditable } from 'common/hooks/useEditable';
import { AutosizeTextarea } from 'features/nodes/components/sidePanel/builder/AutosizeTextarea';
import { FormElementEditModeWrapper } from 'features/nodes/components/sidePanel/builder/FormElementEditModeWrapper';
import { formElementTextDataChanged, selectWorkflowMode, useElement } from 'features/nodes/store/workflowSlice';
import type { TextElement } from 'features/nodes/types/workflow';
import { isTextElement, TEXT_CLASS_NAME } from 'features/nodes/types/workflow';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

export const TextElementComponent = memo(({ id }: { id: string }) => {
  const el = useElement(id);
  const mode = useAppSelector(selectWorkflowMode);

  if (!el || !isTextElement(el)) {
    return null;
  }

  if (mode === 'view') {
    return <TextElementComponentViewMode el={el} />;
  }

  // mode === 'edit'
  return <TextElementComponentEditMode el={el} />;
});
TextElementComponent.displayName = 'TextElementComponent';

const TextElementComponentViewMode = memo(({ el }: { el: TextElement }) => {
  const { id, data } = el;
  const { content } = data;

  return (
    <Flex id={id} className={TEXT_CLASS_NAME} w="full">
      <TextContentDisplay content={content} />
    </Flex>
  );
});
TextElementComponentViewMode.displayName = 'TextElementComponentViewMode';

const textSx: SystemStyleObject = {
  fontSize: 'md',
  overflowWrap: 'anywhere',
  '&[data-is-empty="true"]': {
    opacity: 0.3,
  },
};

const TextContentDisplay = memo(({ content, ...rest }: { content: string } & TextProps) => {
  const { t } = useTranslation();
  return (
    <Text sx={textSx} data-is-empty={content === ''} {...rest}>
      {content || t('workflows.builder.textPlaceholder')}
    </Text>
  );
});
TextContentDisplay.displayName = 'TextContentDisplay';

const TextElementComponentEditMode = memo(({ el }: { el: TextElement }) => {
  const { id } = el;

  return (
    <FormElementEditModeWrapper element={el}>
      <Flex id={id} className={TEXT_CLASS_NAME} w="full">
        <EditableText el={el} />
      </Flex>
    </FormElementEditModeWrapper>
  );
});
TextElementComponentEditMode.displayName = 'TextElementComponentEditMode';

const EditableText = memo(({ el }: { el: TextElement }) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { id, data } = el;
  const { content } = data;
  const ref = useRef<HTMLTextAreaElement>(null);

  const onChange = useCallback(
    (content: string) => {
      dispatch(formElementTextDataChanged({ id, changes: { content } }));
    },
    [dispatch, id]
  );

  const editable = useEditable({
    value: content,
    defaultValue: '',
    onChange,
    inputRef: ref,
  });

  if (!editable.isEditing) {
    return <TextContentDisplay content={editable.value} onDoubleClick={editable.startEditing} />;
  }

  return (
    <AutosizeTextarea
      ref={ref}
      placeholder={t('workflows.builder.textPlaceholder')}
      {...editable.inputProps}
      fontSize="md"
      variant="outline"
      overflowWrap="anywhere"
      w="full"
      minRows={1}
      maxRows={10}
      resize="none"
      p={2}
    />
  );
});

EditableText.displayName = 'EditableText';
