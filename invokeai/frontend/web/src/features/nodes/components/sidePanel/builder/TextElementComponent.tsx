import { Flex, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { AutosizeTextarea } from 'features/nodes/components/sidePanel/builder/AutosizeTextarea';
import { FormElementEditModeWrapper } from 'features/nodes/components/sidePanel/builder/FormElementEditModeWrapper';
import { formElementTextDataChanged, selectWorkflowFormMode, useElement } from 'features/nodes/store/workflowSlice';
import type { TextElement } from 'features/nodes/types/workflow';
import { isTextElement, TEXT_CLASS_NAME } from 'features/nodes/types/workflow';
import type { ChangeEvent, KeyboardEvent } from 'react';
import { memo, useCallback, useRef, useState } from 'react';

export const TextElementComponent = memo(({ id }: { id: string }) => {
  const el = useElement(id);
  const mode = useAppSelector(selectWorkflowFormMode);

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

export const TextElementComponentViewMode = memo(({ el }: { el: TextElement }) => {
  const { id, data } = el;
  const { content, fontSize } = data;

  return (
    <Flex id={id} className={TEXT_CLASS_NAME}>
      <Text fontSize={fontSize} overflowWrap="anywhere">
        {content || 'Edit to add text'}
      </Text>
    </Flex>
  );
});
TextElementComponentViewMode.displayName = 'TextElementComponentViewMode';

export const TextElementComponentEditMode = memo(({ el }: { el: TextElement }) => {
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

export const EditableText = memo(({ el }: { el: TextElement }) => {
  const dispatch = useAppDispatch();
  const { id, data } = el;
  const { content, fontSize } = data;
  const [localContent, setLocalContent] = useState(content);
  const ref = useRef<HTMLTextAreaElement>(null);

  const onChange = useCallback((e: ChangeEvent<HTMLTextAreaElement>) => {
    setLocalContent(e.target.value);
  }, []);

  const onBlur = useCallback(() => {
    const trimmedContent = localContent.trim();
    if (trimmedContent === content) {
      return;
    }
    setLocalContent(trimmedContent);
    dispatch(formElementTextDataChanged({ id, changes: { content: trimmedContent } }));
  }, [localContent, content, id, dispatch]);

  const onKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter') {
        onBlur();
      } else if (e.key === 'Escape') {
        setLocalContent(content);
      }
    },
    [content, onBlur]
  );

  return (
    <AutosizeTextarea
      ref={ref}
      placeholder="Text"
      value={localContent}
      onChange={onChange}
      onBlur={onBlur}
      onKeyDown={onKeyDown}
      fontSize={fontSize}
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
