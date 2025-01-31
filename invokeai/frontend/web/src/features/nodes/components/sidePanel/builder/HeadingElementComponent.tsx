import { Flex, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { AutosizeTextarea } from 'features/nodes/components/sidePanel/builder/AutosizeTextarea';
import { FormElementEditModeWrapper } from 'features/nodes/components/sidePanel/builder/FormElementEditModeWrapper';
import { formElementHeadingDataChanged, selectWorkflowFormMode, useElement } from 'features/nodes/store/workflowSlice';
import type { HeadingElement } from 'features/nodes/types/workflow';
import { HEADING_CLASS_NAME, isHeadingElement } from 'features/nodes/types/workflow';
import type { ChangeEvent, KeyboardEvent } from 'react';
import { memo, useCallback, useRef, useState } from 'react';

export const HeadingElementComponent = memo(({ id }: { id: string }) => {
  const el = useElement(id);
  const mode = useAppSelector(selectWorkflowFormMode);

  if (!el || !isHeadingElement(el)) {
    return null;
  }

  if (mode === 'view') {
    return <HeadingElementComponentViewMode el={el} />;
  }

  // mode === 'edit'
  return <HeadingElementComponentEditMode el={el} />;
});

HeadingElementComponent.displayName = 'HeadingElementComponent';

export const HeadingElementComponentViewMode = memo(({ el }: { el: HeadingElement }) => {
  const { id, data } = el;
  const { content } = data;

  return (
    <Flex id={id} className={HEADING_CLASS_NAME}>
      <Text fontWeight="bold" fontSize="4xl">
        {content || 'Edit to add heading'}
      </Text>
    </Flex>
  );
});

HeadingElementComponentViewMode.displayName = 'HeadingElementComponentViewMode';

export const HeadingElementComponentEditMode = memo(({ el }: { el: HeadingElement }) => {
  const { id } = el;

  return (
    <FormElementEditModeWrapper element={el}>
      <Flex id={id} className={HEADING_CLASS_NAME} w="full">
        <EditableHeading el={el} />
      </Flex>
    </FormElementEditModeWrapper>
  );
});

HeadingElementComponentEditMode.displayName = 'HeadingElementComponentEditMode';

export const EditableHeading = memo(({ el }: { el: HeadingElement }) => {
  const dispatch = useAppDispatch();
  const { id, data } = el;
  const { content } = data;

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
    dispatch(formElementHeadingDataChanged({ id, changes: { content: trimmedContent } }));
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
      placeholder="Heading"
      value={localContent}
      onChange={onChange}
      onBlur={onBlur}
      onKeyDown={onKeyDown}
      variant="outline"
      overflowWrap="anywhere"
      w="full"
      minRows={1}
      maxRows={10}
      resize="none"
      p={2}
      fontWeight="bold"
      fontSize="4xl"
    />
  );
});

EditableHeading.displayName = 'EditableHeading';
