import { Flex, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { FormElementEditModeWrapper } from 'features/nodes/components/sidePanel/builder/FormElementEditModeWrapper';
import { selectWorkflowFormMode, useElement } from 'features/nodes/store/workflowSlice';
import type { TextElement } from 'features/nodes/types/workflow';
import { isTextElement, TEXT_CLASS_NAME } from 'features/nodes/types/workflow';
import { memo } from 'react';

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
      <Text fontSize={fontSize}>{content}</Text>
    </Flex>
  );
});
TextElementComponentViewMode.displayName = 'TextElementComponentViewMode';

export const TextElementComponentEditMode = memo(({ el }: { el: TextElement }) => {
  const { id, data } = el;
  const { content, fontSize } = data;

  return (
    <FormElementEditModeWrapper element={el}>
      <Flex id={id} className={TEXT_CLASS_NAME}>
        <Text fontSize={fontSize}>{content}</Text>
      </Flex>
    </FormElementEditModeWrapper>
  );
});
TextElementComponentEditMode.displayName = 'TextElementComponentEditMode';
