import type { HeadingProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEditable } from 'common/hooks/useEditable';
import { AutosizeTextarea } from 'features/nodes/components/sidePanel/builder/AutosizeTextarea';
import { FormElementEditModeWrapper } from 'features/nodes/components/sidePanel/builder/FormElementEditModeWrapper';
import { formElementHeadingDataChanged, selectWorkflowMode, useElement } from 'features/nodes/store/workflowSlice';
import type { HeadingElement } from 'features/nodes/types/workflow';
import { HEADING_CLASS_NAME, isHeadingElement } from 'features/nodes/types/workflow';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

export const HeadingElementComponent = memo(({ id }: { id: string }) => {
  const el = useElement(id);
  const mode = useAppSelector(selectWorkflowMode);

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

const HeadingElementComponentViewMode = memo(({ el }: { el: HeadingElement }) => {
  const { id, data } = el;
  const { content } = data;

  return (
    <Flex id={id} className={HEADING_CLASS_NAME}>
      <HeadingContentDisplay content={content} />
    </Flex>
  );
});

HeadingElementComponentViewMode.displayName = 'HeadingElementComponentViewMode';

const HeadingElementComponentEditMode = memo(({ el }: { el: HeadingElement }) => {
  const { id } = el;

  return (
    <FormElementEditModeWrapper element={el}>
      <Flex id={id} className={HEADING_CLASS_NAME} w="full">
        <EditableHeading el={el} />
      </Flex>
    </FormElementEditModeWrapper>
  );
});

const FONT_SIZE = '2xl';

const headingSx: SystemStyleObject = {
  fontWeight: 'bold',
  fontSize: FONT_SIZE,
  '&[data-is-empty="true"]': {
    opacity: 0.3,
  },
};

const HeadingContentDisplay = memo(({ content, ...rest }: { content: string } & HeadingProps) => {
  const { t } = useTranslation();
  return (
    <Text sx={headingSx} data-is-empty={content === ''} {...rest}>
      {content || t('workflows.builder.headingPlaceholder')}
    </Text>
  );
});
HeadingContentDisplay.displayName = 'HeadingContentDisplay';

HeadingElementComponentEditMode.displayName = 'HeadingElementComponentEditMode';

const EditableHeading = memo(({ el }: { el: HeadingElement }) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { id, data } = el;
  const { content } = data;
  const ref = useRef<HTMLTextAreaElement>(null);

  const onChange = useCallback(
    (content: string) => {
      dispatch(formElementHeadingDataChanged({ id, changes: { content } }));
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
    return <HeadingContentDisplay content={editable.value} onDoubleClick={editable.startEditing} />;
  }

  return (
    <AutosizeTextarea
      ref={ref}
      placeholder={t('workflows.builder.headingPlaceholder')}
      {...editable.inputProps}
      variant="outline"
      overflowWrap="anywhere"
      w="full"
      minRows={1}
      maxRows={10}
      resize="none"
      p={1}
      px={2}
      fontWeight="bold"
      fontSize={FONT_SIZE}
      _focusVisible={{ borderRadius: 'base', h: 'unset' }}
    />
  );
});

EditableHeading.displayName = 'EditableHeading';
