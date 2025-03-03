import { useAppDispatch } from 'app/store/storeHooks';
import { useEditable } from 'common/hooks/useEditable';
import { AutosizeTextarea } from 'features/nodes/components/sidePanel/builder/AutosizeTextarea';
import { HeadingElementContent } from 'features/nodes/components/sidePanel/builder/HeadingElementContent';
import { formElementHeadingDataChanged } from 'features/nodes/store/workflowSlice';
import type { HeadingElement } from 'features/nodes/types/workflow';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

export const HeadingElementContentEditable = memo(({ el }: { el: HeadingElement }) => {
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
    return <HeadingElementContent content={editable.value} onDoubleClick={editable.startEditing} />;
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
      fontSize="2xl"
      _focusVisible={{ borderRadius: 'base', h: 'unset' }}
    />
  );
});

HeadingElementContentEditable.displayName = 'HeadingElementContentEditable';
