import { useAppDispatch } from 'app/store/storeHooks';
import { useEditable } from 'common/hooks/useEditable';
import { AutosizeTextarea } from 'features/nodes/components/sidePanel/builder/AutosizeTextarea';
import { TextElementContent } from 'features/nodes/components/sidePanel/builder/TextElementContent';
import { formElementTextDataChanged } from 'features/nodes/store/workflowSlice';
import type { TextElement } from 'features/nodes/types/workflow';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

export const TextElementContentEditable = memo(({ el }: { el: TextElement }) => {
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
    return <TextElementContent content={editable.value} onDoubleClick={editable.startEditing} />;
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

TextElementContentEditable.displayName = 'TextElementContentEditable';
