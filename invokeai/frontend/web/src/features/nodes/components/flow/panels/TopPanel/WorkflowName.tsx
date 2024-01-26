import { Editable, EditableInput, EditablePreview, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const TopCenterPanel = () => {
  const { t } = useTranslation();
  const name = useAppSelector((s) => s.workflow.name);
  const isTouched = useAppSelector((s) => s.workflow.isTouched);
  const isWorkflowLibraryEnabled =
    useFeatureStatus('workflowLibrary').isFeatureEnabled;

  const displayName = useMemo(() => {
    let _displayName = name || t('workflows.unnamedWorkflow');

    return _displayName;
  }, [t, name, isTouched, isWorkflowLibraryEnabled]);

  return (
    <Editable defaultValue={displayName}>
      <EditablePreview
        m={2}
        fontSize="lg"
        noOfLines={1}
        wordBreak="break-all"
        fontWeight="semibold"
        opacity={0.8}
      />
      <EditableInput />
      {isTouched && isWorkflowLibraryEnabled ? (
        <Text
          m={2}
          fontSize="lg"
          userSelect="none"
          noOfLines={1}
          wordBreak="break-all"
          fontWeight="semibold"
          opacity={0.8}
        >
          ({t('common.unsaved')})
        </Text>
      ) : (
        ''
      )}
    </Editable>
    // <Text
    //   m={2}
    //   fontSize="lg"
    //   userSelect="none"
    //   noOfLines={1}
    //   wordBreak="break-all"
    //   fontWeight="semibold"
    //   opacity={0.8}
    // >
    //   <Editable
    //   {displayName}
    //   {isTouched && isWorkflowLibraryEnabled ? `(${t('common.unsaved')})` : ''}
    // </Text>
  );
};

export default memo(TopCenterPanel);
