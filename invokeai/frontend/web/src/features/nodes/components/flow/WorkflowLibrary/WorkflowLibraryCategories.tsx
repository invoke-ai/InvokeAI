import { ButtonGroup, Flex } from '@chakra-ui/react';
import IAIButton from 'common/components/IAIButton';
import { WorkflowCategory } from './types';
import { Dispatch, SetStateAction, memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  category: WorkflowCategory;
  setCategory: Dispatch<SetStateAction<WorkflowCategory>>;
};

const WorkflowLibraryCategories = ({ category, setCategory }: Props) => {
  const { t } = useTranslation();
  const handleClickUser = useCallback(() => {
    setCategory('user');
  }, [setCategory]);
  const handleClickSystem = useCallback(() => {
    setCategory('system');
  }, [setCategory]);
  return (
    <Flex layerStyle="second" p={2} borderRadius="base">
      <ButtonGroup orientation="vertical">
        <IAIButton
          onClick={handleClickUser}
          variant={category === 'user' ? 'invokeAI' : 'ghost'}
        >
          {t('workflows.userCategory')}
        </IAIButton>
        <IAIButton
          onClick={handleClickSystem}
          variant={category === 'system' ? 'invokeAI' : 'ghost'}
        >
          {t('workflows.systemCategory')}
        </IAIButton>
      </ButtonGroup>
    </Flex>
  );
};

export default memo(WorkflowLibraryCategories);
