import { Divider, Flex, useColorMode } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import { setAddNewModelUIOption } from 'features/ui/store/uiSlice';
import { useTranslation } from 'react-i18next';
import AddCheckpointModel from './AddModelsPanel/AddCheckpointModel';
import AddDiffusersModel from './AddModelsPanel/AddDiffusersModel';

export default function AddModelsPanel() {
  const addNewModelUIOption = useAppSelector(
    (state: RootState) => state.ui.addNewModelUIOption
  );

  const { colorMode } = useColorMode();

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  return (
    <Flex flexDirection="column" gap={4}>
      <Flex columnGap={4}>
        <IAIButton
          onClick={() => dispatch(setAddNewModelUIOption('ckpt'))}
          isChecked={addNewModelUIOption == 'ckpt'}
        >
          {t('modelManager.addCheckpointModel')}
        </IAIButton>
        <IAIButton
          onClick={() => dispatch(setAddNewModelUIOption('diffusers'))}
          isChecked={addNewModelUIOption == 'diffusers'}
        >
          {t('modelManager.addDiffuserModel')}
        </IAIButton>
      </Flex>

      <Divider />

      {addNewModelUIOption == 'ckpt' && <AddCheckpointModel />}
      {addNewModelUIOption == 'diffusers' && <AddDiffusersModel />}
    </Flex>
  );
}
