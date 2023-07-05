import { Divider, Flex } from '@chakra-ui/react';
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

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  return (
    <Flex flexDirection="column" gap={4}>
      <Flex columnGap={4}>
        <IAIButton
          onClick={() => dispatch(setAddNewModelUIOption('ckpt'))}
          sx={{
            backgroundColor:
              addNewModelUIOption == 'ckpt' ? 'accent.700' : 'base.700',
            '&:hover': {
              backgroundColor:
                addNewModelUIOption == 'ckpt' ? 'accent.700' : 'base.600',
            },
          }}
        >
          {t('modelManager.addCheckpointModel')}
        </IAIButton>
        <IAIButton
          onClick={() => dispatch(setAddNewModelUIOption('diffusers'))}
          sx={{
            backgroundColor:
              addNewModelUIOption == 'diffusers' ? 'accent.700' : 'base.700',
            '&:hover': {
              backgroundColor:
                addNewModelUIOption == 'diffusers' ? 'accent.700' : 'base.600',
            },
          }}
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
