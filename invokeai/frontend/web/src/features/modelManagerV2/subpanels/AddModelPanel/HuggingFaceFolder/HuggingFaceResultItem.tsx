import { Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';
import { useInstallModelMutation } from 'services/api/endpoints/models';

type Props = {
  result: string;
};
export const HuggingFaceResultItem = ({ result }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const [installModel] = useInstallModelMutation();

  const handleInstall = useCallback(() => {
    installModel({ source: result })
      .unwrap()
      .then((_) => {
        dispatch(
          addToast(
            makeToast({
              title: t('toast.modelAddedSimple'),
              status: 'success',
            })
          )
        );
      })
      .catch((error) => {
        if (error) {
          dispatch(
            addToast(
              makeToast({
                title: `${error.data.detail} `,
                status: 'error',
              })
            )
          );
        }
      });
  }, [installModel, result, dispatch, t]);

  return (
    <Flex alignItems="center" justifyContent="space-between" w="100%" gap={3}>
      <Flex fontSize="sm" flexDir="column">
        <Text fontWeight="semibold">{result.split('/').slice(-1)[0]}</Text>
        <Text variant="subtext" noOfLines={1} wordBreak="break-all">
          {result}
        </Text>
      </Flex>
      <IconButton aria-label={t('modelManager.install')} icon={<PiPlusBold />} onClick={handleInstall} size="sm" />
    </Flex>
  );
};
