import { Flex, Link, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $accountSettingsLink } from 'app/store/nanostores/accountSettingsLink';
import { useAppSelector } from 'app/store/storeHooks';
import { selectIsChatGTP4o, selectIsImagen3, selectModel } from 'features/controlLayers/store/paramsSlice';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { useMemo } from 'react';
import { Trans, useTranslation } from 'react-i18next';

export const DisabledModelWarning = () => {
  const { t } = useTranslation();
  const model = useAppSelector(selectModel);
  const isImagen3 = useAppSelector(selectIsImagen3);
  const isChatGPT4o = useAppSelector(selectIsChatGTP4o);
  const areApiModelsEnabled = useFeatureStatus('apiModels');
  const accountSettingsLink = useStore($accountSettingsLink);

  const isModelDisabled = useMemo(() => {
    return (isImagen3 || isChatGPT4o) && !areApiModelsEnabled;
  }, [isImagen3, isChatGPT4o, areApiModelsEnabled]);

  if (!isModelDisabled) {
    return null;
  }

  return (
    <Flex bg="error.500" borderRadius="base" padding={4} direction="column" fontSize="sm" gap={2}>
      <Text>
        <Trans
          i18nKey="parameters.modelDisabledForTrial"
          values={{
            modelName: model?.name,
          }}
          components={{
            LinkComponent: (
              <Link textDecor="underline" href={accountSettingsLink}>
                {t('parameters.invoke.accountSettings')}
              </Link>
            ),
          }}
        />
      </Text>
    </Flex>
  );
};
