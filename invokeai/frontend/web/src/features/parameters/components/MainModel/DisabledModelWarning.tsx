import { Flex, Link, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $accountSettingsLink } from 'app/store/nanostores/accountSettingsLink';
import { useAppSelector } from 'app/store/storeHooks';
import { selectModel } from 'features/controlLayers/store/paramsSlice';
import { useIsModelDisabled } from 'features/parameters/hooks/useIsModelDisabled';
import { Trans, useTranslation } from 'react-i18next';

export const DisabledModelWarning = () => {
  const { t } = useTranslation();
  const model = useAppSelector(selectModel);

  const accountSettingsLink = useStore($accountSettingsLink);
  const { isChatGPT4oHighModelDisabled } = useIsModelDisabled();

  if (!model || !isChatGPT4oHighModelDisabled(model)) {
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
