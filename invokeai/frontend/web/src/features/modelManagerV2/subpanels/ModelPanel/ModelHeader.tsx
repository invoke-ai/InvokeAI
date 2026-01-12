import { Flex, Heading, Spacer, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import ModelImageUpload from 'features/modelManagerV2/subpanels/ModelPanel/Fields/ModelImageUpload';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import type { AnyModelConfig } from 'services/api/types';

type Props = PropsWithChildren<{
  modelConfig: AnyModelConfig;
}>;

export const ModelHeader = memo(({ modelConfig, children }: Props) => {
  const { t } = useTranslation();
  const user = useAppSelector(selectCurrentUser);
  const isAdmin = user?.is_admin ?? false;

  return (
    <Flex alignItems="flex-start" gap={4}>
      {isAdmin && <ModelImageUpload model_key={modelConfig.key} model_image={modelConfig.cover_image} />}
      <Flex flexDir="column" gap={1} flexGrow={1} minW={0}>
        <Flex gap={2}>
          <Heading as="h2" fontSize="lg" noOfLines={1} wordBreak="break-all">
            {modelConfig.name}
          </Heading>
          <Spacer />
          {children}
        </Flex>
        {modelConfig.source && (
          <Text variant="subtext" noOfLines={1} wordBreak="break-all">
            {t('modelManager.source')}: {modelConfig.source}
          </Text>
        )}
        <Text noOfLines={3}>{modelConfig.description}</Text>
      </Flex>
    </Flex>
  );
});

ModelHeader.displayName = 'ModelHeader';
