import { Flex, Heading, Link, Spacer, Text } from '@invoke-ai/ui-library';
import { useIsModelManagerEnabled } from 'features/modelManagerV2/hooks/useIsModelManagerEnabled';
import ModelImageUpload from 'features/modelManagerV2/subpanels/ModelPanel/Fields/ModelImageUpload';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import type { AnyModelConfigWithExternal } from 'services/api/types';

const isSafeUrl = (url: string): boolean => {
  return url.startsWith('https://') || url.startsWith('http://');
};

type Props = PropsWithChildren<{
  modelConfig: AnyModelConfigWithExternal;
}>;

export const ModelHeader = memo(({ modelConfig, children }: Props) => {
  const { t } = useTranslation();
  const canManageModels = useIsModelManagerEnabled();

  return (
    <Flex alignItems="flex-start" gap={4}>
      {canManageModels && <ModelImageUpload model_key={modelConfig.key} model_image={modelConfig.cover_image} />}
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
        {'source_url' in modelConfig && modelConfig.source_url && isSafeUrl(modelConfig.source_url) && (
          <Text variant="subtext" noOfLines={1} wordBreak="break-all">
            {t('modelManager.sourceUrl')}:{' '}
            <Link href={modelConfig.source_url} isExternal color="invokeBlue.300">
              {modelConfig.source_url}
            </Link>
          </Text>
        )}
        <Text noOfLines={3}>{modelConfig.description}</Text>
      </Flex>
    </Flex>
  );
});

ModelHeader.displayName = 'ModelHeader';
