import { Flex, Heading, type SystemStyleObject } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import type { AnyModelConfig } from 'services/api/types';

import { ModelConvertButton } from './ModelConvertButton';
import { ModelDeleteButton } from './ModelDeleteButton';
import { ModelEditButton } from './ModelEditButton';

const footerRowSx: SystemStyleObject = {
  justifyContent: 'space-between',
  alignItems: 'center',
  gap: 3,
  '&:not(:last-of-type)': {
    borderBottomWidth: '1px',
    borderBottomStyle: 'solid',
    borderBottomColor: 'border',
  },
  p: 3,
};

type Props = {
  modelConfig: AnyModelConfig;
  isEditing: boolean;
};

export const ModelFooter = memo(({ modelConfig, isEditing }: Props) => {
  const { t } = useTranslation();

  const shouldShowConvertOption = !isEditing && modelConfig.format === 'checkpoint' && modelConfig.type === 'main';

  return (
    <Flex flexDirection="column" borderWidth="1px" borderRadius="base">
      {shouldShowConvertOption && (
        <Flex sx={footerRowSx}>
          <Heading size="sm" color="base.100">
            {t('modelManager.convertToDiffusers')}
          </Heading>
          <Flex py={1}>
            <ModelConvertButton modelConfig={modelConfig} />
          </Flex>
        </Flex>
      )}
      {!isEditing && (
        <Flex sx={footerRowSx}>
          <Heading size="sm" color="base.100">
            {t('modelManager.edit')}
          </Heading>
          <Flex py={1}>
            <ModelEditButton />
          </Flex>
        </Flex>
      )}
      <Flex sx={footerRowSx}>
        <Heading size="sm" color="error.200">
          {t('modelManager.deleteModel')}
        </Heading>
        <Flex py={1}>
          <ModelDeleteButton modelConfig={modelConfig} />
        </Flex>
      </Flex>
    </Flex>
  );
});

ModelFooter.displayName = 'ModelFooter';
