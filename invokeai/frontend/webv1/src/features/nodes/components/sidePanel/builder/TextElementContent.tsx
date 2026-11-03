import type { SystemStyleObject, TextProps } from '@invoke-ai/ui-library';
import { Text } from '@invoke-ai/ui-library';
import { linkifyOptions, linkifySx } from 'common/components/linkify';
import Linkify from 'linkify-react';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const textSx: SystemStyleObject = {
  fontSize: 'md',
  whiteSpace: 'pre-wrap',
  overflowWrap: 'anywhere',
  '&[data-is-empty="true"]': {
    opacity: 0.3,
  },
  ...linkifySx,
};

export const TextElementContent = memo(({ content, ...rest }: { content: string } & TextProps) => {
  const { t } = useTranslation();
  return (
    <Text sx={textSx} data-is-empty={content === ''} {...rest}>
      <Linkify options={linkifyOptions}>{content || t('workflows.builder.textPlaceholder')}</Linkify>
    </Text>
  );
});

TextElementContent.displayName = 'TextElementContent';
