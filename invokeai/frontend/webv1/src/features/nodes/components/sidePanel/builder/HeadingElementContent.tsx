import type { HeadingProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { Text } from '@invoke-ai/ui-library';
import { linkifyOptions, linkifySx } from 'common/components/linkify';
import Linkify from 'linkify-react';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const headingSx: SystemStyleObject = {
  fontWeight: 'bold',
  fontSize: '2xl',
  whiteSpace: 'pre-wrap',
  '&[data-is-empty="true"]': {
    opacity: 0.3,
  },
  ...linkifySx,
};

export const HeadingElementContent = memo(({ content, ...rest }: { content: string } & HeadingProps) => {
  const { t } = useTranslation();
  return (
    <Text sx={headingSx} data-is-empty={content === ''} {...rest}>
      <Linkify options={linkifyOptions}>{content || t('workflows.builder.headingPlaceholder')}</Linkify>
    </Text>
  );
});

HeadingElementContent.displayName = 'HeadingElementContent';
