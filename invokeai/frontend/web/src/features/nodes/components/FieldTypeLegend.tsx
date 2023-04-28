import 'reactflow/dist/style.css';
import { Tooltip, Badge, Flex } from '@chakra-ui/react';
import { map } from 'lodash-es';
import { FIELDS } from '../types/constants';
import { memo } from 'react';

const FieldTypeLegend = () => {
  return (
    <Flex gap={2} flexDirection={{ base: 'column', xl: 'row' }}>
      {map(FIELDS, ({ title, description, color }, key) => (
        <Tooltip key={key} label={description}>
          <Badge
            colorScheme={color}
            sx={{ userSelect: 'none' }}
            textAlign="center"
          >
            {title}
          </Badge>
        </Tooltip>
      ))}
    </Flex>
  );
};

export default memo(FieldTypeLegend);
