import { Badge, Flex, Tooltip } from '@chakra-ui/react';
import { FIELDS } from 'features/nodes/types/constants';
import { map } from 'lodash-es';
import { memo } from 'react';
import 'reactflow/dist/style.css';

const FieldTypeLegend = () => {
  return (
    <Flex sx={{ gap: 2, flexDir: 'column' }}>
      {map(FIELDS, ({ title, description, color }, key) => (
        <Tooltip key={key} label={description}>
          <Badge
            sx={{
              userSelect: 'none',
              color:
                parseInt(color.split('.')[1] ?? '0', 10) < 500
                  ? 'base.800'
                  : 'base.50',
              bg: color,
            }}
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
