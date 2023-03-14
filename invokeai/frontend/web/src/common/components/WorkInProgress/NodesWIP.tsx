import { Flex, Heading, Text, VStack } from '@chakra-ui/react';
import NodeAPITest from 'app/NodeAPITest';
import { useTranslation } from 'react-i18next';
import WorkInProgress from './WorkInProgress';

export default function NodesWIP() {
  const { t } = useTranslation();
  return (
    <WorkInProgress>
      <Flex
        sx={{
          flexDirection: 'column',
          w: '100%',
          h: '100%',
          gap: 4,
          textAlign: 'center',
        }}
      >
        <NodeAPITest />
      </Flex>
    </WorkInProgress>
  );
}
